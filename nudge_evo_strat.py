import google.generativeai as genai
from rate_limiter import RateLimiter, send_message_with_retry
import random
from utils.tools import *
import math
from create_prompts import create_nudge_prompt_file
import pandas as pd
from radon.raw import analyze
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from config import Config
from dotenv import load_dotenv


def get_init_population():
    init_population = []
    for i in range(1, population_size + 1):
        heuristic_file = os.path.join(heur_dir, "heuristics_gem_" + str(i) + ".txt")
        with open(heuristic_file, 'r') as f:
            heuristic_message = f.read()

        init_population.append(heuristic_message)
    return init_population


def extract_python_code(text):
    pattern = r'```python(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "No Python code found."


def extract_message(text):
    pattern = r"\\communication\{([^}]+)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ""


def init_gemini_model(policy_model_name="gemini-2.0-flash-thinking-exp-01-21", farm_model_name="gemini-2.0-flash-thinking-exp-01-21",
                      fix_model_name="gemini-2.0-flash-thinking-exp-01-21"):
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variables
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("No API key found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")
    
    genai.configure(api_key=api_key)
    policy_system_instructions = ("You are an expert in land use policy, communication, incentives, and economics. "
                                  "Your task is to come up with the best message to be communicated to the farmers "
                                  "so as to change their behaviour from one set of heuristics to another set of heuristics. "
                                  "Provide your final message to the farmer in this format \communication{message}. \n\n")

    farm_system_instructions = ("You are a farmer who currently follows a given set of python heuristics. "
                                "Your task is to respond to communication from policy professionals by altering "
                                "your heuristics in an appropriate way. You will return final answers in python code."
                                "You should keep all data loading from input.geojson and dumping to output.geojson "
                                "the exact same, and just alter the logic depending on what your context is "
                                "and what the message is. Don't invent new variable names in the output, keep "
                                "using margin_intervention and habitat_conversion. \n\n")

    policy_model = genai.GenerativeModel(
        model_name=policy_model_name,
        # gemini-2.0-flash-thinking-exp-01-21, gemini-2.0-flash-exp, gemini-exp-1206, gemini-1.5-pro, gemini-1.5-flash
        system_instruction=policy_system_instructions
    )
    farm_model = genai.GenerativeModel(
        model_name=farm_model_name,
        system_instruction=farm_system_instructions
    )
    fix_model = genai.GenerativeModel(
        model_name=fix_model_name,
        system_instruction="You are a helpful assistant who is an expert in graph and spatial optimization methods and python. "
                           "Given the python code and the stack traceback, fix the errors and return the correct functioning python code."
    )
    # Initialize rate limiter
    from config import Config
    cfg = Config()
    global rate_limiter
    rate_limiter = RateLimiter(**cfg.rate_limit)
    
    return policy_model, farm_model, fix_model


def validate_response(response):
    temp_file = os.path.join(heur_dir, "temp_fix.py")
    with open(temp_file, 'w') as f:
        f.write(response)

    input_src = os.path.join(heur_dir, "input_cp.geojson")
    input_dst = os.path.join(heur_dir, "input.geojson")

    os.chdir(heur_dir)
    code = 1
    tries = 0
    while code and tries <= 1:
        shutil.copyfile(input_src, input_dst)
        code, out, err = capture.run_python_script("temp_fix.py")
        if code:
            try:
                response = fix_errors(fix_model, response, err)
                tries += 1
            except Exception as e:
                print(e)
    return response


def evaluate_heuristics(heuristic_policy, ground_truth, mode="temp"):
    def run_heurisic_policy(heuristic_policy):
        session = farm_model.start_chat(history=[])
        prompt = farm_task_instructions + f"Message from the policy professional is: {heuristic_policy}\n"
        completion = send_message_with_retry(session, prompt, rate_limiter)
        response = completion.parts[0].text
        heuristics_code = extract_python_code(response)
        heuristics_code = validate_response(heuristics_code)
        return heuristics_code

    try:
        heuristics_code = run_heurisic_policy(heuristic_policy)

        src_file = os.path.join(heur_dir, mode + ".py")
        with open(src_file, 'w') as f:
            f.write(heuristics_code)
    except Exception as e:
        return 0, heuristic_policy, ""

    try:
        os.chdir(heur_dir)
        code, _, err = capture.run_python_script(mode + ".py")

        with open(os.path.join(heur_dir, 'output.geojson')) as f:
            predicted_output = json.load(f)

        # Initialize error accumulators for margin_intervention and habitat_conversion
        margin_errors = []
        habitat_errors = []

        pred_features = predicted_output["features"]
        input_json_features = input_json["features"]

        # Iterate over the features and calculate the error
        for inp_feature in input_json_features:
            if inp_feature["properties"]["type"] == "hab_plots":
                continue

            plot_id = inp_feature["properties"]["id"]
            gt_margin = 0
            gt_hab = 0
            for feat in ground_truth:
                if plot_id == feat["id"]:
                    gt_margin = len(set(feat["margin_directions"]))/4
                    gt_hab = len(set(feat["habitat_directions"]))/4
                    break

            pred_found = 0
            for pred_feat in pred_features:
                if pred_feat["properties"]["id"] == plot_id:
                    pred_margin = pred_feat["properties"]["margin_intervention"]
                    pred_hab = pred_feat["properties"]["habitat_conversion"]

                    margin_error = abs(gt_margin - pred_margin)
                    habitat_error = abs(gt_hab - pred_hab)
                    pred_found = 1
                    break

            if not pred_found:
                margin_error = 10
                habitat_error = 10

            # Append the errors to the respective lists
            margin_errors.append(margin_error)
            habitat_errors.append(habitat_error)

        # Compute overall error as mean absolute error
        total_margin_error = np.mean(margin_errors)
        total_habitat_error = np.mean(habitat_errors)
        total_error = total_margin_error + total_habitat_error + 0.01
        return 1 / total_error, heuristic_policy, heuristics_code
    except Exception as e:
        return 0, heuristic_policy, heuristics_code


def fix_errors(fix_model, heuristics_code, trace):
    session = fix_model.start_chat(
        history=[
        ]
    )

    prompt = f"""Given the following Python code:

    ```python
    {heuristics_code}
    ```
    And the following traceback: 
    {trace}
    Fix the errors and return the correct functioning python code. Give the full code. 
    """

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    python_code = extract_python_code(response)
    return python_code


def mutate_heuristics(heuristic_policy):
    session = policy_model.start_chat(history=[])

    prompt = (f"""The following message is communicated to the farmer in order to change their behaviour from 
        following the ecological intensification heuristics that benefits solely their farm to following the 
        ecological connectivity heuristics that increase landscape connectivity:
        {heuristic_policy}\n\n""" +
        """Suggest a subtle mutation to the message to better nudge the farmer. 
        Focus on small, logical changes in terms of how the message is communicated or the kind of incentives you provide.
        The new message should still be a valid message that can be communicated to the farmer. 
        Explain your reasoning and think step by step.  
        Provide your final message to the farmer in this format \communication{message}.\n\n
        """)

    prompt = prompt + policy_task_instructions

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_message(response)
    return response


def crossover_heuristics(parent1_messaqge, parent2_message):
    session = policy_model.start_chat(history=[])

    prompt = (f"""Given two sets of messages communicated to the farmer in order to change 
    their behaviour (parent 1 and parent 2):

    Parent 1:
    {parent1_messaqge}

    Parent 2:
    {parent2_message}
    """ +
    """Combine these two sets of messages in an optimal way to cover information from both parent 1 and parent 2.
    The combination should still result in a valid message that can be communicated to the farmer. 
    Explain your reasoning and think step by step.  
    Provide your final message to the farmer in this format \communication{message}.\n\n
    """)

    prompt = prompt + policy_task_instructions

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_message(response)
    return response


def evolve_heuristics_1(parent1_messaqge, parent2_message):
    session = policy_model.start_chat(history=[])

    prompt = (f"""Given two sets of messages communicated to the farmer in order to change 
    their behaviour (parent 1 and parent 2):

    Parent 1:
    {parent1_messaqge}

    Parent 2:
    {parent2_message}
    """ +
    """Generate a new message that are is as much different as possible from parent messages, in order to explore new ideas.
    The new message should still be a valid message that can be communicated to the farmer.  
    Explain your reasoning and think step by step.  
    Provide your final message to the farmer in this format \communication{message}.\n\n
    """)

    prompt = prompt + policy_task_instructions

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_message(response)
    return response


def evolve_heuristics_2(parent1_messaqge, parent2_message):
    session = policy_model.start_chat(history=[])

    prompt = (f"""Given two sets of messages communicated to the farmer in order to change 
        their behaviour (parent 1 and parent 2):

        Parent 1:
        {parent1_messaqge}

        Parent 2:
        {parent2_message}
        """ +
        """Explore a new message that shares the same idea as the parent messages. 
        Identify common ideas behind these messages. Then, design a new message that is based the common ideas but 
        is as much different as possible from the parents by introducing new parts.
        The new message should still be a valid message that can be communicated to the farmer.  
        Explain your reasoning and think step by step.  
        Provide your final message to the farmer in this format \communication{message}.\n\n 
        """)

    prompt = prompt + policy_task_instructions

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_message(response)
    return response


def reflect(top_heuristics, top_scores):
    # Start a new chat session with the heuristics model.
    session = policy_model.start_chat(history=[])

    # Create a summary of the top heuristics and their fitness scores.
    heuristics_info = ""
    for i, (heur, score) in enumerate(zip(top_heuristics, top_scores), start=1):
        heuristics_info += f"Message {i} (Fitness Score: {score}):\n{heur}\n\n"

    # Create a prompt that displays the heuristics and asks for a new, improved heuristic.
    prompt = (f"Based on the following top 5 messages and their corresponding fitness scores: {heuristics_info}" +
            """Please analyze these messages and craft a new message that is expected to have increased fitness.
            Ensure that the new message is a valid message that can be communicated to the farmer.
            Explain your reasoning and think step by step.  
            Provide your final message to the farmer in this format \communication{message}.\n\n 
            """)

    prompt = prompt + policy_task_instructions

    # Send the prompt to the model.
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_message(response)
    return response


def run_reflect(population, scores):
    pop_with_scores = list(zip(population, scores))
    pop_with_scores_sorted = sorted(pop_with_scores, key=lambda x: x[1], reverse=True)
    sorted_population = [p for p, s in pop_with_scores_sorted]
    sorted_scores = [s for p, s in pop_with_scores_sorted]

    reflect_results = []
    # ---- Run 1: Use the top 5 heuristics ----
    top5_heuristics = sorted_population[:5]
    top5_scores = sorted_scores[:5]
    result1 = reflect(top5_heuristics, top5_scores)
    reflect_results.append(result1)

    # ---- Runs 3 to 5: Use first (best) and 4 random individuals from the rest ----
    for run in range(4):
        # Ensure the best heuristic (index 0) is always included.
        # Then randomly sample 4 additional heuristics from the remaining ones.
        if len(sorted_population) >= 5:
            random_indices = random.sample(range(0, len(sorted_population)), 5)
        else:
            # If there aren't enough individuals, simply use all remaining
            random_indices = list(range(0, len(sorted_population)))
        indices_run = random_indices
        sel_heuristics = [sorted_population[i] for i in indices_run]
        sel_scores = [sorted_scores[i] for i in indices_run]
        try:
            result = reflect(sel_heuristics, sel_scores)
            reflect_results.append(result)
        except Exception as e:
            print(e)

    reflect_scores, reflect_results, reflect_codes = evaluate_population(reflect_results, ground_truth)
    return reflect_scores, reflect_results, reflect_codes


def select_population(population, scores, codes, population_size):
    # Select the top individuals based on their scores
    population_with_scores = sorted(zip(population, scores, codes), key=lambda item: item[1], reverse=True)
    scores = []
    population = []
    codes = []
    for individual in population_with_scores[:population_size]:
        population.append(individual[0])
        scores.append(individual[1])
        codes.append(individual[2])
    return population, scores, codes


def evaluate_population(population, ground_truth):
    scores = []
    new_population = []
    codes = []
    for individual in population:
        score, policy, code = evaluate_heuristics(individual, ground_truth)
        if not (math.isnan(score)):
            scores.append(score)
            new_population.append(policy)
            codes.append(code)
    return scores, new_population, codes


def run_gemini_flashexp(prompt_path, heuristics_file, model):
    with open(prompt_path, "r") as f:
        prompt = f.read()

    session = model.start_chat(
        history=[
        ]
    )

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    #policy_message = extract_python_code(response)
    policy_message =  extract_message(response)

    with open(heuristics_file, 'w') as f:
        f.write(policy_message)


def create_init_population(policy_model):
    prompt_path = os.path.join(conn_dir, "prompt_input.txt")

    for i in range(1, population_size + 1):
        heur_file = os.path.join(heur_dir, "heuristics_gem_" + str(i) + ".txt")

        try:
            run_gemini_flashexp(prompt_path, heur_file, policy_model)
        except Exception as e:
            print(e)
            continue


def compute_radon_metrics(old_df, python_codes, fitness_scores):
    new_rows = []

    for code, score in zip(python_codes, fitness_scores):
        # 1) Raw metrics
        try:
            raw_metrics = analyze(code)
        except Exception as e:
            continue

        # raw_metrics has:
        #   loc -> total number of lines
        #   lloc -> logical lines of code
        #   sloc -> source lines of code
        #   comment -> number of comment lines
        #   multi -> number of multi-line strings
        #   blank -> number of blank lines

        # 2) Cyclomatic complexity
        try:
            complexities = cc_visit(code)
        except Exception as e:
            complexities = 0

        # complexities is a list of FunctionInfo / ClassInfo (or empty if no defs).
        # Each entry has a .complexity attribute (among others).
        if complexities:
            avg_cyclomatic_complexity = sum(c.complexity for c in complexities) / len(complexities)
        else:
            avg_cyclomatic_complexity = 0.0

        # 3) Halstead metrics
        # h_visit returns a list of Halstead objects for each function/class definition.
        # Each Halstead object has:
        #   h1 (distinct operators), h2 (distinct operands)
        #   N1 (total operators), N2 (total operands)
        #   vocabulary, length, calculated_length, volume, difficulty,
        #   effort, time, bugs
        try:
            halstead_results = h_visit(code)
        except Exception as e:
            halstead_results = []

        # We can aggregate Halstead results across all functions/classes.
        # For example, let's take an average for each property:
        if len(halstead_results) > 0:
            total_h1 = halstead_results.total.h1
            total_h2 = halstead_results.total.h2
            total_N1 = halstead_results.total.N1
            total_N2 = halstead_results.total.N2
            total_vocabulary = halstead_results.total.vocabulary
            total_length = halstead_results.total.length
            total_volume = halstead_results.total.volume
            total_difficulty = halstead_results.total.difficulty
            total_effort = halstead_results.total.effort
            total_time = halstead_results.total.time
            total_bugs = halstead_results.total.bugs
        else:
            # If there are no functions/classes, fill with 0 or None
            total_h1 = total_h2 = total_N1 = total_N2 = 0
            total_vocabulary = total_length = total_volume = 0
            total_difficulty = total_effort = total_time = total_bugs = 0

        # 4) Maintainability Index
        # mi_approx(code, multi=True, comments=True) returns a single MI float.
        # Typically, you can experiment with toggling multi/comments if needed.
        try:
            mi_value = mi_visit(code, multi=True)
        except Exception as e:
            mi_value = 0

        # Create a row dict
        row = {
            'fitness_score': score,

            # Raw metrics
            'loc': raw_metrics.loc,
            'lloc': raw_metrics.lloc,
            'sloc': raw_metrics.sloc,
            'comment': raw_metrics.comments,
            'multi': raw_metrics.multi,
            'blank': raw_metrics.blank,

            # Cyclomatic complexity
            'avg_cyclomatic_complexity': avg_cyclomatic_complexity,

            # Maintainability Index
            'maintainability_index': mi_value,

            # Halstead average metrics
            'halstead_h1': total_h1,
            'halstead_h2': total_h2,
            'halstead_N1': total_N1,
            'halstead_N2': total_N2,
            'halstead_vocabulary': total_vocabulary,
            'halstead_length': total_length,
            'halstead_volume': total_volume,
            'halstead_difficulty': total_difficulty,
            'halstead_effort': total_effort,
            'halstead_time': total_time,
            'halstead_bugs': total_bugs,
        }
        new_rows.append(row)

    # Create a DataFrame from these new rows
    new_df = pd.DataFrame(new_rows)

    # Concatenate with the existing old_df
    metrics_df = pd.concat([old_df, new_df], ignore_index=True)
    return metrics_df


def run_evo_strat(population):
    scores, population, codes = evaluate_population(population, ground_truth)
    best_index = scores.index(max(scores))
    best_policy = population[best_index]
    best_python = codes[best_index]
    line = f"Generation - {0} : Best Score - {scores[best_index]}\n"
    print(line)
    with open(score_file, 'a+') as f:
        f.write(line)

    best_policy_file = os.path.join(gen_dir, "best_policy_gem_gen_" + str(0) + ".txt")
    with open(best_policy_file, 'w') as f:
        f.write(best_policy)

    best_python_file = os.path.join(gen_dir, "best_python_gem_gen_" + str(0) + ".py")
    with open(best_python_file, 'w') as f:
        f.write(best_python)

    metrics_df = pd.DataFrame(
        columns=['fitness_score', 'loc', 'lloc', 'sloc', 'comment', 'multi', 'blank', 'avg_cyclomatic_complexity',
                 'maintainability_index', 'halstead_h1', 'halstead_h2', 'halstead_N1', 'halstead_N2',
                 'halstead_vocabulary',
                 'halstead_length', 'halstead_volume', 'halstead_difficulty', 'halstead_effort', 'halstead_time',
                 'halstead_bugs'])
    metrics_df = compute_radon_metrics(metrics_df, codes, scores)
    metrics_df.to_csv(metrics_file, index=False)

    selected_population, selected_scores, selected_codes = select_population(population, scores, codes, population_size)
    for generation in range(num_generations):
        best_policy_file = os.path.join(gen_dir, "best_policy_gem_gen_" + str(generation + 1) + ".txt")
        best_python_file = os.path.join(gen_dir, "best_python_gem_gen_" + str(generation + 1) + ".py")

        next_generation = []
        for i in range(inner_loop_size):
            # Crossover
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            try:
                next_generation.append(crossover_heuristics(parent1, parent2))
            except Exception as e:
                print(e)

            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            try:
                next_generation.append(evolve_heuristics_1(parent1, parent2))
                next_generation.append(evolve_heuristics_2(parent1, parent2))
            except Exception as e:
                print(e)

            individual = random.choice(selected_population)
            try:
                next_generation.append(mutate_heuristics(individual))
            except Exception as e:
                print(e)

        # Evaluate fitness of each individual in parallel
        next_gen_scores, next_generation, next_codes = evaluate_population(next_generation, ground_truth)

        population = selected_population + next_generation
        scores = selected_scores + next_gen_scores
        codes = selected_codes + next_codes

        r_scores, r_population, r_codes = run_reflect(population, scores)
        population = population + r_population
        scores = scores + r_scores
        codes = codes + r_codes

        metrics_df = compute_radon_metrics(metrics_df, codes, scores)

        # Keep the best heuristics of this generation
        best_index = scores.index(max(scores))
        line = f"Generation - {generation + 1} : Best Score - {scores[best_index]}\n"
        print(line)
        with open(score_file, 'a+') as f:
            f.write(line)

        with open(best_policy_file, 'w') as f:
            f.write(population[best_index])

        with open(best_python_file, 'w') as f:
            f.write(codes[best_index])

        metrics_df.to_csv(metrics_file, index=False)

        selected_population, selected_scores, selected_codes = select_population(population, scores, codes, population_size)
        for k, ind in enumerate(selected_population):
            file = os.path.join(heur_dir, "heuristics_gem_" + str(k + 1) + ".txt")
            with open(file, 'w') as f:
                f.write(ind)
                
            code = selected_codes[k]
            file = os.path.join(heur_dir, "heuristics_gem_" + str(k + 1) + ".py")
            with open(file, 'w') as f:
                f.write(code)

        if scores[best_index] > 10:
            break


if __name__ == "__main__":
    cfg = Config()
    capture = CommandOutputCapture()
    population_size = 25
    num_generations = 10
    inner_loop_size = 10
    init = True

    params_instructions = (
        "These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
        "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
        "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
        "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}.\n\n"
    )

    policy_task_instructions = (
        "The messages are communications to the farmers in order to change their behaviour from "
        "following the ecological intensification heuristics that benefits solely their farm to following the "
        "ecological connectivity heuristics that increase landscape connectivity. "
        "Your communication to the farmer can be persuasive. It can provide incentives such as reducing the "
        "implementation or maintenance cost of an intervention by providing a one-time subsidy or yearly subsidies. "
        "It can compensate the farmers for yield that is lost to habitat conversion. It can communicate the benefits of "
        "landscape connectivity, and so on. \n\n"
        "Your goal should be to communicate a message that gets them to alter the margin_intervention and "
        "habitat_conversion values for each of the plots, from the former to the latter. "
        "Your final message to the farmer should be in this format \communication{message}. \n\n"
    )

    #farms_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms")
    farms_dir = os.path.join(cfg.disk_dir, "syn_farms")

    farm_ids = [3]
    for farm_id in farm_ids:

        print(f"Running farm:{farm_id}")
        farm_dir = os.path.join(farms_dir, f"farm_{farm_id}")
        conn_dir = os.path.join(farm_dir, "connectivity", "run_1")
        nudge_dir = os.path.join(farm_dir, "nudge")

        create_nudge_prompt_file(nudge_dir)

        heur_eco_intens_path = os.path.join(nudge_dir, "heuristics_gem_eco_intens.py")
        with open(heur_eco_intens_path, "r") as input_file:
            heur_eco_intens = input_file.read()
        farm_task_instructions = (
            "The current set of heuristics you follow are:\n"
            f"```python\n{heur_eco_intens}\n```\n Don't get easily persuaded to change your behaviour. "
            "You need to be relatively selfish in following your heuristics so that you look out for your own interests. "
            "However, if the message convinces you, or provides the right incentives that align with what you want, "
            "enable you to do more for the landscape ecological connectivity than you are currently, then you should"
            "respond accordingly by changing your behaviour. If the message proposes changes to some key parameters that you "
            "are using in your heuristics, you should change them according to the proposed changes. If the message "
            "introduces new parameters or ideas, you can incorporate them in your heuristics if you want to. "
            "Your should react to the message from the policy professional, resulting in altered margin_intervention and "
            "habitat_conversion values for your plots, in an appropriate way that aligns "
            "with your interests and the message you received. "
            "Your final answer should be in python code, keeping overall framework of the original python code the same. \n\n"
        )

        heur_dir = os.path.join(nudge_dir, "heuristics")
        if not os.path.exists(heur_dir):
            os.makedirs(heur_dir)
        gen_dir = os.path.join(nudge_dir, "generations")
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

        files = ["input.geojson"]
        for file in files:
            inp_src = os.path.join(farm_dir, file)
            inp_dst = os.path.join(heur_dir, file)
            shutil.copyfile(inp_src, inp_dst)

        input_src = os.path.join(heur_dir, "input.geojson")
        input_json_path = os.path.join(heur_dir, "input_cp.geojson")
        shutil.copyfile(input_src, input_json_path)
        with open(input_json_path) as f:
            input_json = json.load(f)

        delete_outputs(heur_dir)

        score_file = os.path.join(heur_dir, "scores_es.txt")
        metrics_file = os.path.join(heur_dir, "metrics_es.csv")

        # Load the ground truth and output JSON files
        with open(os.path.join(conn_dir, 'output_gt_directions.json')) as f:
            ground_truth = json.load(f)

        # Extract model name from config
        config_model = cfg.lm.split('/')[-1] if '/' in cfg.lm else cfg.lm
        policy_model, farm_model, fix_model = init_gemini_model(policy_model_name=config_model,
                                                                farm_model_name=config_model,
                                                                fix_model_name=config_model)
        if init:
            create_nudge_prompt_file(nudge_dir)
            create_init_population(policy_model)

        init_population = get_init_population()
        run_evo_strat(init_population)

        print("done")
