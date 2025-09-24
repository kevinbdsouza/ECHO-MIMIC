import google.generativeai as genai
import random
from tools import *
import math
from create_prompts import create_farm_prompt_file_2
import pandas as pd
from radon.raw import analyze
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from config import Config
from rate_limiter import RateLimiter, send_message_with_retry
from dotenv import load_dotenv


def extract_python_code(text):
    pattern = r'```python(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "No Python code found."


def init_gemini_model(heur_model_name="gemini-2.0-flash-thinking-exp-01-21", fix_model_name="gemini-2.0-flash-thinking-exp-01-21"):
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variables
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("No API key found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")
    
    genai.configure(api_key=api_key)

    system_instructions = ("You are a helpful assistant who is an expert in spatial optimization methods "
                           "and who helps the user with optimization related queries. You will return final answers in python code. "
                           "You can't use the given values of interventions (margin_intervention and habitat_conversion) to "
                           "produce the output, they are only for reference to produce the heuristics. "
                           "Do not create any dummy data and dump to input geojson at any cost. "
                           "You should load the input geojson from the existing input.geojson. "
                           "You can only use this input data and no other data. "
                           "Every feature (plot) in the input geojson should have output interventions, don't skip any feature (plot). "
                           "Save outputs to output.geojson. \n\n")

    if use_template:
        with open(os.path.join(cfg.src_dir, "old", "backup", "heuristic_template.py"), 'r') as f:
            heuristics_template = f.read()
        system_instructions = system_instructions + f"You can use the following heuristic template: ```python{heuristics_template}```. \n\n"

    if halstead_metrics:
        system_instructions = system_instructions + halstead_instructions

    heur_model = genai.GenerativeModel(
        model_name=heur_model_name,
        # gemini-2.0-flash-thinking-exp-01-21, gemini-2.0-flash-exp, gemini-exp-1206, gemini-1.5-pro, gemini-1.5-flash
        system_instruction=system_instructions
    )
    fix_model = genai.GenerativeModel(
        model_name=fix_model_name,  # gemini-2.0-flash-thinking-exp-01-21, gemini-2.0-flash-exp
        system_instruction="You are a helpful assistant who is an expert in spatial optimization methods and python. "
                           "Given the python code and the stack traceback, fix the errors and return the correct functioning python code."
    )
    
    # Initialize rate limiter with config values
    global rate_limiter
    rate_limiter = RateLimiter(**cfg.rate_limit)
    
    return heur_model, fix_model


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


def evaluate_heuristics(heuristics_code, ground_truth, mode="temp"):
    src_file = os.path.join(heur_dir, mode + ".py")
    with open(src_file, 'w') as f:
        f.write(heuristics_code)

    try:
        os.chdir(heur_dir)
        code, _, err = capture.run_python_script(mode + ".py")

        with open(os.path.join(heur_dir, 'output.geojson')) as f:
            predicted_output = json.load(f)

        # Extract features from both ground truth and predicted output
        gt_features = ground_truth["features"]
        pred_features = predicted_output["features"]
        input_json_features = input_json["features"]

        # Initialize error accumulators for margin_intervention and habitat_conversion
        margin_errors = []
        habitat_errors = []

        # Iterate over the features and calculate the error
        for inp_feature in input_json_features:
            if inp_feature["properties"]["type"] == "hab_plots":
                continue

            plot_id = inp_feature["properties"]["id"]
            gt_margin = 0
            gt_hab = 0
            for feat in gt_features:
                if plot_id == feat["properties"]["id"]:
                    gt_margin = feat["properties"]["margin_intervention"]
                    gt_hab = feat["properties"]["habitat_conversion"]
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
        return 1 / total_error, heuristics_code
    except Exception as e:
        return 0, heuristics_code


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


def pick_parent_for_multi(parent1, parent2):
    if len(parent1["trajectory"]) >= len(parent2["trajectory"]):
        return parent1
    elif len(parent2["trajectory"]) > len(parent1["trajectory"]):
        return parent2

def copy_parent_candidate(parent):
    return {
        "code": parent["code"],
        "trajectory": parent["trajectory"][:],
        "counts": parent["counts"].copy(),
        "fitness_deltas": parent["fitness_deltas"].copy(),
        "score": 0.0
    }

def mutate_heuristics(parent):
    parent_code = parent["code"]

    session = heur_model.start_chat(history=[])

    prompt = f"""Given the following Python code for agricultural plot heuristics:

    ```python
    {parent_code}
    ```
    Suggest a subtle mutation to the heuristics to improve overall performance and logic. 
    The mutation should still result in valid Python code. 
    You can't use the given values of interventions (margin_intervention and habitat_conversion) to produce the output, 
    they are only for reference to produce the heuristics. 
    Focus on small, logical changes related to the plot's properties, geometry, or interactions with neighboring plots.
    Retain loading input geojson from input.geojson and saving outputs to output.geojson. 
    Do not create any dummy data and dump to input geojson at any cost. 
    Don't create new variable names. Use variable names  margin_intervention and habitat_conversion for predicted values in the output. 
    Return the modified Python code for the heuristics. Explain your reasoning and think step by step.  Do not hallucinate. 
    """

    if halstead_metrics:
        prompt = prompt + common_task_instructions + params_instructions + halstead_instructions
    else:
        prompt = prompt + common_task_instructions + params_instructions

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_python_code(response)
    new_code = validate_response(response)

    child = copy_parent_candidate(parent)
    child["code"] = new_code
    child["pending_op"] = "mutate"
    child["pending_parent_score"] = parent["score"]
    return child


def crossover_heuristics(parent1, parent2):
    session = heur_model.start_chat(history=[])

    prompt = f"""Given two sets of Python code for agricultural plot heuristics (parent 1 and parent 2):

    Parent 1:
    ```python
    {parent1["code"]}
    ```

    Parent 2:
    ```python
    {parent2["code"]}
    ```
    Combine these two sets of heuristics in an optimal way to cover heuristics from both parent 1 and parent 2.
    The combination should still result in valid Python code. 
    You can't use the given values of interventions (margin_intervention and habitat_conversion) to produce the output, they are only for reference to produce the heuristics. 
    Retain loading input geojson from input.geojson and saving outputs to output.geojson.
    Do not create any dummy data and dump to input geojson at any cost. 
    Don't create new variable names. Use variable names margin_intervention and habitat_conversion for predicted values in the output. 
    Return the modified Python code for the heuristics. Explain your reasoning and think step by step.  Do not hallucinate. 
    """

    dominant_parent = pick_parent_for_multi(parent1, parent2)

    if halstead_metrics:
        prompt = prompt + common_task_instructions + params_instructions + halstead_instructions
    else:
        prompt = prompt + common_task_instructions + params_instructions

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_python_code(response)
    new_code = validate_response(response)

    child = copy_parent_candidate(dominant_parent)
    child["code"] = new_code
    child["pending_op"] = "crossover"
    child["pending_parent_score"] = dominant_parent["score"]
    return child


def evolve_heuristics_1(parent1, parent2):
    session = heur_model.start_chat(history=[])

    prompt = f"""Given two sets of Python code for agricultural plot heuristics (parent 1 and parent 2):

    Parent 1:
    ```python
    {parent1["code"]}
    ```

    Parent 2:
    ```python
    {parent2["code"]}
    ```
    Generate new heuristics that are as much different as possible from parent heuristics, in order to explore new ideas.
    The new heuristics should still result in valid Python code. 
    You can't use the given values of interventions (margin_intervention and habitat_conversion) to produce the output, they are only for reference to produce the heuristics. 
    Retain loading input geojson from input.geojson and saving outputs to output.geojson.
    Do not create any dummy data and dump to input geojson at any cost. 
    Don't create new variable names. Use variable names margin_intervention and habitat_conversion for predicted values in the output. 
    Return the modified Python code for the heuristics. Explain your reasoning and think step by step.  Do not hallucinate. 
    """

    dominant_parent = pick_parent_for_multi(parent1, parent2)

    if halstead_metrics:
        prompt = prompt + common_task_instructions + params_instructions + halstead_instructions
    else:
        prompt = prompt + common_task_instructions + params_instructions

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_python_code(response)
    new_code = validate_response(response)

    child = copy_parent_candidate(dominant_parent)
    child["code"] = new_code
    child["pending_op"] = "evolve_1"
    child["pending_parent_score"] = dominant_parent["score"]
    return child


def evolve_heuristics_2(parent1, parent2):
    session = heur_model.start_chat(history=[])

    prompt = f"""Given two sets of Python code for agricultural plot heuristics (parent 1 and parent 2):

    Parent 1:
    ```python
    {parent1["code"]}
    ```

    Parent 2:
    ```python
    {parent2["code"]}
    ```
    Explore new heuristics that share the same idea as the parent heuristics. 
    Identify common ideas behind these heuristics. Then, design new heuristics that are based the common ideas but 
    are as much different as possible from the parents by introducing new parts.
    The new heuristics should still result in valid Python code. 
    You can't use the given values of interventions (margin_intervention and habitat_conversion) to produce the output, they are only for reference to produce the heuristics. 
    Retain loading input geojson from input.geojson and saving outputs to output.geojson.
    Do not create any dummy data and dump to input geojson at any cost. 
    Don't create new variable names. Use variable names margin_intervention and habitat_conversion for predicted values in the output. 
    Return the modified Python code for the heuristics. Explain your reasoning and think step by step.  Do not hallucinate. 
    """

    dominant_parent = pick_parent_for_multi(parent1, parent2)

    if halstead_metrics:
        prompt = prompt + common_task_instructions + params_instructions + halstead_instructions
    else:
        prompt = prompt + common_task_instructions + params_instructions

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_python_code(response)
    new_code = validate_response(response)

    child = copy_parent_candidate(dominant_parent)
    child["code"] = new_code
    child["pending_op"] = "evolve_2"
    child["pending_parent_score"] = dominant_parent["score"]
    return child

def reflect(top_candidates):
    # Start a new chat session with the heuristics model.
    best_candidate = max(top_candidates, key=lambda c: c["score"])
    session = heur_model.start_chat(history=[])

    # Create a summary of the top heuristics and their fitness scores.
    heuristics_info = ""
    for i, candidate in enumerate(top_candidates, start=1):
        heuristics_info += f"Heuristic {i} (Fitness: {candidate['score']}):\n{candidate['code']}\n\n"

    # Create a prompt that displays the heuristics and asks for a new, improved heuristic.
    prompt = f"""You are an expert in spatial optimization and agricultural heuristics.
            Based on the following top 5 heuristics and their corresponding fitness scores: {heuristics_info}

            Please analyze these heuristics and craft a new heuristic that is expected to have increased fitness.
            Ensure that the new heuristic:
            - Is valid Python code.
            - Loads the input geojson from 'input.geojson' and writes the output to 'output.geojson'.
            - Do not create any dummy data and dump to input geojson at any cost. 
            - Uses the variable names 'margin_intervention' and 'habitat_conversion' for the predicted values.

            Explain your reasoning step by step and return the complete Python code for the new heuristic.
            """

    if halstead_metrics:
        prompt = prompt + common_task_instructions + params_instructions + halstead_instructions
    else:
        prompt = prompt + common_task_instructions + params_instructions

    # Send the prompt to the model.
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text

    # Extract and validate the Python code from the response.
    response = extract_python_code(response)
    new_code = validate_response(response)

    child = copy_parent_candidate(best_candidate)
    child["code"] = new_code
    child["pending_op"] = "reflect"
    child["pending_parent_score"] = best_candidate["score"]
    return child


def run_reflect(population, scores):
    pop_with_scores = list(zip(population, scores))
    pop_with_scores_sorted = sorted(pop_with_scores, key=lambda x: x[1], reverse=True)
    sorted_population = [p for p, s in pop_with_scores_sorted]
    sorted_scores = [s for p, s in pop_with_scores_sorted]

    reflect_results = []
    # ---- Run 1: Use the top 5 heuristics ----
    top5 = sorted_population[:5]
    top5_scores = sorted_scores[:5]
    child1 = reflect(top5)
    reflect_results.append(child1)

    # ---- Runs 3 to 5: Use first (best) and 4 random individuals from the rest ----
    for run in range(4):
        if len(sorted_population) >= 5:
            random_candidates = random.sample(sorted_population, 5)
        else:
            random_candidates = sorted_population

        try:
            childX = reflect(random_candidates)
            reflect_results.append(childX)
        except Exception as e:
            print(e)

    reflect_results_evaluated = evaluate_population(reflect_results, ground_truth)
    reflect_scores = [cand["score"] for cand in reflect_results_evaluated]
    return reflect_scores, reflect_results_evaluated


def select_population(population, population_size):
    sorted_pop = sorted(population, key=lambda c: c["score"], reverse=True)
    return sorted_pop[:population_size]


def evaluate_population(population, ground_truth):
    new_population = []
    for candidate in population:
        score, code = evaluate_heuristics(candidate["code"], ground_truth)
        if not math.isnan(score):
            candidate["score"] = score
            candidate["code"] = code
            op = candidate.pop("pending_op", None)
            parent_fitness = candidate.pop("pending_parent_score", None)
            if op is not None and parent_fitness is not None:
                delta = candidate["score"] - parent_fitness
                if op not in candidate["counts"]:
                    candidate["counts"][op] = 0
                candidate["counts"][op] += 1
                if op not in candidate["fitness_deltas"]:
                    candidate["fitness_deltas"][op] = 0.0
                candidate["fitness_deltas"][op] += delta
                candidate["trajectory"].append(f"{op}({delta:+.4f})")
            new_population.append(candidate)
    return new_population


def run_gemini_flashexp(prompt_path, heuristics_file, init_model):
    with open(prompt_path, "r") as f:
        prompt = f.read()

    session = init_model.start_chat(
        history=[
        ]
    )

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    python_code = extract_python_code(response)

    with open(heuristics_file, 'w') as f:
        f.write(python_code)


def create_init_population(init_model, fix_model):
    prompt_path = os.path.join(farm_dir, "prompt_input.txt")

    for i in range(1, population_size + 1):
        heur_file = os.path.join(heur_dir, "heuristics_gem_" + str(i) + ".py")
        input_src = os.path.join(heur_dir, "input_cp.geojson")
        input_dst = os.path.join(heur_dir, "input.geojson")

        try:
            run_gemini_flashexp(prompt_path, heur_file, init_model)
        except Exception as e:
            print(e)
            continue

        with open(heur_file, 'r') as f:
            heuristics_code = f.read()

        os.chdir(heur_dir)
        code = 1
        tries = 0
        while code and tries <= 2:
            shutil.copyfile(input_src, input_dst)

            code, out, err = capture.run_python_script("heuristics_gem_" + str(i) + ".py")
            if code:
                try:
                    heuristics_code = fix_errors(fix_model, heuristics_code, err)
                    tries += 1
                except Exception as e:
                    print(e)

        with open(heur_file, 'w') as f:
            f.write(heuristics_code)


def get_init_population():
    init_population = []
    operator_list = ["init", "mutate", "crossover", "evolve_1", "evolve_2", "reflect"]
    for i in range(1, population_size + 1):
        heuristic_file = os.path.join(heur_dir, "heuristics_gem_" + str(i) + ".py")
        with open(heuristic_file, 'r') as f:
            heuristics_code = f.read()

        candidate = {
            "code": heuristics_code,
            "trajectory": ["init(0.0000)"],
            "counts": {op: 0 for op in operator_list},
            "fitness_deltas": {op: 0.0 for op in operator_list},
            "score": 0.0
        }
        candidate["counts"]["init"] = 1
        init_population.append(candidate)
    return init_population


def compute_radon_metrics(old_df, population):
    new_rows = []

    for candidate in population:
        code = candidate["code"]
        score = candidate["score"]

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


def plot_best_trajectory_across_generations(best_history, gen_dir):
    base_op_list = ["init", "mutate", "crossover", "evolve_1", "evolve_2", "reflect"]
    operator_map = {op: i for i, op in enumerate(base_op_list)}
    for entry in best_history:
        gen = entry["generation"]
        score = entry["score"]
        trajectory_list = entry["trajectory"]
        counts_dict = entry["counts"]
        fitness_deltas_dict = entry["fitness_deltas"]

        ops = []
        deltas = []

        for t in trajectory_list:
            idx = t.find("(")
            if idx == -1:
                # If we can't find "(" for some reason, fallback
                op_name = t
                delta_str = "0.0"
            else:
                op_name = t[:idx]
                # inside parentheses => t[idx:] e.g. "(+0.0123)"
                # remove parentheses:
                inside = t[idx:].strip("()")
                # e.g. inside = "+0.0123"
                delta_str = inside

            # Convert op_name to something standard if there's extra whitespace
            op_name = op_name.strip()
            # Convert the delta_str to float if possible
            try:
                delta_val = float(delta_str)
            except:
                delta_val = 0.0

            ops.append(op_name)
            deltas.append(delta_val)

        x_vals = list(range(len(ops)))  # step index
        y_op = [operator_map.get(op, -1) for op in ops]  # -1 if not found

        plt.figure()
        plt.plot(x_vals, y_op, marker='o')
        plt.title(f"Best Candidate Trajectory (Gen {gen})\nScore={score:.4f}")
        plt.xlabel("Trajectory Step")
        plt.ylabel("Operators)")

        plt.yticks(
            list(range(len(base_op_list))),
            base_op_list
        )

        for i, (x, y) in enumerate(zip(x_vals, y_op)):
            d = deltas[i]
            plt.text(x, y, f"{d:+.3f}", fontsize=9, ha='left', va='bottom')

        outname = os.path.join(gen_dir, f"best_trajectory_gen_{gen}.png")
        plt.tight_layout()
        plt.savefig(outname)
        plt.close()

        plt.figure()
        op_names = list(counts_dict.keys())
        usage_vals = [counts_dict[op] for op in op_names]
        plt.bar(op_names, usage_vals)
        plt.title(f"Operator Usage Counts - Gen {gen}, Score={score:.4f}")
        plt.xlabel("Operator")
        plt.ylabel("Count")
        outname = os.path.join(gen_dir, f"best_counts_gen_{gen}.png")
        plt.savefig(outname)
        plt.close()

        plt.figure()
        ops_list = list(fitness_deltas_dict.keys())
        deltas_list = [fitness_deltas_dict[op] for op in ops_list]
        plt.bar(ops_list, deltas_list)
        plt.axhline(y=0.0, color='gray', linestyle='--')  # helps see positive vs negative
        plt.title(f"Operator Cumulative Deltas (Gen {gen})\nScore={score:.4f}")
        plt.xlabel("Operator")
        plt.ylabel("Cumulative Fitness Delta")
        plt.tight_layout()
        outname = os.path.join(gen_dir, f"best_fitness_deltas_gen_{gen}.png")
        plt.savefig(outname)
        plt.close()


def plot_population_operator_stats(population, generation, gen_dir):
    """ At the end of each generation, we want to see how operators were used across
    all candidates in the population and the net deltas they contributed.

    We'll produce two bar charts:

    1) operator_usage_all_gen_{gen}.png
       Summation of usage counts across all individuals.

    2) operator_deltas_all_gen_{gen}.png
       Summation of net fitness deltas across all individuals.
    """

    # We'll gather sums across the population
    operator_list = ["init", "mutate", "crossover", "evolve_1", "evolve_2", "reflect"]
    usage_sums = {op: 0 for op in operator_list}
    delta_sums = {op: 0.0 for op in operator_list}

    for cand in population:
        for op in operator_list:
            usage_sums[op] += cand["counts"].get(op, 0)
            delta_sums[op] += cand["fitness_deltas"].get(op, 0.0)

    # 1) Plot usage sums
    plt.figure()
    ops_list = list(usage_sums.keys())
    usage_vals = [usage_sums[op] for op in ops_list]
    plt.bar(ops_list, usage_vals)
    plt.title(f"All-Pop Operator Usage (Gen {generation})")
    plt.xlabel("Operator")
    plt.ylabel("Usage Count")
    outname = os.path.join(gen_dir, f"operator_usage_all_gen_{generation}.png")
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()

    # 2) Plot delta sums
    plt.figure()
    delta_vals = [delta_sums[op] for op in ops_list]
    plt.bar(ops_list, delta_vals)
    plt.axhline(y=0.0, color='gray', linestyle='--')
    plt.title(f"All-Pop Operator Deltas (Gen {generation})")
    plt.xlabel("Operator")
    plt.ylabel("Sum of Fitness Deltas")
    outname = os.path.join(gen_dir, f"operator_deltas_all_gen_{generation}.png")
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()


def save_population_to_csv(population, generation, gen_dir):
    """ Writes a CSV file with info about all candidates in the population: columns:
    [generation, candidate_id, score, trajectory, counts, fitness_deltas, code] We'll store it in e.g.
    'population_gen_{generation}.csv'. """

    rows = []
    for i, cand in enumerate(population):
        row = { "generation": generation, "candidate_id": i, "score": cand["score"], "trajectory": json.dumps(cand["trajectory"]),
                "counts": json.dumps(cand["counts"]), "fitness_deltas": json.dumps(cand["fitness_deltas"]),
                "code": cand["code"]}
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(gen_dir, f"population_gen_{generation}.csv")
    df.to_csv(csv_path, index=False)


def save_best_history_to_csv(best_history, gen_dir):
    """ Writes a CSV with one row per generation's best candidate: columns: [generation, best_score, trajectory, counts, fitness_deltas] We'll store it in 'best_history.csv'. """
    rows = []
    for entry in best_history:
        row = { "generation": entry["generation"], "best_score": entry["score"], "trajectory": json.dumps(entry["trajectory"]),
                "counts": json.dumps(entry["counts"]), "fitness_deltas": json.dumps(entry["fitness_deltas"]) }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(gen_dir, "best_history.csv")
    df.to_csv(csv_path, index=False)


def run_evo_strat(population):
    # Evaluate fitness of each individual in parallel
    population = evaluate_population(population, ground_truth)
    scores = [cand["score"] for cand in population]
    best_index = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_index]
    best_heuristics = population[best_index]["code"]
    line = f"Generation - {0} : Best Score - {best_score}\n"
    print(line)
    with open(score_file, 'a+') as f:
        f.write(line)

    best_heuristics_file = os.path.join(gen_dir, "best_heuristics_gem_gen_" + str(0) + ".py")
    with open(best_heuristics_file, 'w') as f:
        f.write(best_heuristics)

    metrics_df = pd.DataFrame(
        columns=['fitness_score', 'loc', 'lloc', 'sloc', 'comment', 'multi', 'blank', 'avg_cyclomatic_complexity',
                 'maintainability_index', 'halstead_h1', 'halstead_h2', 'halstead_N1', 'halstead_N2',
                 'halstead_vocabulary',
                 'halstead_length', 'halstead_volume', 'halstead_difficulty', 'halstead_effort', 'halstead_time',
                 'halstead_bugs'])
    metrics_df = compute_radon_metrics(metrics_df, population)
    metrics_df.to_csv(metrics_file, index=False)

    best_history = []
    best_history.append({
        "generation": 0,
        "score": best_score,
        "trajectory": population[best_index]["trajectory"][:],
        "counts": population[best_index]["counts"].copy(),
        "fitness_deltas": population[best_index]["fitness_deltas"].copy()
    })
    plot_best_trajectory_across_generations(best_history, gen_dir)
    save_population_to_csv(population, 0, gen_dir)
    save_best_history_to_csv(best_history, gen_dir)

    population = select_population(population, population_size)
    for generation in range(num_generations):
        best_heuristics_file = os.path.join(gen_dir, "best_heuristics_gem_gen_" + str(generation + 1) + ".py")

        next_generation = []
        for i in range(inner_loop_size):
            print(f"Inner loop: {i}")

            # Crossover
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            try:
                next_generation.append(crossover_heuristics(parent1, parent2))
            except Exception as e:
                print(e)

            parent1 = random.choice(population)
            parent2 = random.choice(population)
            try:
                next_generation.append(evolve_heuristics_1(parent1, parent2))
                next_generation.append(evolve_heuristics_2(parent1, parent2))
            except Exception as e:
                print(e)

            individual = random.choice(population)
            try:
                next_generation.append(mutate_heuristics(individual))
            except Exception as e:
                print(e)

        # Evaluate fitness of each individual in parallel
        next_generation = evaluate_population(next_generation, ground_truth)
        combined_pop = population + next_generation
        combined_scores = [cand["score"] for cand in combined_pop]

        r_scores, r_population = run_reflect(combined_pop, combined_scores)
        combined_pop += r_population

        combined_pop = select_population(combined_pop, population_size)
        scores_cp = [c["score"] for c in combined_pop]
        best_index_g = max(range(len(scores_cp)), key=lambda i: scores_cp[i])
        best_score_g = scores_cp[best_index_g]

        line = f"Generation - {generation + 1} : Best Score - {best_score_g}\n"
        print(line)
        with open(score_file, 'a+') as f:
            f.write(line)

        with open(best_heuristics_file, 'w') as f:
            f.write(combined_pop[best_index_g]["code"])

        metrics_df = compute_radon_metrics(metrics_df, combined_pop)
        metrics_df.to_csv(metrics_file, index=False)

        plot_population_operator_stats(combined_pop, generation, gen_dir)

        for k, ind in enumerate(combined_pop):
            file = os.path.join(heur_dir, "heuristics_gem_" + str(k + 1) + ".py")
            with open(file, 'w') as f:
                f.write(ind["code"])

        best_history.append({
            "generation": generation + 1,
            "score": best_score_g,
            "trajectory": combined_pop[best_index_g]["trajectory"][:],
            "counts": combined_pop[best_index_g]["counts"].copy(),
            "fitness_deltas": combined_pop[best_index_g]["fitness_deltas"].copy()
        })
        population = combined_pop

        save_population_to_csv(combined_pop, generation, gen_dir)
        save_best_history_to_csv(best_history, gen_dir)
        plot_best_trajectory_across_generations(best_history, gen_dir)


if __name__ == "__main__":
    cfg = Config()
    capture = CommandOutputCapture()
    population_size = 25
    num_generations = 25
    inner_loop_size = 25
    init = True
    use_template = False
    halstead_metrics = False

    common_task_instructions = (
        "The python programs are trying to solve the task of deciding which interventions need to be done at which agricultural plots "
        "(crops, type='ag_plot') based on how the interventions affect NPV. The choice is between margin "
        "(convert only the margins) and habitat (convert a contiguous region) interventions. "
        "The interventions can be fractional. Existing habitat plots (type='hab_plots') "
        "remain unaffected. The NPV is calculated based on how the interventions affect pollination and pest control "
        "services over distance and time, and how these affect yield. There is a tradeoff between the cost of implementation and "
        "maintenance vs the benefit of increased yield.")

    params_instructions = (
        "You can incorporate parameters like crop prices and implementation and maintenance costs "
        "provided here in your heuristics. These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
        "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
        "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
        "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}. \n\n"
    )

    halstead_instructions = ("Generate python code with high halstead h1 metric, lower maintainability index, and "
                             "high halstead difficulty. \n\n")

    #syn_farm_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms")
    syn_farm_dir = os.path.join(cfg.disk_dir, "syn_farms")
    all_farms_geojson_path = os.path.join(syn_farm_dir, "farms_cp.geojson")

    farm_ids = [3]
    for farm_id in farm_ids:
        farm_dir = os.path.join(syn_farm_dir, "farm_" + str(farm_id))
        heur_dir = os.path.join(farm_dir, "heuristics")
        if not os.path.exists(heur_dir):
            os.makedirs(heur_dir)
        gen_dir = os.path.join(heur_dir, "generations")
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

        # files = ["input.geojson", "geometry.geojson"]
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
        with open(os.path.join(farm_dir, 'output_gt.geojson')) as f:
            ground_truth = json.load(f)

        # Extract model name from config
        config_model = cfg.lm.split('/')[-1] if '/' in cfg.lm else cfg.lm
        heur_model, fix_model = init_gemini_model(heur_model_name=config_model,
                                                  fix_model_name=config_model)
        if init:
            #create_farm_prompt_file_2(farm_id, all_farms_geojson_path, syn_farm_dir)
            create_init_population(heur_model, fix_model)

        init_population = get_init_population()
        run_evo_strat(init_population)

        print("done")
