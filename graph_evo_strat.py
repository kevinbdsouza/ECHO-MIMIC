import google.generativeai as genai
from rate_limiter import RateLimiter, send_message_with_retry
import random
import math
from create_prompts import create_graph_prompt_file
from radon.raw import analyze
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from tools import *
from dotenv import load_dotenv


# ---------------------------
# Utility Functions
# ---------------------------
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
    system_instructions = ("You are a helpful assistant who is an expert in graph and spatial optimization methods "
                           "and who helps the user with optimization related queries. You will return final answers in python code. "
                           "You can't copy the directions (margin_directions and habitat_directions) given to produce the output, "
                           "they are only for reference to produce the heuristics. And don't assign them by plot_id. "
                           "i.e., do not use plot_ids in the if statements as rules to assign directions. "
                           "Don't invent new variable names, use margin_directions and habitat_directions for directions in the output predictions. "
                           "Do not create any dummy data and dump to input geojson at any cost. "
                           "You should load the input geojson from the existing input.geojson. "
                           "You can only use this input data and no other data. "
                           "Every feature (plot) in the input geojson should have output interventions, don't skip any feature (plot). "
                           "Save outputs to output.json. \n\n")
    if use_template:
        with open(os.path.join(cfg.src_dir, "old", "backup", "heuristic_template_directions.py"), 'r') as f:
            heuristics_template = f.read()
        system_instructions = system_instructions + f"You can use the following heuristic template: ```python{heuristics_template}```. \n\n"

    if halstead_metrics:
        system_instructions = system_instructions + halstead_instructions

    heur_model = genai.GenerativeModel(
        model_name=heur_model_name,
        system_instruction=system_instructions
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
    os.chdir(cfg.src_dir)
    return response


def fix_errors(fix_model, heuristics_code, trace):
    session = fix_model.start_chat(history=[])
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

# ---------------------------
# Candidate Population Functions
# ---------------------------
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

# ---------------------------
# Evaluation Function (using code1’s evaluation of output.json)
# ---------------------------
def evaluate_heuristics(heuristics_code, ground_truth, mode="temp"):
    src_file = os.path.join(heur_dir, mode + ".py")
    with open(src_file, 'w') as f:
        f.write(heuristics_code)

    try:
        os.chdir(heur_dir)
        code, _, err = capture.run_python_script(mode + ".py")
        os.chdir(cfg.src_dir)

        with open(os.path.join(heur_dir, 'output.json')) as f:
            predicted_output = json.load(f)

        # Initialize error accumulators for margin_intervention and habitat_conversion
        margin_errors = []
        habitat_errors = []

        for plot in ground_truth:
            if plot["type"] == "hab_plots":
                continue

            gt_margin_dirs = set(plot["margin_directions"])
            gt_hab_dirs = set(plot["habitat_directions"])

            if "features" in predicted_output:
                predicted_output = predicted_output["features"]

            pred_found = 0
            for pred_feat in predicted_output:
                pred_plot_dict = pred_feat.get("properties", pred_feat)
                if pred_plot_dict["id"] == plot["id"]:
                    pred_margin_dirs = set(pred_plot_dict.get("margin_directions", []))
                    pred_hab_dirs = set(pred_plot_dict.get("habitat_directions", []))
                    margin_error = 1 - (len(gt_margin_dirs.intersection(pred_margin_dirs)) / len(gt_margin_dirs.union(pred_margin_dirs)) if gt_margin_dirs.union(pred_margin_dirs) else 0)
                    habitat_error = 1 - (len(gt_hab_dirs.intersection(pred_hab_dirs)) / len(gt_hab_dirs.union(pred_hab_dirs)) if gt_hab_dirs.union(pred_hab_dirs) else 0)
                    pred_found = 1
                    break

            if not pred_found:
                margin_error = 10
                habitat_error = 10

            margin_errors.append(margin_error)
            habitat_errors.append(habitat_error)

        total_margin_error = np.mean(margin_errors)
        total_habitat_error = np.mean(habitat_errors)
        total_error = total_margin_error + total_habitat_error + 0.01
        return 1 / total_error, heuristics_code
    except Exception as e:
        return 0, heuristics_code

# ---------------------------
# Genetic Operator Helper
# ---------------------------
def copy_parent_candidate(parent):
    return {
        "code": parent["code"],
        "trajectory": parent["trajectory"][:],
        "counts": parent["counts"].copy(),
        "fitness_deltas": parent["fitness_deltas"].copy(),
        "score": 0.0
    }

# ---------------------------
# Genetic Operators (using code1 prompts, operating on candidate dicts)
# ---------------------------
def mutate_heuristics(parent):
    parent_code = parent["code"]
    session = heur_model.start_chat(history=[])
    prompt = f"""Given the following Python code for agricultural plot heuristics:

    ```python
    {parent_code}
    ```
    Suggest a subtle mutation to the heuristics to improve overall performance and logic. 
    The mutation should still result in valid Python code. 
    Focus on small, logical changes related to the plot's properties, geometry, or interactions with neighboring plots.
    You can't use the given directions (margin_directions and habitat_directions) to produce the output, 
    they are only for reference to produce the heuristics. 
    Retain loading input geojson from input.geojson and saving outputs to output.json. 
    Do not create any dummy data and dump to input geojson at any cost. 
    Don't create new variable names. Use variable names margin_directions and habitat_directions for predicted directions in the output. 
    Return the modified Python code for the heuristics. Explain your reasoning and think step by step.  Do not hallucinate. 
    """

    if use_hint:
        prompt += common_task_instructions + params_instructions + hint
    elif halstead_metrics:
        prompt += common_task_instructions + params_instructions + halstead_instructions
    else:
        prompt += common_task_instructions + params_instructions
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
    You can't use the given directions (margin_directions and habitat_directions) to produce the output, they are only for reference to produce the heuristics. 
    Retain loading input geojson from input.geojson and saving outputs to output.json.
    Do not create any dummy data and dump to input geojson at any cost. 
    Don't create new variable names. Use variable names margin_directions and habitat_directions for predicted values in the output. 
    Return the modified Python code for the heuristics. Explain your reasoning and think step by step.  Do not hallucinate. 
    """

    if use_hint:
        prompt += common_task_instructions + params_instructions + hint
    elif halstead_metrics:
        prompt += common_task_instructions + params_instructions + halstead_instructions
    else:
        prompt += common_task_instructions + params_instructions
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_python_code(response)
    new_code = validate_response(response)
    # For determining the dominant parent, we use the longer trajectory
    def pick_parent_for_multi(p1, p2):
        return p1 if len(p1["trajectory"]) >= len(p2["trajectory"]) else p2
    dominant_parent = pick_parent_for_multi(parent1, parent2)
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
You can't use the given directions (margin_directions and habitat_directions) to produce the output, they are only for reference to produce the heuristics. 
Retain loading input geojson from input.geojson and saving outputs to output.json.
Do not create any dummy data and dump to input geojson at any cost. 
Don't create new variable names. Use variable names margin_directions and habitat_directions for predicted values in the output. 
Return the modified Python code for the heuristics. Explain your reasoning and think step by step.  Do not hallucinate. 
"""
    if use_hint:
        prompt += common_task_instructions + params_instructions + hint
    elif halstead_metrics:
        prompt += common_task_instructions + params_instructions + halstead_instructions
    else:
        prompt += common_task_instructions + params_instructions
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_python_code(response)
    new_code = validate_response(response)
    dominant_parent = parent1 if len(parent1["trajectory"]) >= len(parent2["trajectory"]) else parent2
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
    Identify common ideas behind these heuristics. Then, design new heuristics that are based on the common ideas but 
    are as much different as possible from the parents by introducing new parts.
    The new heuristics should still result in valid Python code. 
    You can't use the given directions (margin_directions and habitat_directions) to produce the output, they are only for reference to produce the heuristics.
    Retain loading input geojson from input.geojson and saving outputs to output.json.
    Do not create any dummy data and dump to input geojson at any cost.
    Don't create new variable names. Use variable names margin_directions and habitat_directions for predicted values in the output.
    Return the modified Python code for the heuristics. Explain your reasoning and think step by step.  Do not hallucinate.
    """

    if use_hint:
        prompt += common_task_instructions + params_instructions + hint
    elif halstead_metrics:
        prompt += common_task_instructions + params_instructions + halstead_instructions
    else:
        prompt += common_task_instructions + params_instructions
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    response = extract_python_code(response)
    new_code = validate_response(response)
    dominant_parent = parent1 if len(parent1["trajectory"]) >= len(parent2["trajectory"]) else parent2
    child = copy_parent_candidate(dominant_parent)
    child["code"] = new_code
    child["pending_op"] = "evolve_2"
    child["pending_parent_score"] = dominant_parent["score"]
    return child


def reflect(top_heuristics, top_scores):
    session = heur_model.start_chat(history=[])
    heuristics_info = ""
    for i, (heur, score) in enumerate(zip(top_heuristics, top_scores), start=1):
        heuristics_info += f"Heuristic {i} (Fitness Score: {score}):\n{heur['code']}\n\n"
        prompt = f"""You are an expert in spatial optimization and agricultural heuristics.
    Based on the following top 5 heuristics and their corresponding fitness scores: {heuristics_info}
    
    Please analyze these heuristics and craft a new heuristic that is expected to have increased fitness.
    Ensure that the new heuristic:
    - Is valid Python code.
    - Loads the input geojson from 'input.geojson' and writes the output to 'output.json'.
    - Do not create any dummy data and dump to input geojson at any cost.
    - Uses the variable names margin_directions and habitat_directions for the predicted directions.
    
    Explain your reasoning step by step and return the complete Python code for the new heuristic.
    """

    if use_hint:
        prompt += common_task_instructions + params_instructions + hint
    elif halstead_metrics:
        prompt += common_task_instructions + params_instructions + halstead_instructions
    else:
        prompt += common_task_instructions + params_instructions
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    new_code = validate_response(extract_python_code(response))
    best_candidate = max(top_heuristics, key=lambda c: c["score"])
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
    # Use the top 5 candidates
    top5 = sorted_population[:5]
    top5_scores = sorted_scores[:5]
    child1 = reflect(top5, top5_scores)
    reflect_results.append(child1)

    # Additional runs: choose 5 random candidates
    for run in range(4):
        if len(sorted_population) >= 5:
            random_candidates = random.sample(sorted_population, 5)
        else:
            random_candidates = sorted_population
        try:
            childX = reflect(random_candidates, [c["score"] for c in random_candidates])
            reflect_results.append(childX)
        except Exception as e:
            print(e)

    reflect_results = evaluate_population(reflect_results, ground_truth)
    reflect_scores = [cand["score"] for cand in reflect_results]
    return reflect_scores, reflect_results


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

# ---------------------------
# Initialization and Population Creation
# ---------------------------
def run_gemini_flashexp(prompt_path, heuristics_file, init_model):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    session = init_model.start_chat(history=[])
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    python_code = extract_python_code(response)
    with open(heuristics_file, 'w') as f:
        f.write(python_code)


def create_init_population(init_model, fix_model):
    prompt_path = os.path.join(conn_dir, "prompt_input.txt")
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
        os.chdir(cfg.src_dir)

# ---------------------------
# Radon Metrics Computation (candidate version)
# ---------------------------
def compute_radon_metrics(old_df, population):
    new_rows = []
    for candidate in population:
        code = candidate["code"]
        score = candidate["score"]
        try:
            raw_metrics = analyze(code)
        except Exception as e:
            continue
        try:
            complexities = cc_visit(code)
        except Exception as e:
            complexities = 0
        if complexities:
            avg_cyclomatic_complexity = sum(c.complexity for c in complexities) / len(complexities)
        else:
            avg_cyclomatic_complexity = 0.0
        try:
            halstead_results = h_visit(code)
        except Exception as e:
            halstead_results = []
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
            total_h1 = total_h2 = total_N1 = total_N2 = 0
            total_vocabulary = total_length = total_volume = 0
            total_difficulty = total_effort = total_time = total_bugs = 0
        try:
            mi_value = mi_visit(code, multi=True)
        except Exception as e:
            mi_value = 0
        row = {
            'fitness_score': score,
            'loc': raw_metrics.loc,
            'lloc': raw_metrics.lloc,
            'sloc': raw_metrics.sloc,
            'comment': raw_metrics.comments,
            'multi': raw_metrics.multi,
            'blank': raw_metrics.blank,
            'avg_cyclomatic_complexity': avg_cyclomatic_complexity,
            'maintainability_index': mi_value,
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
    new_df = pd.DataFrame(new_rows)
    metrics_df = pd.concat([old_df, new_df], ignore_index=True)
    return metrics_df

# ---------------------------
# Plotting and CSV Saving Functions (from code2)
# ---------------------------
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
                op_name = t
                delta_str = "0.0"
            else:
                op_name = t[:idx].strip()
                inside = t[idx:].strip("()")
                delta_str = inside
            try:
                delta_val = float(delta_str)
            except:
                delta_val = 0.0
            ops.append(op_name)
            deltas.append(delta_val)
        x_vals = list(range(len(ops)))
        y_op = [operator_map.get(op, -1) for op in ops]
        plt.figure()
        plt.plot(x_vals, y_op, marker='o')
        plt.title(f"Best Candidate Trajectory (Gen {gen})\nScore={score:.4f}")
        plt.xlabel("Trajectory Step")
        plt.ylabel("Operators")
        plt.yticks(list(range(len(base_op_list))), base_op_list)
        for i, (x, y) in enumerate(zip(x_vals, y_op)):
            plt.text(x, y, f"{deltas[i]:+.3f}", fontsize=9, ha='left', va='bottom')
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
        plt.axhline(y=0.0, color='gray', linestyle='--')
        plt.title(f"Operator Cumulative Deltas (Gen {gen})\nScore={score:.4f}")
        plt.xlabel("Operator")
        plt.ylabel("Cumulative Fitness Delta")
        plt.tight_layout()
        outname = os.path.join(gen_dir, f"best_fitness_deltas_gen_{gen}.png")
        plt.savefig(outname)
        plt.close()


def plot_population_operator_stats(population, generation, gen_dir):
    operator_list = ["init", "mutate", "crossover", "evolve_1", "evolve_2", "reflect"]
    usage_sums = {op: 0 for op in operator_list}
    delta_sums = {op: 0.0 for op in operator_list}
    for cand in population:
        for op in operator_list:
            usage_sums[op] += cand["counts"].get(op, 0)
            delta_sums[op] += cand["fitness_deltas"].get(op, 0.0)
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
    rows = []
    for i, cand in enumerate(population):
        row = {
            "generation": generation,
            "candidate_id": i,
            "score": cand["score"],
            "trajectory": json.dumps(cand["trajectory"]),
            "counts": json.dumps(cand["counts"]),
            "fitness_deltas": json.dumps(cand["fitness_deltas"]),
            "code": cand["code"]
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(gen_dir, f"population_gen_{generation}.csv")
    df.to_csv(csv_path, index=False)


def save_best_history_to_csv(best_history, gen_dir):
    rows = []
    for entry in best_history:
        row = {
            "generation": entry["generation"],
            "best_score": entry["score"],
            "trajectory": json.dumps(entry["trajectory"]),
            "counts": json.dumps(entry["counts"]),
            "fitness_deltas": json.dumps(entry["fitness_deltas"])
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(gen_dir, "best_history.csv")
    df.to_csv(csv_path, index=False)

# ---------------------------
# Evolutionary Strategy
# ---------------------------
def run_evo_strat(population):
    # Evaluate initial population (candidate dicts)
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
                 'halstead_vocabulary', 'halstead_length', 'halstead_volume', 'halstead_difficulty',
                 'halstead_effort', 'halstead_time', 'halstead_bugs'])
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
        for i in range(population_size):
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
        save_population_to_csv(combined_pop, generation + 1, gen_dir)
        save_best_history_to_csv(best_history, gen_dir)
        plot_best_trajectory_across_generations(best_history, gen_dir)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    cfg = Config()
    capture = CommandOutputCapture()
    population_size = 25
    num_generations = 25
    init = True
    use_hint = True
    use_template = False
    halstead_metrics = False

    common_task_instructions = (
        "The python programs are trying to solve the task of deciding which interventions need to be done at which agricultural plots "
        "(crops, type='ag_plot') and how to place them geometrically based on how the interventions increase ecological connectivity, "
        "while not decreasing NPV from a baseline value. "
        "The choice is between margin (convert only the margins) and habitat (convert a contiguous region) interventions. "
        "The margin interventions are chosen among the following directions on the boundary: "
        "north-east, north-west, south-east, south-west. The habitat conversions "
        "are chosen among the same directions in the internal area of polygons. "
        "The directions are computed by running a horizontal and a vertical line through the centre of each plot, and "
        "choosing them if they have interventions (as computed by IPOPT optimization) greater than a threshold. "
        "Existing habitat plots (type='hab_plots') remain unaffected. "
        "Integral index of connectivity (IIC) is used as the metric for ecological connectivity, which tries to increase the "
        "size of the connected components in the neighbourhood. It promotes fractions touching each other and extending the "
        "connectivity between existing habitats in the landscape, which includes the farm and its neighbours. "
        "There is a tradeoff between maximizing connectivity and maintaining NPV. "
        "NPV is calculated based on how the interventions affect pollination and pest control "
        "services over distance and time, and how these affect yield. There is also the tradeoff between the cost of implementation and "
        "maintenance vs the benefit of increased yield. Look at properties that you think have a pattern "
        "(like yield, label, type, nbs, geometry, do not use plot_ids to assign rules), "
        "and the relative positioning of both the farm neighbours with respect to your farm and "
        "the plots with respect to each other within the farm in the context of ecological connectivity. "
        "Use all the given geometry information to infer these geographical relationships. \n\n"
    )

    params_instructions = (
        "You can incorporate parameters like crop prices and implementation and maintenance costs "
        "provided here in your heuristics. These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
        "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
        "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
        "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}. \n\n"
    )

    halstead_instructions = ("Generate python code with high halstead metrics like h1, h2, N1, N2, volume, "
                             "difficulty, length, effort, and vocabulary.\n\n")

    farm_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms")
    all_farms_geojson_path = os.path.join(farm_dir, "farms_cp.geojson")

    farm_ids = [3]
    for farm_id in farm_ids:
        print(f"Running farm:{farm_id}")
        conn_dir = os.path.join(farm_dir, f"farm_{farm_id}", "connectivity")
        heur_dir = os.path.join(conn_dir, "heuristics")
        if not os.path.exists(heur_dir):
            os.makedirs(heur_dir)
        gen_dir = os.path.join(conn_dir, "generations")
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

        files = ["input.geojson"]
        for file in files:
            inp_src = os.path.join(conn_dir, file)
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

        with open(os.path.join(conn_dir, 'output_gt_directions.json')) as f:
            ground_truth = json.load(f)

        hint = (
            f"Your hint is that the ground truth output.json for your farm is this: {ground_truth}. "
            "Don't copy these directly, and don't assign them by plot id, they are just to help you "
            "compare and come up with the right heuristics. "
        )

        # Extract model name from config
        config_model = cfg.lm.split('/')[-1] if '/' in cfg.lm else cfg.lm
        heur_model, fix_model = init_gemini_model(heur_model_name=config_model,
                                                  fix_model_name=config_model)
        if init:
            create_graph_prompt_file(farm_id, all_farms_geojson_path, farm_dir, use_hint=use_hint)
            create_init_population(heur_model, fix_model)

        init_population = get_init_population()
        run_evo_strat(init_population)

        print("done")
