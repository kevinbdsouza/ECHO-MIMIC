import argparse
from typing import List, Optional
import google.generativeai as genai
import os
import shutil
import random
from pathlib import Path
from echo_mimic.tools import *
import math
from echo_mimic.prompts import create_farm_prompt_file_2
import pandas as pd
from echo_mimic.common.metrics import compute_radon_metrics_from_population as compute_radon_metrics
from echo_mimic.config import Config
from echo_mimic.common.rate_limiter import send_message_with_retry
from echo_mimic.common import (
    build_model,
    configure_genai,
    ensure_rate_limiter,
    extract_python_code,
    is_openai_model,
    make_code_validator,
)
from echo_mimic.common.operators import (
    make_operator_counts,
    make_operator_deltas,
    plot_best_trajectory_across_generations,
    plot_population_operator_stats,
)
from echo_mimic.prompts.instructions import (
    FARM_COMMON_TASK_INSTRUCTIONS,
    FARM_HALSTEAD_INSTRUCTIONS,
    FARM_PARAMS_INSTRUCTIONS,
)


cfg = Config()
rate_limiter = ensure_rate_limiter(cfg)
capture = CommandOutputCapture()
heur_model = None
fix_model = None

def init_gemini_model(heur_model_name="gemini-2.0-flash-thinking-exp-01-21", fix_model_name="gemini-2.0-flash-thinking-exp-01-21"):
    if not (is_openai_model(heur_model_name) and is_openai_model(fix_model_name)):
        configure_genai()

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

    heur_model = build_model(
        heur_model_name,
        system_instructions,
        ensure_configured=False,
    )
    fix_model = build_model(
        fix_model_name,
        "You are a helpful assistant who is an expert in spatial optimization methods and python. "
        "Given the python code and the stack traceback, fix the errors and return the correct functioning python code.",
        ensure_configured=False,
    )

    global rate_limiter
    rate_limiter = ensure_rate_limiter(cfg)

    return heur_model, fix_model


def _validate_code_with_retry(
    code: str,
    *,
    script_name: str = "temp_fix.py",
    max_attempts: int = 2,
) -> str:
    validator = make_code_validator(
        workdir=Path(heur_dir),
        capture=capture,
        fix_model=fix_model,
        rate_limiter=rate_limiter,
        default_script=script_name,
        default_attempts=max_attempts,
    )
    return validator(code, script_name=script_name, max_attempts=max_attempts)


def validate_response(response):
    return _validate_code_with_retry(response)


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


def generate_candidate_from_model(prompt_path, heuristics_file, init_model):
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
        try:
            generate_candidate_from_model(prompt_path, heur_file, init_model)
        except Exception as e:
            print(e)
            continue

        with open(heur_file, 'r') as f:
            heuristics_code = f.read()

        heuristics_code = _validate_code_with_retry(
            heuristics_code,
            script_name=f"heuristics_gem_{i}.py",
            max_attempts=3,
        )

        with open(heur_file, 'w') as f:
            f.write(heuristics_code)


def get_init_population():
    init_population = []
    for i in range(1, population_size + 1):
        heuristic_file = os.path.join(heur_dir, "heuristics_gem_" + str(i) + ".py")
        with open(heuristic_file, 'r') as f:
            heuristics_code = f.read()

        counts = make_operator_counts()
        deltas = make_operator_deltas()

        candidate = {
            "code": heuristics_code,
            "trajectory": ["init(0.0000)"],
            "counts": counts,
            "fitness_deltas": deltas,
            "score": 0.0
        }
        candidate["counts"]["init"] = 1
        init_population.append(candidate)
    return init_population



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

def run(
    *,
    population_size_value: int = 25,
    num_generations_value: int = 25,
    inner_loop_size_value: int = 25,
    agent_id: Optional[int] = None,
    init_value: bool = True,
    use_template_value: bool = False,
    halstead_metrics_value: bool = False,
    model_name: Optional[str] = None,
) -> None:
    """Execute the farm evolutionary strategy workflow with configurable parameters."""

    global cfg, capture, population_size, num_generations, inner_loop_size
    global init, use_template, halstead_metrics
    global common_task_instructions, params_instructions, halstead_instructions
    global heur_dir, gen_dir, score_file, metrics_file, input_json, ground_truth, farm_dir
    global heur_model, fix_model

    cfg = Config()
    capture = CommandOutputCapture()
    population_size = population_size_value
    num_generations = num_generations_value
    inner_loop_size = inner_loop_size_value
    init = init_value
    use_template = use_template_value
    halstead_metrics = halstead_metrics_value

    common_task_instructions = FARM_COMMON_TASK_INSTRUCTIONS
    params_instructions = FARM_PARAMS_INSTRUCTIONS
    halstead_instructions = FARM_HALSTEAD_INSTRUCTIONS

    target_farm_id = agent_id if agent_id is not None else 3
    target_farm_ids = [target_farm_id]

    syn_farm_dir = cfg.farms_dir
    all_farms_geojson_path = os.path.join(syn_farm_dir, "farms_cp.geojson")

    for farm_identifier in target_farm_ids:
        farm_dir = os.path.join(syn_farm_dir, f"farm_{farm_identifier}")
        heur_dir = os.path.join(farm_dir, "heuristics")
        if not os.path.exists(heur_dir):
            os.makedirs(heur_dir)
        gen_dir = os.path.join(heur_dir, "generations")
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

        with open(os.path.join(farm_dir, 'output_gt.geojson')) as f:
            ground_truth = json.load(f)

        resolved_model = (model_name or cfg.lm)
        heur_model, fix_model = init_gemini_model(
            heur_model_name=resolved_model,
            fix_model_name=resolved_model,
        )
        if init:
            create_init_population(heur_model, fix_model)

        init_population = get_init_population()
        run_evo_strat(init_population)

        print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the farm evolutionary strategy workflow.")
    parser.add_argument("--population-size", type=int, default=25, help="Population size for the evolutionary loop.")
    parser.add_argument("--num-generations", type=int, default=25, help="Number of generations to evolve.")
    parser.add_argument("--inner-loop-size", type=int, default=25, help="Number of offspring attempts per generation.")
    parser.add_argument("--agent-id", type=int, default=3, help="Target farm/agent ID to process.")
    parser.add_argument("--no-init", action="store_true", help="Skip regenerating the initial population with Gemini.")
    parser.add_argument("--use-template", action="store_true", help="Enable template usage for prompt construction.")
    parser.add_argument("--halstead-metrics", action="store_true", help="Enable halstead-focused instructions.")

    args = parser.parse_args()

    run(
        population_size_value=args.population_size,
        num_generations_value=args.num_generations,
        inner_loop_size_value=args.inner_loop_size,
        agent_id=args.agent_id,
        init_value=not args.no_init,
        use_template_value=args.use_template,
        halstead_metrics_value=args.halstead_metrics,
    )
