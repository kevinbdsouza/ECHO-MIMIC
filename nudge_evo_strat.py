import argparse
from typing import List, Optional
import google.generativeai as genai
import json
import os
import random
import shutil
from pathlib import Path
from echo_mimic.tools import *
import math
from echo_mimic.prompts import create_nudge_prompt_file
import pandas as pd
from echo_mimic.metrics import compute_radon_metrics
from echo_mimic.config import Config
import numpy as np
from echo_mimic.rate_limiter import send_message_with_retry
from echo_mimic.common import (
    build_model,
    configure_genai,
    ensure_rate_limiter,
    extract_message,
    extract_python_code,
    make_code_validator,
)


cfg = Config()
rate_limiter = ensure_rate_limiter(cfg)
capture = CommandOutputCapture()


def get_init_population():
    init_population = []
    for i in range(1, population_size + 1):
        heuristic_file = os.path.join(heur_dir, "heuristics_gem_" + str(i) + ".txt")
        with open(heuristic_file, 'r') as f:
            heuristic_message = f.read()

        init_population.append(heuristic_message)
    return init_population


def init_gemini_model(policy_model_name="gemini-2.0-flash-thinking-exp-01-21", farm_model_name="gemini-2.0-flash-thinking-exp-01-21",
                      fix_model_name="gemini-2.0-flash-thinking-exp-01-21"):
    configure_genai()

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

    policy_model = build_model(
        policy_model_name,
        policy_system_instructions,
        ensure_configured=False,
    )
    farm_model = build_model(
        farm_model_name,
        farm_system_instructions,
        ensure_configured=False,
    )
    fix_model = build_model(
        fix_model_name,
        "You are a helpful assistant who is an expert in graph and spatial optimization methods and python. "
        "Given the python code and the stack traceback, fix the errors and return the correct functioning python code.",
        ensure_configured=False,
    )

    global rate_limiter
    rate_limiter = ensure_rate_limiter(cfg)

    return policy_model, farm_model, fix_model


def _validate_code_with_retry(
    code: str,
    *,
    script_name: str = "temp_fix.py",
    max_attempts: int = 2,
    pre_run: Optional[Callable[[Path], None]] = None,
) -> str:
    validator = make_code_validator(
        workdir=Path(heur_dir),
        capture=capture,
        fix_model=fix_model,
        rate_limiter=rate_limiter,
        default_script=script_name,
        default_attempts=max_attempts,
    )
    return validator(
        code,
        script_name=script_name,
        max_attempts=max_attempts,
        pre_run=pre_run,
    )


def validate_response(response):
    return _validate_code_with_retry(response)


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




def run(
    *,
    population_size_value: int = 25,
    num_generations_value: int = 10,
    inner_loop_size_value: int = 10,
    farm_ids: Optional[List[int]] = None,
    init_value: bool = True,
) -> None:
    """Execute the nudge evolutionary strategy workflow with configurable parameters."""

    global cfg, capture, population_size, num_generations, inner_loop_size, init
    global params_instructions, policy_task_instructions
    global farm_dir, conn_dir, nudge_dir, heur_dir, gen_dir
    global score_file, metrics_file, input_json, ground_truth, hybrid_output, eco_intens_output

    cfg = Config()
    capture = CommandOutputCapture()
    population_size = population_size_value
    num_generations = num_generations_value
    inner_loop_size = inner_loop_size_value
    init = init_value

    params_instructions = (
        "These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
        "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
        "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
        "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}. \n\n"
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
        "Your final message to the farmer should be in this format \\communication{message}. \n\n"
    )

    farms_dir = cfg.farms_dir
    target_farm_ids = farm_ids or [3]

    for farm_identifier in target_farm_ids:
        print(f"Running farm:{farm_identifier}")
        farm_dir = os.path.join(farms_dir, f"farm_{farm_identifier}")
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

        with open(os.path.join(conn_dir, 'output_hybrid_heu.py')) as f:
            hybrid_heuristics = f.read()

        with open(os.path.join(conn_dir, 'output_gt.geojson')) as f:
            ground_truth = json.load(f)

        with open(os.path.join(conn_dir, 'output_hybrid.geojson')) as f:
            hybrid_output = json.load(f)

        with open(os.path.join(conn_dir, 'output_eco_intens.geojson')) as f:
            eco_intens_output = json.load(f)

        policy_model, fix_model = init_gemini_model()
        if init:
            create_init_population(policy_model)

        with open(os.path.join(heur_dir, "prompt_input.txt"), "r") as f:
            farm_prompt = f.read()

        best_policy_file = os.path.join(gen_dir, "best_policy_gem_gen_0.txt")
        with open(best_policy_file, 'w') as f:
            f.write(farm_prompt)

        population = get_init_population()
        run_evo_strat(population)

        print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the nudge evolutionary strategy workflow.")
    parser.add_argument("--population-size", type=int, default=25, help="Population size for the evolutionary loop.")
    parser.add_argument("--num-generations", type=int, default=10, help="Number of generations to evolve.")
    parser.add_argument("--inner-loop-size", type=int, default=10, help="Number of offspring attempts per generation.")
    parser.add_argument("--farm-ids", type=int, nargs="+", default=[3], help="Target farm IDs to process.")
    parser.add_argument("--no-init", action="store_true", help="Skip regenerating the initial population with Gemini.")

    args = parser.parse_args()

    run(
        population_size_value=args.population_size,
        num_generations_value=args.num_generations,
        inner_loop_size_value=args.inner_loop_size,
        farm_ids=args.farm_ids,
        init_value=not args.no_init,
    )
