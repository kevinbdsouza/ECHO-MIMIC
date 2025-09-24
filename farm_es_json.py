import google.generativeai as genai
from rate_limiter import RateLimiter, send_message_with_retry
import random
# Removed: import ast - No longer evaluating Python code strings
from tools import *
import math
from concurrent.futures import ThreadPoolExecutor
from create_prompts import create_farm_prompt_file_2
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv


def extract_geojson_content(text):
    """
    Extracts GeoJSON content from text, assuming it might be included as a code block:

        ```geojson
        { ... }
        ```

    or possibly with ```json. We give preference to the geojson block if found.

    If no code block is found, but the entire string might be raw JSON, we do a
    cursory check to see if it starts with '{' and ends with '}' and return it if so.
    Otherwise returns None.
    """
    pattern_geojson = r'```geojson(.*?)```'
    match_geojson = re.search(pattern_geojson, text, re.DOTALL | re.IGNORECASE)

    pattern_json = r'```json(.*?)```'
    match_json = re.search(pattern_json, text, re.DOTALL | re.IGNORECASE)

    if match_geojson:
        return match_geojson.group(1).strip()
    elif match_json:
        # If ```geojson is not found, try ```json
        return match_json.group(1).strip()
    else:
        stripped_text = text.strip()
        if stripped_text.startswith('{') and stripped_text.endswith('}'):
            return stripped_text
        else:
            print("Warning: Could not extract GeoJSON content from LLM response.")
            return None


def init_gemini_model(heur_model_name="gemini-1.5-flash-latest", fix_model_name="gemini-1.5-flash-latest"):
    """
    Initializes two GenerativeModels from google.generativeai for:
      - heur_model: used to generate or evolve solutions with chain-of-thought
      - fix_model: used to correct malformed GeoJSON

    We instruct the heur_model to produce reasoning plus the final GeoJSON in a code block,
    and the fix_model to produce a corrected GeoJSON in a code block.
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variables
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("No API key found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")
    
    genai.configure(api_key=api_key)

    # -- System instructions for Heuristic Generation: produce reasoning + final GeoJSON --
    system_instructions = (
        "You are an expert assistant specializing in spatial optimization for agriculture. "
        "You will propose interventions (margin_intervention, habitat_conversion ∈ [0.0, 1.0]) for each 'ag_plot'.\n\n"
        "IMPORTANT:\n"
        "1) Provide your step-by-step reasoning (chain-of-thought) in natural language.\n"
        "2) After that reasoning, conclude your response with the final GeoJSON in a triple-backtick block labeled 'geojson'.\n"
        "   We'll parse and evaluate ONLY the content within that final triple-backtick block.\n"
        "3) The final GeoJSON MUST be a valid FeatureCollection with a feature for each 'ag_plot' from the conceptual 'input.geojson', "
        "   containing original 'id' plus 'margin_intervention' and 'habitat_conversion' in 'properties'.\n"
        "4) Existing 'hab_plots' may remain unchanged or be included as-is.\n\n"
        "Your chain-of-thought can consider plot areas, crop types, adjacency, or cost parameters. "
        "But ONLY the final code block will be used for subsequent validation.\n\n"
        "Reference Economic Parameters:\n"
        "{params_instructions}\n"
    )

    # -- System instructions for Fixing Invalid GeoJSON --
    fix_system_instructions = (
        "You are an expert assistant tasked with fixing invalid GeoJSON data based on validation errors. "
        "You receive a malformed GeoJSON plus a list of validation errors. "
        "Provide a corrected final GeoJSON, in a triple-backtick 'geojson' code block. "
        "Do not include extra text or explanation outside the code block."
    )

    # Build GenerativeModels
    heur_model = genai.GenerativeModel(
        model_name=heur_model_name,
        system_instruction=system_instructions
    )
    fix_model = genai.GenerativeModel(
        model_name=fix_model_name,
        system_instruction=fix_system_instructions
    )
    # Initialize rate limiter
    from config import Config
    cfg = Config()
    global rate_limiter
    rate_limiter = RateLimiter(**cfg.rate_limit)
    
    return heur_model, fix_model


def validate_geojson(geojson_string, input_plots_info):
    """
    Validates a candidate GeoJSON string. Checks:
      1) JSON parse
      2) Root is a FeatureCollection
      3) Each feature is a valid Feature
      4) All 'ag_plot' IDs from input_plots_info appear, each with margin/habitat in [0,1].

    Returns (is_valid, errors, parsed_geojson).
    """
    errors = []
    parsed_geojson = None

    # 1) Parse as JSON
    try:
        if not geojson_string or not geojson_string.strip():
            errors.append("GeoJSON content is empty or missing.")
            return False, errors, None
        parsed_geojson = json.loads(geojson_string)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON format: {e}")
        return False, errors, None
    except Exception as e:
        errors.append(f"Unexpected parsing error: {e}")
        return False, errors, None

    # 2) Basic structure
    if not isinstance(parsed_geojson, dict):
        errors.append("Root is not a JSON object.")
    else:
        if parsed_geojson.get("type") != "FeatureCollection":
            errors.append("Missing or incorrect 'type': 'FeatureCollection'.")
        if "features" not in parsed_geojson:
            errors.append("Missing 'features' array.")
        else:
            features = parsed_geojson["features"]
            if not isinstance(features, list):
                errors.append("'features' is not a list.")
            else:
                output_plot_ids = set()
                for i, feature in enumerate(features):
                    if not isinstance(feature, dict):
                        errors.append(f"Feature at index {i} is not an object.")
                        continue
                    if feature.get("type") != "Feature":
                        errors.append(f"Feature at index {i} missing or invalid 'type': 'Feature'.")
                    props = feature.get("properties")
                    if props is None or not isinstance(props, dict):
                        errors.append(f"Feature at index {i} missing or invalid 'properties'.")
                        continue
                    fid = props.get("id")
                    if fid is None:
                        # Not an ag_plot if no ID, but let's note it
                        continue
                    if fid in input_plots_info:
                        output_plot_ids.add(fid)
                        margin = props.get("margin_intervention")
                        habitat = props.get("habitat_conversion")

                        # Check margin/habitat in [0,1] if present
                        if margin is None:
                            errors.append(f"ag_plot {fid} missing margin_intervention.")
                        elif not isinstance(margin, (int, float)) or not (0.0 <= margin <= 1.0):
                            errors.append(f"ag_plot {fid} invalid margin_intervention={margin} (must be 0-1).")

                        if habitat is None:
                            errors.append(f"ag_plot {fid} missing habitat_conversion.")
                        elif not isinstance(habitat, (int, float)) or not (0.0 <= habitat <= 1.0):
                            errors.append(f"ag_plot {fid} invalid habitat_conversion={habitat} (must be 0-1).")

                missing = set(input_plots_info.keys()) - output_plot_ids
                if missing:
                    errors.append(f"Missing intervention for ag_plot IDs: {missing}")

    is_valid = (len(errors) == 0)
    return is_valid, errors, parsed_geojson


def fix_geojson_errors(fix_model, invalid_geojson_content, validation_errors):
    """
    Attempts to fix invalid_geojson_content using fix_model and the provided validation_errors.
    We instruct the LLM to produce only corrected GeoJSON in a triple-backtick block.
    """
    session = fix_model.start_chat(history=[])

    error_string = "\n".join(f"- {e}" for e in validation_errors)
    prompt = f"""The following GeoJSON is invalid:

{invalid_geojson_content}

Validation errors were:
{error_string}

Please produce a corrected GeoJSON. Provide only the final fixed GeoJSON in a code block:

```geojson
{{ your corrected feature collection }}
```
"""

    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        if not completion.parts:
            print("Error: fix_model response is empty or blocked.")
            return None
        response_text = completion.parts[0].text.strip()
        fixed_geojson = extract_geojson_content(response_text)
        if not fixed_geojson:
            print("Warning: Could not parse corrected GeoJSON from fix_model response.")
            return None
        return fixed_geojson
    except Exception as e:
        print(f"Error calling fix_model: {e}")
        return None


def attempt_validate_and_fix(llm_output_geojson, input_plots_info, fix_model, max_tries=2):
    """
    Validates llm_output_geojson. If invalid, tries to fix up to max_tries times.
    Returns either a valid GeoJSON string or None if it can't be fixed.
    """
    current_geojson = llm_output_geojson
    attempt = 0
    while attempt < max_tries:
        is_valid, errors, _ = validate_geojson(current_geojson, input_plots_info)
        if is_valid:
            return current_geojson
        else:
            print(f"GeoJSON validation failed (attempt {attempt + 1}/{max_tries}): {errors}")
            fixed = fix_geojson_errors(fix_model, current_geojson, errors)
            if not fixed:
                return None
            current_geojson = fixed
        attempt += 1

    # Final check
    is_valid, errors, _ = validate_geojson(current_geojson, input_plots_info)
    return current_geojson if is_valid else None


def evaluate_heuristics(geojson_content, ground_truth_data, input_plots_info):
    """
    Compares predicted interventions to ground truth. Uses mean absolute error across margin/habitat, then
    fitness = 1 / (total_error + epsilon).

    Returns (fitness_score, geojson_content).
    """
    try:
        predicted_output = json.loads(geojson_content)
        gt_features = ground_truth_data["features"]
        pred_features = predicted_output["features"]

        input_ag_ids = set(input_plots_info.keys())
        pred_map = {}

        for pf in pred_features:
            props = pf.get("properties", {})
            fid = props.get("id")
            if fid in input_ag_ids:
                pred_map[fid] = {
                    "margin_intervention": props.get("margin_intervention", 0.0),
                    "habitat_conversion": props.get("habitat_conversion", 0.0)
                }

        margin_errors = []
        habitat_errors = []

        for gt_feat in gt_features:
            props = gt_feat.get("properties", {})
            fid = props.get("id")
            if fid in input_ag_ids:
                gt_margin = props.get("margin_intervention", 0.0)
                gt_hab = props.get("habitat_conversion", 0.0)

                if fid in pred_map:
                    pm = pred_map[fid]["margin_intervention"]
                    ph = pred_map[fid]["habitat_conversion"]
                    margin_errors.append(abs(gt_margin - pm))
                    habitat_errors.append(abs(gt_hab - ph))
                else:
                    # Missing predicted, treat as error from zero
                    margin_errors.append(abs(gt_margin - 0.0))
                    habitat_errors.append(abs(gt_hab - 0.0))

        if not margin_errors and not habitat_errors:
            # No ag plots?
            total_error = 0.0
        else:
            total_margin_error = np.mean(margin_errors) if margin_errors else 0.0
            total_hab_error = np.mean(habitat_errors) if habitat_errors else 0.0
            total_error = total_margin_error + total_hab_error

        epsilon = 0.01
        fitness = 1.0 / (total_error + epsilon)
        if not math.isfinite(fitness):
            fitness = 0.0
        return fitness, geojson_content

    except Exception as e:
        print(f"Error in evaluate_heuristics: {e}")
        return 0.0, geojson_content


def pick_parent_for_multi(parent1, parent2):
    """
    For certain operators, we track the parent's trajectory. We pick whichever parent
    has a longer trajectory to maintain more history.
    """
    if len(parent1["trajectory"]) >= len(parent2["trajectory"]):
        return parent1
    else:
        return parent2


def copy_parent_candidate(parent):
    """
    Creates a new candidate dict copying parent's data, but resets the new child's score to 0.
    """
    return {
        "geojson": parent["geojson"],
        "trajectory": parent["trajectory"][:],
        "counts": parent["counts"].copy(),
        "fitness_deltas": parent["fitness_deltas"].copy(),
        "score": 0.0
    }


def mutate_heuristics(parent, input_plots_info, heur_model, fix_model):
    """
    Calls heur_model to produce a slightly mutated GeoJSON (with reasoning),
    then validates/fixes it. Returns a new candidate or None on failure.
    """
    parent_geojson = parent["geojson"]
    session = heur_model.start_chat(history=[])

    prompt = f"""You have an existing solution (GeoJSON) for ag_plots:

{parent_geojson}

First, show your reasoning on how to slightly change (mutate) 'margin_intervention' or 'habitat_conversion' for one or more plots.
Then provide your final mutated GeoJSON in a code block:

```geojson
{{ your mutated FeatureCollection }}
```
"""

    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        if not completion.parts:
            print("Mutation LLM response is empty/blocked.")
            return None

        response_text = completion.parts[0].text.strip()
        mutated_geojson = extract_geojson_content(response_text)
        if not mutated_geojson:
            return None

        validated = attempt_validate_and_fix(mutated_geojson, input_plots_info, fix_model)
        if validated:
            child = copy_parent_candidate(parent)
            child["geojson"] = validated
            child["pending_op"] = "mutate"
            child["pending_parent_score"] = parent["score"]
            return child
        else:
            return None
    except Exception as e:
        print(f"Error in mutate_heuristics: {e}")
        return None


def crossover_heuristics(parent1, parent2, input_plots_info, heur_model, fix_model):
    """
    Calls heur_model to produce a crossover solution. Returns a new candidate or None.
    """
    session = heur_model.start_chat(history=[])

    prompt = f"""Combine the strategies from these two GeoJSON solutions:

Parent 1:
{parent1["geojson"]}

Parent 2:
{parent2["geojson"]}

In your reasoning, decide how to merge or average interventions. Then provide a final GeoJSON in a code block:

```geojson
{{ your merged FeatureCollection }}
```
"""

    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        if not completion.parts:
            print("Crossover LLM response is empty/blocked.")
            return None

        response_text = completion.parts[0].text.strip()
        cross_geojson = extract_geojson_content(response_text)
        if not cross_geojson:
            return None

        validated = attempt_validate_and_fix(cross_geojson, input_plots_info, fix_model)
        if validated:
            dominant_parent = pick_parent_for_multi(parent1, parent2)
            child = copy_parent_candidate(dominant_parent)
            child["geojson"] = validated
            child["pending_op"] = "crossover"
            child["pending_parent_score"] = dominant_parent["score"]
            return child
        else:
            return None
    except Exception as e:
        print(f"Error in crossover_heuristics: {e}")
        return None


def evolve_heuristics_1(parent1, parent2, input_plots_info, heur_model, fix_model):
    """
    Asks heur_model to generate a new solution that's significantly different from both parents.
    """
    session = heur_model.start_chat(history=[])

    prompt = f"""We have two existing solutions:

Parent 1:
{parent1["geojson"]}

Parent 2:
{parent2["geojson"]}

Reason about creating a new, significantly different solution. Then provide final GeoJSON in a block:

```geojson
{{ your new FeatureCollection }}
```
"""

    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        if not completion.parts:
            print("Evolve_1 LLM response empty/blocked.")
            return None

        response_text = completion.parts[0].text.strip()
        evo_geojson = extract_geojson_content(response_text)
        if not evo_geojson:
            return None

        validated = attempt_validate_and_fix(evo_geojson, input_plots_info, fix_model)
        if validated:
            dominant_parent = pick_parent_for_multi(parent1, parent2)
            child = copy_parent_candidate(dominant_parent)
            child["geojson"] = validated
            child["pending_op"] = "evolve_1"
            child["pending_parent_score"] = dominant_parent["score"]
            return child
        else:
            return None
    except Exception as e:
        print(f"Error in evolve_heuristics_1: {e}")
        return None


def evolve_heuristics_2(parent1, parent2, input_plots_info, heur_model, fix_model):
    """
    Asks heur_model to generate a new solution exploring common ideas plus some variation.
    """
    session = heur_model.start_chat(history=[])

    prompt = f"""We have two existing solutions:

Parent 1:
{parent1["geojson"]}

Parent 2:
{parent2["geojson"]}

Identify common patterns in interventions, then propose a new solution building on them. Provide final GeoJSON:

```geojson
{{ your new FeatureCollection }}
```
"""

    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        if not completion.parts:
            print("Evolve_2 LLM response empty/blocked.")
            return None

        response_text = completion.parts[0].text.strip()
        evo_geojson = extract_geojson_content(response_text)
        if not evo_geojson:
            return None

        validated = attempt_validate_and_fix(evo_geojson, input_plots_info, fix_model)
        if validated:
            dominant_parent = pick_parent_for_multi(parent1, parent2)
            child = copy_parent_candidate(dominant_parent)
            child["geojson"] = validated
            child["pending_op"] = "evolve_2"
            child["pending_parent_score"] = dominant_parent["score"]
            return child
        else:
            return None
    except Exception as e:
        print(f"Error in evolve_heuristics_2: {e}")
        return None


def reflect(top_candidates, input_plots_info, heur_model, fix_model):
    """
    Summarizes top candidates, asks heur_model to reason about their strengths/weaknesses,
    and produce a new solution. Returns the new candidate or None.
    """
    best_candidate = max(top_candidates, key=lambda c: c["score"])
    session = heur_model.start_chat(history=[])

    heuristics_info = ""
    for i, cand in enumerate(top_candidates, start=1):
        snippet = cand['geojson']
        if len(snippet) > 1000:
            snippet = snippet[:1000] + "... (truncated)"
        heuristics_info += f"Candidate {i} (Fitness {cand['score']:.4f}):\n{snippet}\n\n"

    prompt = f"""Below are {len(top_candidates)} top solutions and their fitness:

{heuristics_info}

Reflect on their strengths/weaknesses, then propose a potentially improved solution.
Show your reasoning, then give the final GeoJSON:

```geojson
{{ your improved FeatureCollection }}
```
"""

    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        if not completion.parts:
            print("Reflect LLM response empty/blocked.")
            return None

        response_text = completion.parts[0].text.strip()
        reflect_geojson = extract_geojson_content(response_text)
        if not reflect_geojson:
            return None

        validated = attempt_validate_and_fix(reflect_geojson, input_plots_info, fix_model)
        if validated:
            child = copy_parent_candidate(best_candidate)
            child["geojson"] = validated
            child["pending_op"] = "reflect"
            child["pending_parent_score"] = best_candidate["score"]
            return child
        else:
            return None

    except Exception as e:
        print(f"Error in reflect: {e}")
        return None


def run_reflect(population, scores, input_plots_info, heur_model, fix_model, ground_truth):
    """
    Gathers top_k from population, also random subsets, calls reflect() to produce new children,
    then evaluates them.

    Returns (list_of_scores, list_of_candidates_with_scores).
    """
    pop_with_scores = []
    for i, p in enumerate(population):
        p['score'] = scores[i]
        pop_with_scores.append(p)

    if not pop_with_scores:
        return [], []

    sorted_pop = sorted(pop_with_scores, key=lambda x: x["score"], reverse=True)

    reflect_results = []

    # 1) Reflect on the top 5
    top_k = min(5, len(sorted_pop))
    if top_k > 0:
        top_candidates = sorted_pop[:top_k]
        child1 = reflect(top_candidates, input_plots_info, heur_model, fix_model)
        if child1:
            reflect_results.append(child1)

    # 2) Random reflects (4 times)
    num_random_reflects = 4
    for _ in range(num_random_reflects):
        sample_size = min(5, len(sorted_pop))
        random_candidates = random.sample(sorted_pop, sample_size)
        childX = reflect(random_candidates, input_plots_info, heur_model, fix_model)
        if childX:
            reflect_results.append(childX)

    # Evaluate reflect results
    reflect_results = evaluate_population(reflect_results, ground_truth, input_plots_info)
    reflect_scores = [cand["score"] for cand in reflect_results]
    return reflect_scores, reflect_results


def select_population(population, population_size):
    """
    Sorts by descending score, returns top 'population_size' individuals.
    """
    for cand in population:
        if "score" not in cand:
            cand["score"] = 0.0
    sorted_pop = sorted(population, key=lambda c: c["score"], reverse=True)
    return sorted_pop[:population_size]


def evaluate_population(population, ground_truth_data, input_plots_info):
    """
    Evaluates each candidate's fitness. Adds/updates 'score' in each candidate.
    Also updates trajectory with pending_op, fitness deltas, etc.
    """
    new_population = []
    for cand in population:
        score, _ = evaluate_heuristics(cand["geojson"], ground_truth_data, input_plots_info)
        if not math.isnan(score) and score >= 0:
            cand["score"] = score
            # Track operator usage/delta
            op = cand.pop("pending_op", None)
            parent_score = cand.pop("pending_parent_score", None)
            if op is not None and parent_score is not None:
                delta = score - parent_score
                if op not in cand["counts"]:
                    cand["counts"][op] = 0
                cand["counts"][op] += 1
                if op not in cand["fitness_deltas"]:
                    cand["fitness_deltas"][op] = 0.0
                cand["fitness_deltas"][op] += delta
                cand["trajectory"].append(f"{op}({delta:+.4f})")
            new_population.append(cand)
    return new_population


def run_gemini_initial(prompt, init_model):
    """
    Calls init_model with a prompt that requests chain-of-thought + final GeoJSON.
    Returns the raw text (which we then parse).
    """
    session = init_model.start_chat(history=[])
    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        if completion.parts:
            return completion.parts[0].text.strip()
        else:
            return None
    except Exception as e:
        print(f"Error in run_gemini_initial: {e}")
        return None


def create_init_population(init_model, fix_model, input_plots_info):
    """
    Reads the prompt from 'prompt_input.txt', uses init_model to generate
    population_size initial solutions, attempts to fix/validate each, and saves them to disk.
    """
    prompt_path = os.path.join(farm_dir, "prompt_input.txt")
    if not os.path.exists(prompt_path):
        print(f"Error: {prompt_path} not found.")
        return

    with open(prompt_path, "r") as f:
        prompt = f.read()

    generated_count = 0
    while generated_count < population_size:
        idx = generated_count + 1
        print(f"Generating initial candidate {idx}/{population_size}...")
        candidate_file = os.path.join(heur_dir, f"candidate_init_{idx}.geojson")

        raw_response = run_gemini_initial(prompt, init_model)
        if not raw_response:
            print(f"No response for candidate {idx}.")
            continue

        # Extract final GeoJSON
        extracted = extract_geojson_content(raw_response)
        if not extracted:
            print(f"Failed extracting GeoJSON for candidate {idx}.")
            continue

        validated = attempt_validate_and_fix(extracted, input_plots_info, fix_model, max_tries=3)
        if validated:
            with open(candidate_file, 'w') as cf:
                cf.write(validated)
            generated_count += 1
            print(f"Successfully generated candidate {idx}.")
        else:
            print(f"Failed to validate/fix candidate {idx}.")

    print(f"Finished generating {generated_count} initial candidates.")


def get_init_population():
    """
    Loads the validated initial solutions from disk into a list of candidate dicts.
    """
    init_population = []
    operator_list = ["init", "mutate", "crossover", "evolve_1", "evolve_2", "reflect"]
    loaded_count = 0
    for i in range(1, population_size + 1):
        candidate_file = os.path.join(heur_dir, f"candidate_init_{i}.geojson")
        if os.path.exists(candidate_file):
            with open(candidate_file, 'r') as f:
                content = f.read()
                if content.strip():
                    candidate = {
                        "geojson": content,
                        "trajectory": ["init(0.0000)"],
                        "counts": {op: 0 for op in operator_list},
                        "fitness_deltas": {op: 0.0 for op in operator_list},
                        "score": 0.0
                    }
                    candidate["counts"]["init"] = 1
                    init_population.append(candidate)
                    loaded_count += 1
    print(f"Loaded {loaded_count} initial candidates.")
    return init_population


def save_population_to_csv(population, generation, gen_dir):
    """
    Saves the population of a given generation to CSV, including the geojson column.
    """
    rows = []
    for i, cand in enumerate(population):
        row = {
            "generation": generation,
            "candidate_id": i,
            "score": cand.get("score", 0.0),
            "trajectory": json.dumps(cand.get("trajectory", [])),
            "counts": json.dumps(cand.get("counts", {})),
            "fitness_deltas": json.dumps(cand.get("fitness_deltas", {})),
            "geojson": cand.get("geojson", "")
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(gen_dir, f"population_gen_{generation}.csv")
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error saving population CSV: {e}")


def save_best_history_to_csv(best_history, gen_dir):
    """
    Saves the best candidate history to CSV.
    """
    rows = []
    for entry in best_history:
        row = {
            "generation": entry.get("generation", -1),
            "best_score": entry.get("score", 0.0),
            "trajectory": json.dumps(entry.get("trajectory", [])),
            "counts": json.dumps(entry.get("counts", {})),
            "fitness_deltas": json.dumps(entry.get("fitness_deltas", {}))
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(gen_dir, "best_history.csv")
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error saving best_history CSV: {e}")


def plot_best_trajectory_across_generations(best_history, gen_dir):
    """
    Plots the best fitness across generations.
    """
    generations = [bh["generation"] for bh in best_history]
    scores = [bh["score"] for bh in best_history]

    plt.figure()
    plt.plot(generations, scores, marker='o')
    plt.title("Best Score vs. Generation")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plot_path = os.path.join(gen_dir, "best_score_plot.png")
    try:
        plt.savefig(plot_path)
    except Exception as e:
        print(f"Error saving best_score_plot.png: {e}")
    plt.close()


def plot_population_operator_stats(population, generation, gen_dir):
    """
    Plots the usage counts of operators in the population.
    """
    op_counts = {}
    for cand in population:
        for op, count_val in cand["counts"].items():
            op_counts[op] = op_counts.get(op, 0) + count_val

    ops = list(op_counts.keys())
    vals = [op_counts[op] for op in ops]

    plt.figure()
    plt.bar(ops, vals)
    plt.title(f"Operator Usage - Gen {generation}")
    plt.xlabel("Operator")
    plt.ylabel("Usage Count")
    plot_path = os.path.join(gen_dir, f"op_usage_gen_{generation}.png")
    try:
        plt.savefig(plot_path)
    except Exception as e:
        print(f"Error saving op_usage_gen_{generation}.png: {e}")
    plt.close()


def run_evo_strat(population, input_plots_info, ground_truth_data):
    """
    Main evolutionary loop: Evaluate -> (Mutate|Crossover|Evolve|Reflect) -> Evaluate -> ...
    Saves best candidate each generation, plus CSV logs.
    """
    global ground_truth  # If needed, or pass ground_truth_data around
    ground_truth = ground_truth_data

    # Evaluate initial
    population = evaluate_population(population, ground_truth_data, input_plots_info)
    if not population:
        print("No valid initial population.")
        return

    scores = [cand["score"] for cand in population]
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_geojson = population[best_idx]["geojson"]

    line = f"Generation - 0 : Best Score - {best_score:.6f}\n"
    print(line)
    with open(score_file, 'a+') as f:
        f.write(line)

    best_candidate_file = os.path.join(gen_dir, f"best_candidate_gen_0.geojson")
    try:
        with open(best_candidate_file, 'w') as bf:
            bf.write(best_geojson)
    except Exception as e:
        print(f"Error saving best_candidate_gen_0: {e}")

    best_history = []
    best_history.append({
        "generation": 0,
        "score": best_score,
        "trajectory": population[best_idx].get("trajectory", [])[:],
        "counts": population[best_idx].get("counts", {}).copy(),
        "fitness_deltas": population[best_idx].get("fitness_deltas", {}).copy()
    })

    try:
        plot_best_trajectory_across_generations(best_history, gen_dir)
    except Exception as e:
        print(f"Plot error (initial): {e}")

    save_population_to_csv(population, 0, gen_dir)
    save_best_history_to_csv(best_history, gen_dir)

    population = select_population(population, population_size)

    for gen in range(num_generations):
        print(f"\n--- Generation {gen + 1} ---")

        best_candidate_file = os.path.join(gen_dir, f"best_candidate_gen_{gen + 1}.geojson")
        next_generation = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            # Try a variety of operators to produce ~population_size new children
            for _ in range(population_size):
                if len(population) >= 2:
                    p1, p2 = random.sample(population, 2)
                    futures.append(
                        executor.submit(crossover_heuristics, p1, p2, input_plots_info, heur_model, fix_model))
                if len(population) >= 2:
                    p1, p2 = random.sample(population, 2)
                    futures.append(
                        executor.submit(evolve_heuristics_1, p1, p2, input_plots_info, heur_model, fix_model))
                if len(population) >= 2:
                    p1, p2 = random.sample(population, 2)
                    futures.append(
                        executor.submit(evolve_heuristics_2, p1, p2, input_plots_info, heur_model, fix_model))
                if population:
                    indiv = random.choice(population)
                    futures.append(executor.submit(mutate_heuristics, indiv, input_plots_info, heur_model, fix_model))

            for fut in futures:
                try:
                    child = fut.result()
                    if child:
                        next_generation.append(child)
                except Exception as e:
                    print(f"Future result error: {e}")

        print(f"New children generated: {len(next_generation)}")

        next_generation = evaluate_population(next_generation, ground_truth_data, input_plots_info)
        print(f"New children evaluated: {len(next_generation)}")

        combined_pop = population + next_generation
        print(f"Combined population: {len(combined_pop)} before reflection.")

        # Reflection step
        if combined_pop:
            combined_scores = [c.get("score", 0.0) for c in combined_pop]
            r_scores, r_cands = run_reflect(combined_pop, combined_scores, input_plots_info, heur_model, fix_model,
                                            ground_truth)
            print(f"Reflection produced: {len(r_cands)} new candidates.")
            combined_pop += r_cands
            print(f"Combined population: {len(combined_pop)} after reflection.")
        else:
            print("Skipping reflection; no population.")

        combined_pop = select_population(combined_pop, population_size)
        print(f"Population after selection: {len(combined_pop)}")

        if not combined_pop:
            print(f"Population extinct at generation {gen + 1}. Stopping.")
            break

        # Identify best
        scores_cp = [cand["score"] for cand in combined_pop]
        best_index_g = np.argmax(scores_cp)
        best_score_g = scores_cp[best_index_g]
        best_cand_g = combined_pop[best_index_g]

        line = f"Generation - {gen + 1} : Best Score - {best_score_g:.6f}\n"
        print(line)
        with open(score_file, 'a+') as f:
            f.write(line)
        try:
            with open(best_candidate_file, 'w') as bf:
                bf.write(best_cand_g["geojson"])
        except Exception as e:
            print(f"Error saving best_candidate_gen_{gen + 1}: {e}")

        # Append to best_history
        best_history.append({
            "generation": gen + 1,
            "score": best_score_g,
            "trajectory": best_cand_g.get("trajectory", [])[:],
            "counts": best_cand_g.get("counts", {}).copy(),
            "fitness_deltas": best_cand_g.get("fitness_deltas", {}).copy()
        })

        # Plot stats
        try:
            plot_population_operator_stats(combined_pop, gen + 1, gen_dir)
        except Exception as e:
            print(f"Plot error (operator stats): {e}")

        population = combined_pop
        save_population_to_csv(combined_pop, gen + 1, gen_dir)
        save_best_history_to_csv(best_history, gen_dir)
        try:
            plot_best_trajectory_across_generations(best_history, gen_dir)
        except Exception as e:
            print(f"Plot error (best trajectory): {e}")

    print("Evolution complete.")


# --- Main Execution ---

if __name__ == "__main__":
    cfg = Config()
    capture = CommandOutputCapture()

    population_size = 10
    num_generations = 5
    init = True

    # Provide cost/price info for the heur_model prompts
    params_instructions = (
        "Crop Prices (USD/Tonne): {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
        "'Barley': 120, 'Spring wheat': 200}.\n"
        "Costs: {'margin': {'implementation': 400 USD/ha, 'maintenance': 60 USD/ha/year}, "
        "'habitat': {'implementation': 300 USD/ha, 'maintenance': 70 USD/ha/year}, "
        "'agriculture': {'maintenance': 100 USD/ha/year}}.\n"
    )

    syn_farm_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms")
    all_farms_geojson_path = os.path.join(syn_farm_dir, "farms_cp.geojson")

    farm_ids = np.arange(10, 11)
    for farm_id in farm_ids:
        print(f"\n===== Processing Farm {farm_id} =====")
        farm_dir = os.path.join(syn_farm_dir, f"farm_{farm_id}")
        heur_dir = os.path.join(farm_dir, "heuristics_geojson")
        if not os.path.exists(heur_dir):
            os.makedirs(heur_dir)
        gen_dir = os.path.join(heur_dir, "generations")
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

        # Copy input.geojson to heur_dir
        input_geojson_src = os.path.join(farm_dir, "input.geojson")
        input_geojson_dst = os.path.join(heur_dir, "input.geojson")
        if os.path.exists(input_geojson_src):
            try:
                shutil.copyfile(input_geojson_src, input_geojson_dst)
            except Exception as e:
                print(f"Error copying input.geojson: {e}")
                continue
        else:
            print(f"input.geojson not found for farm {farm_id}.")
            continue

        # Load input to gather IDs
        try:
            with open(input_geojson_dst) as f:
                input_geojson_data = json.load(f)
            input_plots_info = {}
            for feat in input_geojson_data.get("features", []):
                props = feat.get("properties", {})
                ftype = props.get("type")
                fid = props.get("id")
                if ftype == "ag_plot" and fid is not None:
                    input_plots_info[fid] = props
        except Exception as e:
            print(f"Error loading input.geojson for farm {farm_id}: {e}")
            continue

        # Score file for logging
        score_file = os.path.join(heur_dir, "scores_es.txt")

        # Load ground_truth
        ground_truth_path = os.path.join(farm_dir, "output_gt.geojson")
        if not os.path.exists(ground_truth_path):
            print(f"No ground truth file for farm {farm_id}.")
            continue
        try:
            with open(ground_truth_path) as gt_f:
                ground_truth = json.load(gt_f)
        except Exception as e:
            print(f"Error loading ground truth for farm {farm_id}: {e}")
            continue

        # Initialize models
        print("Initializing models...")
        heur_model, fix_model = init_gemini_model(
            heur_model_name="gemini-1.5-flash-latest",
            fix_model_name="gemini-1.5-flash-latest"
        )
        print("Models ready.")

        if init:
            print("Creating initial population...")
            try:
                create_farm_prompt_file_2(farm_id, all_farms_geojson_path, farm_dir)
            except Exception as e:
                print(f"Error creating prompt file: {e}")
            create_init_population(heur_model, fix_model, input_plots_info)

        print("Loading initial population...")
        initial_population = get_init_population()
        if not initial_population:
            print(f"No initial candidates for farm {farm_id}. Skipping.")
            continue

        print("Starting evolutionary strategy...")
        run_evo_strat(initial_population, input_plots_info, ground_truth)
        print(f"===== Finished Farm {farm_id} =====")

    print("All farms processed.")

