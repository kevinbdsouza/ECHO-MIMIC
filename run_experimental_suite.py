import google.generativeai as genai
from echo_mimic.rate_limiter import send_message_with_retry
import random
import os
import shutil
from pathlib import Path
import json
import numpy as np
import re
import math
import pandas as pd
from echo_mimic.metrics import compute_radon_metrics
from echo_mimic.common import (
    CommandOutputCapture,
    build_model,
    configure_genai,
    ensure_rate_limiter,
    extract_message,
    extract_python_code,
    fix_with_model,
    validate_python_code,
)
from echo_mimic.config import Config

capture = CommandOutputCapture()
cfg = Config()  # Initialize your config
rate_limiter = ensure_rate_limiter(cfg)

PV_FACTOR_20Y_5PC = 12.4622  # Present Value factor for an annuity of 20 years at 5%
BUDGET_PER_FARM = 10000.0

# --- Farmer Personalities ---
FARMER_PERSONALITIES = {
    "resistant": (
        "You are a farmer who is extremely resistant to changing your current practices. "
        "You are highly skeptical of new advice and require overwhelming evidence or extremely compelling reasons to alter your ways. "
        "You prioritize autonomy and established routines above all else. "
        "You will likely ignore most messages and nudges unless they address a critical, undeniable issue you are personally facing, "
        "or offer an exceptionally large and straightforward benefit with minimal effort or risk on your part."
    ),
    "economic": (
        "You are a pragmatic farmer focused primarily on the economic outcomes of your farm. "
        "You are open to changing your heuristics if the proposed changes have a clear, quantifiable positive impact "
        "on your profitability, efficiency, or reduce your financial risks. "
        "You respond well to financial incentives, cost-benefit analyses, and evidence of improved yields or market access. "
        "Social or purely environmental arguments are secondary to economic viability."
    ),
    "socially_influenced": (
        "You are a farmer who is significantly influenced by the practices and opinions of your peers and the broader farming community. "
        "You are concerned about your reputation and how your farm is perceived. "
        "You are more likely to adopt new practices if you see others in your network successfully implementing them "
        "or if there are strong social norms or community expectations favoring such changes. "
        "While economics matter, social validation and community standing are strong motivators for you."
    )
}

# --- Base System Instructions (from original) ---
POLICY_SYSTEM_INSTRUCTIONS_BASE = (
    "You are an expert in land use policy, communication, incentives, and economics. "
    "Your task is to come up with the best message to be communicated to the farmers "
    "so as to change their behaviour from one set of heuristics to another set of heuristics. "
    "Modify your approach based on whether you are asked to use **behavioral economics levers** or **economic incentives**. \n\n"
)

FARM_SYSTEM_INSTRUCTIONS_BASE_CORE = (  # Core part of farmer's role
    "Your task is to respond to communication from policy professionals by altering "
    "your heuristics in an appropriate way. You will return final answers in python code. "
    "You should keep all data loading from input.geojson and dumping to output.geojson "
    "the exact same, and just alter the logic depending on your context and personality, "
    "and what the message is. Don't invent new variable names in the output, keep "
    "using margin_intervention and habitat_conversion. "
    "If you do decide to follow the message and implement suggestions in the message, then:"
    "a) If the message proposes changes to some key parameters that you "
    "are using in your heuristics, you should change them according to the proposed changes. "
    "b) If the message introduces new parameters or ideas, you should incorporate them in your heuristics. "
    "c) Your should react to the message from the policy professional, resulting in altered margin_intervention and "
    "habitat_conversion values for your plots, in an appropriate way that aligns "
    "with your personality, interests and the message you received. "
    "d) Your final answer should be your full altered python code heuristics, keeping overall framework of the original python code the same. "
    "If you decide not to follow the message or implement suggestions in the message, "
    "then you should just return the full original code and not make any changes.\n\n"
)

FIX_MODEL_SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant who is an expert in graph and spatial optimization methods and python. "
    "Given the python code and the stack traceback, fix the errors and return the correct functioning python code."
)


def get_modified_farm_system_instructions(personality_key):
    if personality_key not in FARMER_PERSONALITIES:
        raise ValueError(f"Unknown personality: {personality_key}")
    # The personality description comes first, then the base instructions.
    return FARM_SYSTEM_INSTRUCTIONS_BASE_CORE + "\n\n" + FARMER_PERSONALITIES[personality_key]


def get_nudge_specific_policy_instructions(nudge_type_key, params_instructions_str, heur_eco_intens_str,
                                           heur_eco_conn_str, social_comparison_data_str):
    common_policy_intro = (
        "Your task is to come up with a message to the farmers in order to change their behaviour from "
        "following the ecological intensification heuristics that benefits solely their farm to following the "
        "ecological connectivity heuristics that increase landscape connectivity.\n"
        f"The ecological intensification heuristics the farmer is currently following are:\n "
        f"```python\n{heur_eco_intens_str}\n``` \nThe ecological connectivity heuristics that you should nudge them towards are:\n "
        f"```python\n{heur_eco_conn_str}\n``` \n\n"
        f"The current parameters like prices and costs are: {params_instructions_str}\n\n"
        "One caveat is that the ecological connectivity heuristics are given in directions (margin_directions and "
        "habitat_directions), where there are 4 possible directions north-west, north-east, south-west, south-east. "
        "That means the resulting margin_intervention and habitat_conversion values can only be in multiples of 0.25 "
        "(i.e., 0.0, 0.25, 0.5, 0.75, 1.0). You need to ensure your nudge encourages outcomes consistent with this. "
        "Your goal should be to communicate a message that gets the farmer to alter the margin_intervention and "
        "habitat_conversion values for each of the plots, from the former to the latter. "
        "Provide your final message to the farmer in this format \communication{message}. Explain your reasoning step-by-step BEFORE providing the final message block.\n\n"
    )

    if nudge_type_key == "behavioral":
        if not social_comparison_data_str:
            social_comparison_data_str = "No specific farm and neighbor data provided. Use general examples."
        behavioral_instructions = (
            "For this task, employ **behavioral economics levers** by analyzing the provided farm and neighbor data. Your message should subtly guide the farmer. "
            "Base your strategies on the following data:\n"
            "--- BEGIN FARM AND NEIGHBOR DATA ---\n"
            f"{social_comparison_data_str}\n"
            "--- END FARM AND NEIGHBOR DATA ---\n\n"
            "Using this data, customize your approach for each lever:\n"
            "1.  **Social Comparisons**: Analyze the data to find relevant comparisons between the target farm and its neighbors. "
            "Highlight similarities in farm characteristics (e.g., size, crops, yield if similar) and then point to neighbors' "
            "adoption of connectivity measures.\n"
            "2.  **Defaults**: Based on prevalent practices among neighbors (if any are clear from the data) or common successful starting "
            "points observed, suggest a 'default' or 'recommended first step' that seems most suitable for the target farm's context "
            "(e.g., 'Given your farm's layout and what's worked for others nearby with similar watercourses, a common starting point is "
            "establishing margins along...'). Make this default easy to adopt. If many neighbors are doing something, you can frame it "
            "as 'Many farmers in your community are finding that X is a practical way to start and have collectively decided that this should be a "
            "default practice for everyone.'.\n"
            "3.  **Commitments**: Look at the target farm's current state and neighbors' outputs. Propose a small, manageable, voluntary commitment "
            "that aligns with the farm's characteristics (e.g., 'Seeing that your neighbor Farm Z started with a small trial patch, "
            "would you be open to a similar low-commitment trial on one of your less productive field edges this season?').\n"
            "4.  **Framing**: Use the provided data to frame the benefits of ecological connectivity in a way that is "
            "concrete and relatable to the target farm. Acknowledge potential concerns and frame solutions or "
            "outcomes in light of those. Emphasize tangible "
            "gains and opportunities relevant to the farm's context as suggested by the data.\n\n"
            "Your message should weave these data-driven, customized elements into a persuasive narrative. "
            "You are primarily shaping the choice context and message, not offering new major financial packages beyond existing "
            "general support programs (which can be alluded to if they fit the framing)."
        )
        return common_policy_intro + behavioral_instructions
    elif nudge_type_key == "economic":
        economic_incentives_table_info = (
            "You can use a mix of the following policy instruments:\n"
            "- Subsidy factor for margin establishment adjacent to existing habitat (on farmer's cost of $400/ha, one-time).\n"
            "- Subsidy factor for habitat establishment adjacent to existing habitat (on farmer's cost of $300/ha, one-time).\n"
            "- Subsidy factor for margin maintenance (on farmer's cost of $60/ha/year, for 20 years).\n"
            "- Subsidy factor for habitat maintenance (on farmer's cost of $70/ha/year, for 20 years).\n"
            "- Payment per hectare for habitat conversion (direct one-time payment, range $[0, 150]$ currency units/ha).\n"
            "The following are constraints or market factors, not direct subsidy levers from your budget:\n"
            "- Mandated minimum total habitat area per farm (range $[0, 10]$ ha).\n"
            "- Mandated minimum fraction of margin adjacent to existing habitats (range $[0, 0.3]$ ha).\n"
            "- Eco-premium factor for crop (increase selected individual crop prices by a factor in the range [1, 1.3]).\n"
        )
        return common_policy_intro + (
            f"For this task, design a policy message that primarily uses **economic incentives**. You have a total notional budget of **${BUDGET_PER_FARM:,.0f} per farm** (Present Value over a 20-year horizon with a 5% annual discount rate; the PV factor for 20 years of annual payments is {PV_FACTOR_20Y_5PC:.4f}).\n"
            f"{economic_incentives_table_info}\n"
            "When proposing your policy, explicitly state the subsidy rates (as percentages or factors, e.g., 'a 0.4 subsidy factor means 40% subsidy') or payment amounts you are offering. These rates must be within their specified ranges.\n"
            "Your proposed incentive *rates* should be chosen considering the total budget. For example, the PV cost to the budget per hectare for your offered incentives would be:\n"
            "  - Margin Establishment \\times 400$\n"
            "  - Habitat Establishment \\times 300 + P_{ha}$ (if payment is for new habitat)\n"
            "  - Margin Maintenance (PV over 20 yrs) \\times 60 \\times {PV_FACTOR_20Y_5PC:.4f}$\n"
            "  - Habitat Maintenance (PV over 20 yrs) \\times 70 \\times {PV_FACTOR_20Y_5PC:.4f}$\n\n"
            "Design an attractive package of *rates*. Assume a farmer might adopt measures on a few hectares (e.g., 1-5 ha of habitat, 1-3 ha of margins). Your offered rates should aim to keep the potential total PV cost for such a scenario within the ${BUDGET_PER_FARM:,.0f} limit. You don't need to calculate the farmer's exact uptake, but the *offer itself* must be responsibly designed with this budget constraint in mind."
        )
    else:
        raise ValueError(f"Unknown nudge type: {nudge_type_key}")


def init_experimental_gemini_model(personality_key,
                                   policy_model_name="gemini-2.0-flash-thinking-exp-01-21",
                                   farm_model_name="gemini-2.0-flash-thinking-exp-01-21",
                                   fix_model_name="gemini-2.0-flash-thinking-exp-01-21"):
    configure_genai()

    policy_model = build_model(
        policy_model_name,
        POLICY_SYSTEM_INSTRUCTIONS_BASE,
        ensure_configured=False,
    )

    farm_system_instructions = get_modified_farm_system_instructions(personality_key)
    farm_model = build_model(
        farm_model_name,
        farm_system_instructions,
        ensure_configured=False,
    )

    fix_model = build_model(
        fix_model_name,
        FIX_MODEL_SYSTEM_INSTRUCTIONS,
        ensure_configured=False,
    )

    global rate_limiter
    rate_limiter = ensure_rate_limiter(cfg)

    return policy_model, farm_model, fix_model


def validate_response_exp(response_code, heur_dir, fix_model, input_json_path_src_for_sim):
    heur_dir_path = Path(heur_dir)
    input_source = Path(input_json_path_src_for_sim)

    def _prepare(workdir_path: Path) -> None:
        shutil.copyfile(input_source, workdir_path / "input.geojson")

    def _fixer(code: str, trace: str) -> str:
        return fix_with_model(
            fix_model,
            code,
            trace,
            rate_limiter=rate_limiter,
            include_code_fence_hint=True,
        )

    return validate_python_code(
        response_code,
        workdir=heur_dir_path,
        capture=capture,
        fixer=_fixer,
        script_name="temp_fix.py",
        max_attempts=2,
        copy_template=False,
        pre_run=_prepare,
    )


def evaluate_heuristics_exp(heuristic_policy_message, ground_truth_data, farm_model, fix_model,
                            current_farm_prompt_instructions_str, heur_dir_for_eval,
                            base_input_geojson_for_farm, input_json_data_for_eval):
    """
    Evaluates a heuristic policy message.
    heuristic_policy_message: The message string from the policy model.
    ground_truth_data: Loaded JSON of the ground truth.
    farm_model: The specific farmer model (with personality).
    fix_model: The code fixing model.
    current_farm_prompt_instructions_str: The task instructions for the farmer model.
    heur_dir_for_eval: Directory to run simulation, should be specific to this eval run if concurrent.
    base_input_geojson_for_farm: Path to the clean input.geojson for this farm.
    input_json_data_for_eval: Loaded input.geojson data for error calculation.
    """
    session = farm_model.start_chat(history=[])
    # The farmer model gets the policy message AND its general task instructions
    prompt_to_farmer = current_farm_prompt_instructions_str + f"\n\nMessage from the policy professional is: {heuristic_policy_message}\n"

    try:
        completion = session.send_message(prompt_to_farmer)
        farm_response_text = completion.parts[0].text
        farmer_heuristics_code = extract_python_code(farm_response_text)

        if not farmer_heuristics_code or "No Python code found" in farmer_heuristics_code:  # or if farmer refuses
            print(f"Farmer model did not return valid Python code. Response: {farm_response_text}")
            return 0, heuristic_policy_message, ""  # Low score, original policy, no code

        # Validate and fix the farmer's code
        # Ensure heur_dir_for_eval is clean or unique for this evaluation
        if not os.path.exists(heur_dir_for_eval):
            os.makedirs(heur_dir_for_eval)

        # Each validation needs its own copy of input.geojson
        temp_eval_script_path = os.path.join(heur_dir_for_eval, "temp_eval_script.py")

        # The validation needs the input.geojson to be in the script's CWD.
        # The validate_response_exp function handles copying the geojson.
        validated_farmer_code = validate_response_exp(farmer_heuristics_code, heur_dir_for_eval, fix_model,
                                                      base_input_geojson_for_farm)

        if validated_farmer_code is None:
            print("Farmer code could not be validated after fixes.")
            return 0, heuristic_policy_message, farmer_heuristics_code  # Low score, original policy, attempt code

        # Write final validated code to a uniquely named file for this evaluation run
        with open(temp_eval_script_path, 'w') as f:
            f.write(validated_farmer_code)

    except Exception as e:
        print(f"Error during farm model interaction or initial validation: {e}")
        return 0, heuristic_policy_message, ""  # Low score

    # --- Run the validated farmer's code and calculate error ---
    try:
        original_cwd = os.getcwd()
        os.chdir(heur_dir_for_eval)  # Change to script's directory
        # The input.geojson should have been copied by validate_response_exp

        code, _, err = capture.run_python_script(os.path.basename(temp_eval_script_path))
        os.chdir(original_cwd)  # Change back

        if code != 0:
            print(f"Validated farmer code failed to run. Error: {err}")
            return 0, heuristic_policy_message, validated_farmer_code

        output_geojson_path = os.path.join(heur_dir_for_eval, 'output.geojson')
        if not os.path.exists(output_geojson_path):
            print("output.geojson not found after running farmer script.")
            return 0, heuristic_policy_message, validated_farmer_code

        with open(output_geojson_path) as f:
            predicted_output = json.load(f)

        margin_errors = []
        habitat_errors = []
        pred_features = predicted_output.get("features", [])
        input_json_features = input_json_data_for_eval.get("features", [])

        for inp_feature in input_json_features:
            if inp_feature.get("properties", {}).get("type") == "hab_plots":  # Check existence of keys
                continue

            plot_id = inp_feature.get("properties", {}).get("id")
            if not plot_id: continue

            gt_margin, gt_hab = 0, 0
            for feat in ground_truth_data:  # ground_truth_data is already loaded list of features
                if plot_id == feat.get("id"):
                    gt_margin = len(set(feat.get("margin_directions", []))) / 4.0
                    gt_hab = len(set(feat.get("habitat_directions", []))) / 4.0
                    break

            pred_found = False
            for pred_feat in pred_features:
                if pred_feat.get("properties", {}).get("id") == plot_id:
                    pred_margin = pred_feat.get("properties", {}).get("margin_intervention", 0)
                    pred_hab = pred_feat.get("properties", {}).get("habitat_conversion", 0)
                    margin_errors.append(abs(gt_margin - pred_margin))
                    habitat_errors.append(abs(gt_hab - pred_hab))
                    pred_found = True
                    break

            if not pred_found:
                margin_errors.append(10)  # Penalize heavily if a plot is missed
                habitat_errors.append(10)

        if not margin_errors and not habitat_errors:  # No plots to evaluate?
            total_error = 20  # High error if no valid plots found for comparison
        else:
            total_margin_error = np.mean(margin_errors) if margin_errors else 0
            total_habitat_error = np.mean(habitat_errors) if habitat_errors else 0
            total_error = total_margin_error + total_habitat_error + 0.01  # Add small constant to avoid zero division

        fitness = 1.0 / total_error if total_error > 0 else 100.0  # High fitness if error is near zero
        return fitness, heuristic_policy_message, validated_farmer_code

    except Exception as e:
        print(f"Error during heuristic evaluation (scoring phase): {e}")
        # Ensure we change back directory if an error occurs mid-try
        if os.getcwd() != original_cwd:
            os.chdir(original_cwd)
        return 0, heuristic_policy_message, validated_farmer_code if 'validated_farmer_code' in locals() else ""


# --- Modified Evolution Operators ---
def mutate_heuristics_exp(heuristic_policy, policy_model, current_nudge_policy_instructions_str):
    session = policy_model.start_chat(history=[])
    prompt = (f"""The following message is communicated to the farmer:
        {heuristic_policy}\n\n""" +
              """Suggest a subtle mutation to this message to better nudge the farmer towards ecological connectivity.
              Focus on small, logical changes in how the message is communicated or in the parameters of the incentives/levers being used,
              consistent with the overall policy strategy outlined below.
              The new message must remain a valid communication to the farmer.
              Explain your reasoning and think step by step.
              Provide your final mutated message in the format \communication{message}.\n\n
              Current Policy Strategy Context:\n""" + current_nudge_policy_instructions_str
              )
    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        response = completion.parts[0].text
        return extract_message(response)
    except Exception as e:
        print(f"Error in mutate_heuristics_exp: {e}")
        return heuristic_policy  # Return original if mutation fails


# Other operators (crossover, evolve1, evolve2, reflect) would be similarly adapted:
# - Accept policy_model and current_nudge_policy_instructions_str
# - Append current_nudge_policy_instructions_str to their specific prompts for context
# For brevity, I'll sketch one more:

def crossover_heuristics_exp(parent1_message, parent2_message, policy_model, current_nudge_policy_instructions_str):
    session = policy_model.start_chat(history=[])
    prompt = (f"""Given two messages communicated to the farmer (Parent 1 and Parent 2):
    Parent 1: {parent1_message}
    Parent 2: {parent2_message}""" +
              """Combine these two messages optimally. The combination should synthesize the strengths of both,
              resolve any contradictions, and result in a single, coherent, and valid message to the farmer.
              Explain your reasoning step by step.
              Provide your final combined message in the format \communication{message}.\n\n
              Current Policy Strategy Context:\n""" + current_nudge_policy_instructions_str
              )
    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        response = completion.parts[0].text
        return extract_message(response)
    except Exception as e:
        print(f"Error in crossover_heuristics_exp: {e}")
        return random.choice([parent1_message, parent2_message])  # Simple fallback


def evolve_heuristics_1_exp(parent1_message, parent2_message, policy_model,
                            current_nudge_policy_instructions_str):  # Explore new ideas
    session = policy_model.start_chat(history=[])
    prompt = (f"""Given two parent messages:
    Parent 1: {parent1_message}
    Parent 2: {parent2_message}""" +
              """"Generate a new message that is as different as possible from these parents, exploring novel approaches or parameters,
              while still aiming to nudge the farmer towards ecological connectivity and adhering to the overall policy strategy.
              Explain your reasoning step by step.
              Provide your final new message in the format \communication{message}.\n\n
              Current Policy Strategy Context:\n""" + current_nudge_policy_instructions_str
              )
    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        response = completion.parts[0].text
        return extract_message(response)
    except Exception as e:
        print(f"Error in evolve_heuristics_1_exp: {e}")
        return random.choice([parent1_message, parent2_message])


def evolve_heuristics_2_exp(parent1_message, parent2_message, policy_model,
                            current_nudge_policy_instructions_str):  # Share idea, new parts
    session = policy_model.start_chat(history=[])
    prompt = (f"""Given two parent messages:
    Parent 1: {parent1_message}
    Parent 2: {parent2_message}""" +
              """Identify common underlying ideas or effective components in these messages.
              Then, design a new message that builds on these common ideas but introduces new elements or framings
              to make it distinct and potentially more effective. Ensure it aligns with the policy strategy.
              Explain your reasoning step by step.
              Provide your final new message in the format \communication{message}.\n\n
              Current Policy Strategy Context:\n""" + current_nudge_policy_instructions_str
              )
    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        response = completion.parts[0].text
        return extract_message(response)
    except Exception as e:
        print(f"Error in evolve_heuristics_2_exp: {e}")
        return random.choice([parent1_message, parent2_message])


def reflect_exp(top_heuristics_msgs, top_scores, policy_model, current_nudge_policy_instructions_str):
    session = policy_model.start_chat(history=[])
    heuristics_info = ""
    for i, (msg, score) in enumerate(zip(top_heuristics_msgs, top_scores), start=1):
        heuristics_info += f"Message {i} (Fitness Score: {score:.4f}):\n{msg}\n\n"
    prompt = (f"""Based on the following top messages and their fitness scores:
    {heuristics_info}""" +
              """Analyze these messages. Identify common strengths, weaknesses, or patterns.
              Craft a new, improved message expected to achieve a higher fitness score.
              The new message must be a valid communication to the farmer and align with the overall policy strategy.
              Explain your reasoning step by step.
              Provide your final new message in the format \communication{message}.\n\n
              Current Policy Strategy Context:\n""" + current_nudge_policy_instructions_str
              )
    try:
        completion = send_message_with_retry(session, prompt, rate_limiter)
        response = completion.parts[0].text
        return extract_message(response)
    except Exception as e:
        print(f"Error in reflect_exp: {e}")
        return random.choice(top_heuristics_msgs) if top_heuristics_msgs else ""


# --- Population Management (can largely use original if adapted for _exp versions) ---
def evaluate_population_exp(population_messages, ground_truth_data, farm_model, fix_model,
                            current_farm_prompt_instructions_str, eval_run_dir,
                            base_input_geojson_for_farm, input_json_data_for_eval):
    scores = []
    new_population_messages = []  # Policies (messages)
    codes = []  # Farmer codes

    for i, individual_message in enumerate(population_messages):
        # Each eval needs a unique sub-directory to avoid file overwrites (input.geojson, temp_eval_script.py, output.geojson)

        score, policy_msg_after_eval, farmer_code = evaluate_heuristics_exp(
            individual_message, ground_truth_data, farm_model, fix_model,
            current_farm_prompt_instructions_str, eval_run_dir,
            base_input_geojson_for_farm, input_json_data_for_eval
        )
        if not (math.isnan(score)):
            scores.append(score)
            new_population_messages.append(policy_msg_after_eval)  # Use the message that was evaluated
            codes.append(farmer_code if farmer_code else "")
        else:  # Handle NaN scores if they occur
            scores.append(0)  # Assign a low score
            new_population_messages.append(individual_message)
            codes.append("")
            print(f"Warning: NaN score for individual. Message: {individual_message[:50]}...")

    return scores, new_population_messages, codes


def select_population(population_messages, scores, codes, population_size_target):
    population_with_scores = sorted(zip(population_messages, scores, codes), key=lambda item: item[1], reverse=True)

    selected_population_messages = []
    selected_scores = []
    selected_codes = []

    for ind_msg, score, code_str in population_with_scores[:population_size_target]:
        selected_population_messages.append(ind_msg)
        selected_scores.append(score)
        selected_codes.append(code_str)

    return selected_population_messages, selected_scores, selected_codes



def run_reflect_exp(population_messages, scores, policy_model, current_nudge_policy_instructions_str,
                    ground_truth_data, farm_model, fix_model, current_farm_prompt_instructions_str,
                    eval_run_dir_reflect, base_input_geojson_for_farm, input_json_data_for_eval):
    pop_with_scores = sorted(zip(population_messages, scores), key=lambda x: x[1], reverse=True)
    sorted_population_msgs = [p for p, s in pop_with_scores]
    sorted_scores = [s for p, s in pop_with_scores]

    reflect_results_msgs = []
    num_reflect_runs = 4  # As in original (1 top-5 + 3 random-ish) -> let's make it 4 total for simplicity

    if len(sorted_population_msgs) < 1: return [], [], []  # Not enough individuals
    if len(sorted_population_msgs) < 5:
        top_k_msgs = sorted_population_msgs
        top_k_scores = sorted_scores
    else:
        top_k_msgs = sorted_population_msgs[:5]
        top_k_scores = sorted_scores[:5]

    try:
        result = reflect_exp(top_k_msgs, top_k_scores, policy_model, current_nudge_policy_instructions_str)
        if result: reflect_results_msgs.append(result)
    except Exception as e:
        print(f"Reflect (top 5) failed: {e}")

    for _ in range(num_reflect_runs - 1):  # Additional runs with random samples
        if len(sorted_population_msgs) >= 5:
            sample_indices = random.sample(range(len(sorted_population_msgs)), 5)
        else:  # take all if less than 5
            sample_indices = list(range(len(sorted_population_msgs)))

        if not sample_indices: continue

        sel_heuristics_msgs = [sorted_population_msgs[i] for i in sample_indices]
        sel_scores = [sorted_scores[i] for i in sample_indices]
        try:
            result = reflect_exp(sel_heuristics_msgs, sel_scores, policy_model, current_nudge_policy_instructions_str)
            if result: reflect_results_msgs.append(result)
        except Exception as e:
            print(f"Reflect (random sample) failed: {e}")

    if not reflect_results_msgs: return [], [], []

    reflect_scores, reflect_results_msgs_eval, reflect_codes = evaluate_population_exp(
        reflect_results_msgs, ground_truth_data, farm_model, fix_model,
        current_farm_prompt_instructions_str, eval_run_dir_reflect,  # Pass specific dir for these evaluations
        base_input_geojson_for_farm, input_json_data_for_eval
    )
    return reflect_scores, reflect_results_msgs_eval, reflect_codes


# --- Experiment Orchestration ---
def create_initial_population_exp(policy_model, full_policy_generation_prompt_str, population_size,
                                  exp_heur_dir, start_idx=1):
    """Generates initial policy messages and saves them."""
    if not os.path.exists(exp_heur_dir):
        os.makedirs(exp_heur_dir)

    generated_messages = []
    for i in range(start_idx, start_idx + population_size):
        heur_file_path = os.path.join(exp_heur_dir, f"heuristics_gem_{i}.txt")
        try:
            session = policy_model.start_chat(history=[])
            completion = session.send_message(full_policy_generation_prompt_str)
            policy_message = extract_message(completion.parts[0].text)
            if policy_message:
                with open(heur_file_path, 'w') as f:
                    f.write(policy_message)
                generated_messages.append(policy_message)
            else:
                print(f"Failed to generate valid initial policy message {i}")
                # Add a dummy or skip? For now, skip, so population might be smaller.
        except Exception as e:
            print(f"Error creating initial policy {i}: {e}")
    return generated_messages  # Return messages, not just write to file


def get_initial_population_exp(population_size, exp_heur_dir):
    """Reads policy messages from files."""
    init_population_messages = []
    for i in range(1, population_size + 1):
        heuristic_file = os.path.join(exp_heur_dir, f"heuristics_gem_{i}.txt")
        if os.path.exists(heuristic_file):
            with open(heuristic_file, 'r') as f:
                heuristic_message = f.read()
            init_population_messages.append(heuristic_message)
        else:
            # This case should ideally be handled by create_initial_population_exp ensuring all files are made
            # Or, if create_initial_population_exp returns the list directly, this function might not be needed.
            print(f"Warning: Initial heuristic file not found: {heuristic_file}")
    return init_population_messages


def run_evo_strat_exp(
        experiment_label, initial_population_messages,
        policy_model, farm_model, fix_model,
        current_nudge_policy_instructions_str, current_farm_prompt_instructions_str,
        ground_truth_data, base_input_geojson_for_farm, input_json_data_for_eval,
        exp_output_base_dir, population_size_target=25, num_generations=10, inner_loop_size=10):
    print(f"\n--- Starting Evolutionary Strategy for: {experiment_label} ---")

    exp_gen_dir = os.path.join(exp_output_base_dir, "generations")  # For best policies per gen
    exp_eval_dir = os.path.join(exp_output_base_dir, "eval_runs")  # For temp files during evaluation
    exp_heur_dir = os.path.join(exp_output_base_dir, "heuristics")

    for d in [exp_gen_dir, exp_eval_dir]:
        if not os.path.exists(d): os.makedirs(d)

    score_file_path = os.path.join(exp_output_base_dir, "scores_es.txt")
    metrics_file_path = os.path.join(exp_output_base_dir, "metrics_es.csv")

    # Initial population evaluation
    current_scores, current_population_messages, current_codes = evaluate_population_exp(
        initial_population_messages, ground_truth_data, farm_model, fix_model,
        current_farm_prompt_instructions_str, exp_eval_dir,
        base_input_geojson_for_farm, input_json_data_for_eval
    )

    if not current_scores:  # Handle case of no valid individuals from initial pop
        print(f"No valid individuals after initial evaluation for {experiment_label}. Stopping.")
        return

    best_idx = np.argmax(current_scores)
    line = f"Experiment: {experiment_label} - Generation 0 : Best Score - {current_scores[best_idx]:.4f}\n"
    print(line)
    with open(score_file_path, 'a+') as f:
        f.write(line)

    best_policy_file = os.path.join(exp_gen_dir, f"best_policy_gen_0.txt")
    with open(best_policy_file, 'w') as f:
        f.write(current_population_messages[best_idx])
    best_python_file = os.path.join(exp_gen_dir, f"best_python_gen_0.py")
    if current_codes and current_codes[best_idx]:
        with open(best_python_file, 'w') as f: f.write(current_codes[best_idx])

    metrics_df = pd.DataFrame()  # Initialize empty
    metrics_df = compute_radon_metrics(metrics_df, current_codes, current_scores)
    metrics_df.to_csv(metrics_file_path, index=False)

    selected_population_msgs, selected_scores, selected_codes = select_population(
        current_population_messages, current_scores, current_codes, population_size_target
    )

    for generation in range(num_generations):
        print(f"\n{experiment_label} - Generation {generation + 1}")
        next_generation_msgs = []

        if not selected_population_msgs:
            print(f"No individuals selected for generation {generation + 1}. Stopping.")
            break

        for _ in range(inner_loop_size):
            # Crossover
            if len(selected_population_msgs) >= 2:
                p1, p2 = random.sample(selected_population_msgs, 2)
                child_co = crossover_heuristics_exp(p1, p2, policy_model, current_nudge_policy_instructions_str)
                if child_co: next_generation_msgs.append(child_co)

                child_e1 = evolve_heuristics_1_exp(p1, p2, policy_model, current_nudge_policy_instructions_str)
                if child_e1: next_generation_msgs.append(child_e1)
                child_e2 = evolve_heuristics_2_exp(p1, p2, policy_model, current_nudge_policy_instructions_str)
                if child_e2: next_generation_msgs.append(child_e2)

            # Mutation
            if selected_population_msgs:
                ind_mut = random.choice(selected_population_msgs)
                child_mut = mutate_heuristics_exp(ind_mut, policy_model, current_nudge_policy_instructions_str)
                if child_mut: next_generation_msgs.append(child_mut)

        if not next_generation_msgs:  # If operators failed to produce offspring
            print(f"No new individuals generated for generation {generation + 1}. Using selected population only.")
            next_gen_scores, next_gen_codes = [], []
        else:
            next_gen_scores, next_generation_msgs, next_gen_codes = evaluate_population_exp(
                next_generation_msgs, ground_truth_data, farm_model, fix_model,
                current_farm_prompt_instructions_str, exp_eval_dir,
                base_input_geojson_for_farm, input_json_data_for_eval
            )

        # Combine selected from previous generation with new offspring
        combined_population_msgs = selected_population_msgs + next_generation_msgs
        combined_scores = selected_scores + next_gen_scores
        combined_codes = selected_codes + next_gen_codes

        if not combined_scores:  # Should not happen if selected_scores existed
            print(f"No scores in combined population for gen {generation + 1}. Stopping.")
            break

        # Reflection
        r_scores, r_population_msgs, r_codes = run_reflect_exp(
            combined_population_msgs, combined_scores, policy_model, current_nudge_policy_instructions_str,
            ground_truth_data, farm_model, fix_model, current_farm_prompt_instructions_str,
            exp_eval_dir,
            base_input_geojson_for_farm, input_json_data_for_eval
        )

        # Add reflection results
        final_population_msgs = combined_population_msgs + r_population_msgs
        final_scores = combined_scores + r_scores
        final_codes = combined_codes + r_codes

        if not final_scores:
            print(f"No scores after reflection for gen {generation + 1}. Stopping.")
            break

        metrics_df = compute_radon_metrics(metrics_df, final_codes, final_scores)  # Compute for all evaluated

        best_idx_gen = np.argmax(final_scores)
        line = f"Experiment: {experiment_label} - Generation {generation + 1} : Best Score - {final_scores[best_idx_gen]:.4f}\n"
        print(line)
        with open(score_file_path, 'a+') as f:
            f.write(line)

        best_policy_file_gen = os.path.join(exp_gen_dir, f"best_policy_gen_{generation + 1}.txt")
        with open(best_policy_file_gen, 'w') as f:
            f.write(final_population_msgs[best_idx_gen])
        best_python_file_gen = os.path.join(exp_gen_dir, f"best_python_gen_{generation + 1}.py")
        if final_codes and final_codes[best_idx_gen]:
            with open(best_python_file_gen, 'w') as f: f.write(final_codes[best_idx_gen])

        metrics_df.to_csv(metrics_file_path, index=False)

        selected_population_msgs, selected_scores, selected_codes = select_population(
            final_population_msgs, final_scores, final_codes, population_size_target
        )

        # Save current selected population (optional, for resuming or inspection)
        for k, (msg, code_str) in enumerate(zip(selected_population_msgs, selected_codes)):
            sel_msg_file = os.path.join(exp_heur_dir,
                                        f"heuristics_gem_{k+1}.txt")
            sel_code_file = os.path.join(exp_heur_dir,
                                         f"heuristics_gem_{k+1}.py")
            # ensure dir exists...
            with open(sel_msg_file, 'w') as f:
                f.write(msg)
            if code_str:
                with open(sel_code_file, 'w') as f: f.write(code_str)

        if final_scores[best_idx_gen] > 10:  # Corresponds to original's early stopping
            print(f"High score reached for {experiment_label}. Stopping early.")
            break
    print(f"--- Finished Evolutionary Strategy for: {experiment_label} ---")


def run_single_experiment(
        farm_id_str, personality_key, nudge_type_key,
        global_cfg,
        population_size=25, num_generations=10, inner_loop_size=10, init_new_population=True):
    experiment_label = f"pers_{personality_key}_nudge_{nudge_type_key}"
    print(f"\n========== RUNNING EXPERIMENT: {experiment_label} ==========")

    # --- Load Farm-Specific Base Data (paths need to be accurate) ---
    # These paths are based on the original script's structure. Adjust if your layout differs.
    farm_data_root = os.path.join(global_cfg.data_dir, "crop_inventory", "syn_farms", f"farm_{farm_id_str}")
    nudge_data_dir_for_farm = os.path.join(farm_data_root, "nudge")  # Contains base heuristics

    heur_eco_intens_path = os.path.join(nudge_data_dir_for_farm, "heuristics_gem_eco_intens.py")
    heur_eco_conn_path = os.path.join(nudge_data_dir_for_farm, "heuristics_gem_eco_conn.py")
    base_input_geojson_path = os.path.join(farm_data_root, "input.geojson")  # The clean input.geojson for this farm
    ground_truth_path = os.path.join(farm_data_root, "connectivity", "run_1",
                                     'output_gt_directions.json')  # From original

    # --- Setup Directories for this specific experiment run ---
    exp_run_base_dir = os.path.join(nudge_data_dir_for_farm, experiment_label)
    exp_heur_init_dir = os.path.join(exp_run_base_dir, "heuristics")  # For initially generated policies
    if not os.path.exists(exp_heur_init_dir): os.makedirs(exp_heur_init_dir)

    try:
        with open(heur_eco_intens_path, "r") as f:
            heur_eco_intens_str = f.read()
        with open(heur_eco_conn_path, "r") as f:
            heur_eco_conn_str = f.read()
        with open(base_input_geojson_path) as f:
            input_json_data = json.load(f)  # Loaded once for error calc
        with open(ground_truth_path) as f:
            ground_truth_features = json.load(f)  # Assuming GT is like output.json
    except FileNotFoundError as e:
        print(f"Error loading base data for {experiment_label}: {e}. Skipping experiment.")
        return

    # Base parameters (crop prices, costs) - assumed to be somewhat static or loaded similarly
    # For now, using the hardcoded string from the original snippet for params_instructions
    params_instructions_str = (
        "These are the crop prices in USD/Tonne: {'Soybeans': 370, 'Oats': 95, 'Corn': 190, 'Canola/rapeseed': 1100, "
        "'Barley': 120, 'Spring wheat': 200}, and these are the costs (implementation costs one time and in USD/ha, and "
        "maintenance costs in USD/ha/year) : {'margin': {'implementation': 400,  'maintenance': 60}, 'habitat': {"
        "'implementation': 300, 'maintenance': 70}, 'agriculture': {'maintenance': 100}}.\n\n"
    )

    # --- Generate Nudge-Specific and Farmer-Specific Instructions ---
    neighbour_data = os.path.join(nudge_data_dir_for_farm, "farm_neighbours.txt")
    with open(neighbour_data, "r") as f:
        social_comparison_data_str = f.read()
    current_nudge_policy_instructions = get_nudge_specific_policy_instructions(
        nudge_type_key, params_instructions_str, heur_eco_intens_str, heur_eco_conn_str, social_comparison_data_str)

    # Base farmer task instructions (dynamic part based on current heuristics)
    current_farm_prompt_instructions = (  # Matches original structure
        f"The current set of heuristics you follow are:\n```python\n{heur_eco_intens_str}\n```\n"
    )

    # --- Initialize Models ---
    # Extract model name from config
    config_model = cfg.lm.split('/')[-1] if '/' in cfg.lm else cfg.lm
    policy_model, farm_model, fix_model = init_experimental_gemini_model(
        personality_key,
        policy_model_name=config_model,
        farm_model_name=config_model,
        fix_model_name=config_model
    )

    # --- Initial Population ---
    if init_new_population:
        print(f"Generating initial population for {experiment_label}...")
        # The prompt for initial generation IS the nudge_policy_instructions
        initial_population_messages = create_initial_population_exp(
            policy_model, current_nudge_policy_instructions, population_size, exp_heur_init_dir
        )
        if not initial_population_messages:
            print(f"Failed to create any initial policies for {experiment_label}. Skipping.")
            return
    else:  # Load existing if not initializing (useful for reruns/debugging specific stages)
        print(f"Loading existing initial population for {experiment_label} from {exp_heur_init_dir}...")
        initial_population_messages = get_initial_population_exp(population_size, exp_heur_init_dir)
        if not initial_population_messages:
            print(
                f"Failed to load initial policies for {experiment_label}. Ensure they exist or set init_new_population=True. Skipping.")
            return

    # --- Run Evolutionary Strategy ---
    run_evo_strat_exp(
        experiment_label, initial_population_messages,
        policy_model, farm_model, fix_model,
        current_nudge_policy_instructions, current_farm_prompt_instructions,
        ground_truth_features, base_input_geojson_path, input_json_data,
        exp_run_base_dir,  # Base output directory for THIS experiment's files
        population_size, num_generations, inner_loop_size
    )
    print(f"========== FINISHED EXPERIMENT: {experiment_label} ==========\n")


# ======================== MAIN EXECUTION ========================
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    global_cfg_instance = Config()  # Your global configuration

    # --- Define Experiment Parameters ---
    farm_ids_to_run = ["3"]  # Example farm ID from original code, make it a list of strings
    farmer_personality_keys_to_run = ["resistant", "economic", "socially_influenced"]
    policy_nudge_keys_to_run = ["behavioral", "economic"]

    # --- Loop Through All Combinations ---
    for farm_id in farm_ids_to_run:
        for personality in farmer_personality_keys_to_run:
            for nudge_type in policy_nudge_keys_to_run:
                run_single_experiment(
                    farm_id_str=farm_id,
                    personality_key=personality,
                    nudge_type_key=nudge_type,
                    global_cfg=global_cfg_instance,
                    population_size=25,  # Smaller for quicker test runs
                    num_generations=10,  # Smaller for quicker test runs
                    inner_loop_size=10,  # Smaller for quicker test runs
                    init_new_population=True  # Set to False to reuse generated policies for a given combo
                )
    print("\nAll experimental combinations processed.")
