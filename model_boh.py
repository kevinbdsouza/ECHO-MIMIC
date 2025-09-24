from openai import OpenAI
import google.generativeai as genai
import json
import os
from pathlib import Path
import numpy as np
from rate_limiter import send_message_with_retry
from iga_eh.create_prompts import *
from tools import *
from config import Config
from common import (
    build_model,
    configure_genai,
    ensure_rate_limiter,
    extract_python_code,
    make_code_validator,
)


cfg = Config()
rate_limiter = ensure_rate_limiter(cfg)
capture = CommandOutputCapture()


def run_gpt4omini(farm_id):
    prompt_input_path = os.path.join(cfg.data_dir, "crop_inventory", "farms", f"farm_{farm_id}",
                                     "prompt_input.txt")
    heuristics_file = os.path.join(cfg.data_dir, "crop_inventory", "farms", f"farm_{farm_id}",
                                   "heuristics_mini.py")

    with open(prompt_input_path, "r") as f:
        prompt = f.read()

    print("Running GPT-4o-mini")
    client = OpenAI(api_key="sk-proj-nEY2rmvi03SHS_xhM30qsLDn7pjdnJ_MwgLfx9vqP6QLRv5jEY3rR9f0yYnsw_"
                            "3YPQrQ5MciuVT3BlbkFJj6HyZ-zMmMTp8JGfAu_bnDbtvYrxI5VwDq1We6vxifTa_oQIo5rEBifc_LE75iQyrKaAzDdfAA")
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text":
                            "You are a helpful assistant who is an expert in spatial optimization methods "
                            "and who helps the user with optimization related queries. You will return final answers in python code."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    print("Extracting heuristics python code")
    response = completion.choices[0].message.content
    python_code = extract_python_code(response)

    print("Saving heuristics")
    with open(heuristics_file, 'w') as f:
        f.write(python_code)


def init_gemini_model():
    configure_genai()

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    system_instruction = (
        "You are a helpful assistant who is an expert in spatial optimization methods "
        "and who helps the user with optimization related queries. You will return final answers in python code. "
        "You can't use the given values of interventions (margin_intervention and habitat_conversion) to "
        "produce the output, they are only for reference to produce the heuristics. "
        "You can load the input geojson from input.geojson. You can only use this input data and no other data. "
        "Save outputs to output.geojson."
    )

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
        system_instruction=system_instruction,
    )
    fix_model = build_model(
        "gemini-2.0-flash-exp",
        "You are a helpful assistant who is an expert in spatial optimization methods and python. "
        "Given the python code and the stack traceback, fix the errors and return the correct functioning python code.",
        ensure_configured=False,
    )

    global rate_limiter
    rate_limiter = ensure_rate_limiter(cfg)

    return model, fix_model



def validate_code_in_dir(
    code: str,
    *,
    workdir: Path,
    fix_model,
    script_name: str,
    max_attempts: int = 2,
) -> str:
    validator = make_code_validator(
        workdir=workdir,
        capture=capture,
        fix_model=fix_model,
        rate_limiter=rate_limiter,
        default_script=script_name,
        default_attempts=max_attempts,
    )
    return validator(code, script_name=script_name, max_attempts=max_attempts)


def run_gemini_flashexp(prompt_path, heuristics_file):
    with open(prompt_path, "r") as f:
        prompt = f.read()


    session = model.start_chat(
        history=[
        ]
    )

    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    python_code = extract_python_code(response)

    print("Saving heuristics")
    with open(heuristics_file, 'w') as f:
        f.write(python_code)


def compute_pred_error(farm_id):
    farm_path = os.path.join(cfg.data_dir, "crop_inventory", "farms", f"farm_{farm_id}")

    os.chdir(farm_path)
    os.system("python heuristics.py")
    os.chdir(cfg.src_dir)

    # Load the ground truth and output JSON files
    with open(os.path.join(farm_path, 'output_gt.geojson')) as f:
        ground_truth = json.load(f)

    with open(os.path.join(farm_path, 'output.geojson')) as f:
        predicted_output = json.load(f)

    # Extract features from both ground truth and predicted output
    gt_features = ground_truth["features"]
    pred_features = predicted_output["features"]

    # Initialize error accumulators for margin_intervention and habitat_conversion
    margin_errors = []
    habitat_errors = []

    # Iterate over the features and calculate the error
    for gt_feature, pred_feature in zip(gt_features, pred_features):
        gt_properties = gt_feature["properties"]
        pred_properties = pred_feature["properties"]

        # Calculate the absolute error for margin_intervention and habitat_conversion
        margin_error = abs(gt_properties["margin_intervention"] - pred_properties["margin_intervention"])
        habitat_error = abs(gt_properties["habitat_conversion"] - pred_properties["habitat_conversion"])

        # Append the errors to the respective lists
        margin_errors.append(margin_error)
        habitat_errors.append(habitat_error)

    # Compute overall error as mean absolute error
    overall_margin_error = np.mean(margin_errors)
    overall_habitat_error = np.mean(habitat_errors)

    # Print the results
    print(f"Overall Margin Intervention Error: {overall_margin_error}")
    print(f"Overall Habitat Conversion Error: {overall_habitat_error}")


def run_plots():
    plot_dirs = os.path.join(cfg.data_dir, "crop_inventory", "farms", f"farm_{farm_id}", "plots_geojsons")
    plots = os.listdir(plot_dirs)
    for i, plot in enumerate(plots):
        try:
            plot_id = int(plot.split("_")[1])
        except:
            continue

        plot_dir = os.path.join(plot_dirs, plot)
        heur_file = os.path.join(plot_dir, "heuristics_gem.py")
        # if os.path.exists(heur_file):
        #    continue

        print(f"Running plot :{plot_id}")

        create_plot_prompt_file(cfg, farm_id, plot_id)
        prompt_path = os.path.join(plot_dir, "prompt_input.txt")
        run_gemini_flashexp(prompt_path, heur_file)

        with open(heur_file, 'r') as f:
            heuristics_code = f.read()

        heuristics_code = validate_code_in_dir(
            heuristics_code,
            workdir=Path(plot_dir),
            fix_model=fix_model,
            script_name="heuristics_gem.py",
            max_attempts=3,
        )

        print("Saving heuristics")
        with open(heur_file, 'w') as f:
            f.write(heuristics_code)


def run_farms():
    pop_size = 25

    farm_dir = os.path.join(cfg.data_dir, "crop_inventory", "farms", f"farm_{farm_id}")
    heur_dir = os.path.join(farm_dir, "heuristics")
    if not os.path.exists(heur_dir):
        os.makedirs(heur_dir)

    create_farm_prompt_file(cfg, farm_id)
    prompt_path = os.path.join(cfg.data_dir, "crop_inventory", "farms", f"farm_{farm_id}",
                                     "prompt_input.txt")

    for i in range(1, pop_size+1):
        heur_file = os.path.join(heur_dir, "heuristics_gem_" + str(i) + ".py")

        run_gemini_flashexp(prompt_path, heur_file)

        with open(heur_file, 'r') as f:
            heuristics_code = f.read()

        heuristics_code = validate_code_in_dir(
            heuristics_code,
            workdir=Path(heur_dir),
            fix_model=fix_model,
            script_name=f"heuristics_gem_{i}.py",
            max_attempts=3,
        )

        print("Saving heuristics")
        with open(heur_file, 'w') as f:
            f.write(heuristics_code)


if __name__ == "__main__":
    cfg = Config()
    rate_limiter = ensure_rate_limiter(cfg)
    capture = CommandOutputCapture()
    farm_id = 5
    plot_id = 0

    #run_gpt4omini(farm_id)

    #compute_pred_error(farm_id)

    model, fix_model = init_gemini_model()

    #run_plots()

    run_farms()
