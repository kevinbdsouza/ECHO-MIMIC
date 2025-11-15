import os
import google.generativeai as genai
from echo_mimic.rate_limiter import RateLimiter, send_message_with_retry
from ei_ec.config import Config
import numpy as np
from dotenv import load_dotenv

explanation_system_prompt = (
    "You are a highly knowledgeable code analysis assistant with expertise in optimization algorithms and Python programming. "
    "You will be provided with a group of Python programs generated to solve an agricultural intervention planning task. "
    "These programs are ordered from lowest to highest performance. Your task is to carefully examine the provided code and explain, "
    "in a detailed, step-by-step manner, the differences and improvements between successive versions.\n\n"
    "Guidelines:\n"
    "  - Identify modifications in heuristics, variable usage, control flow, and algorithmic structures that may lead to better performance.\n"
    "  - Explain how these modifications might improve the trade-off between intervention cost and increased yield, by affecting factors such as pollination and pest control.\n"
    "  - Point out any recurring patterns or trends across the successive versions.\n"
    "  - Explain your reasoning and think step by step before providing your final explanation.\n"
    "Do not include any information not present in the provided code or background context."
)

merge_system_prompt = (
    "You are a skilled analytical assistant tasked with synthesizing and merging insights from multiple analyses of Python programs designed for an agricultural optimization task. "
    "You will be provided with a previous summary of insights and a new detailed explanation generated for the current group of programs. "
    "Your goal is to merge these insights into an updated, consolidated summary.\n\n"
    "Guidelines:\n"
    "  - Explain your reasoning and think step by step before providing your final merged summary.\n"
    "  - Identify common patterns, modifications in heuristics, variables, and control structures that enhance performance.\n"
    "  - Ensure that the final summary is clear, comprehensive, and strictly based on the provided inputs.\n"
    "Do not include any extraneous or invented details."
)


def init_gemini_model(model_name="gemini-2.0-flash-thinking-exp-01-21", system_prompt=""):
    """
    Initialize the Gemini model with the specified model name.
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variables
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("No API key found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt
    )
    # Initialize rate limiter
    from echo_mimic.config import Config
    cfg = Config()
    global rate_limiter
    rate_limiter = RateLimiter(**cfg.rate_limit)
    
    return model


def run_explanation(model, code_snippet):
    common_task_instructions = (
        "\n\nTask Background:\n"
        "The Python programs are designed to decide which interventions to apply at agricultural plots. The interventions—'margin' (converting only the margins) "
        "and 'habitat' (converting contiguous regions)—may be fractional. The performance is evaluated based on "
        "their effect on the Net Present Value (NPV), which is influenced by pollination and pest control services affecting yield, balanced against the costs of implementation and maintenance. "
        "Existing habitat plots remain unaffected.\n"
    )

    prompt = (
        f"Please analyze the following group of Python programs:\n\n"
        f"{code_snippet}\n\n"
        f"{common_task_instructions}\n\n"
        "Explain your reasoning and think step by step before providing your final explanation:"
    )

    session = model.start_chat(history=[])
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    return response


def run_merge(model, previous_summary, current_explanation):
    common_task_instructions = (
        "\n\nTask Background:\n"
        "The programs under analysis decide on interventions at agricultural plots, balancing the cost of interventions against benefits in yield through improved pollination and pest control. "
        "They choose between 'margin' and 'habitat' interventions, while existing habitat plots remain unaffected.\n"
    )

    prompt = (
        "You have a previous summary of insights:\n"
        f"{previous_summary}\n\n"
        "Additionally, here is the new explanation for the current group:\n\n"
        f"{current_explanation}\n\n"
        f"{common_task_instructions}\n\n"
        "Please merge the previous insights with this new analysis. Explain your reasoning and think step by step before providing your updated, consolidated summary:"
    )

    session = model.start_chat(history=[])
    completion = send_message_with_retry(session, prompt, rate_limiter)
    response = completion.parts[0].text
    return response


def main():
    # Directory containing the heuristic files
    folder = "./"  # Change this if your files are in a different folder

    # Collect all files matching the pattern best_heuristics_gem_gen_X.py
    files = []
    index = 0
    while True:
        filename = os.path.join(run_dir, f"best_heuristics_gem_gen_{index}.py")
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            break  # Stop when no file is found for the current index
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        files.append(content)
        index += 1

    if not files:
        print("No heuristic files found matching the pattern.")
        return

    # Initialize the Gemini model
    from echo_mimic.config import Config
    cfg = Config()
    config_model = cfg.lm.split('/')[-1] if '/' in cfg.lm else cfg.lm
    explanation_model = init_gemini_model(model_name=config_model,
                                          system_prompt=explanation_system_prompt)
    merge_model = init_gemini_model(model_name=config_model, system_prompt=merge_system_prompt)

    # Process the files in groups of 3
    num_files = len(files)
    current_summary = ""  # This will hold the aggregated summary across groups
    group_index = 0

    while group_index < num_files:
        print(f"Running heuristic programs: {group_index}-{group_index+2}")

        # Group next 3 files (or the remaining ones if less than 3)
        group_files = files[group_index:group_index + 3]
        group_code = ""
        for i, code in enumerate(group_files, start=group_index):
            group_code += f"\n# File: best_heuristics_gem_gen_{i}.py\n{code}\n"

        # Call run_explanation every loop to analyze the current group
        current_explanation = run_explanation(explanation_model, group_code)

        if group_index == 0:
            # For the first group, the explanation becomes the current summary.
            current_summary = current_explanation
        else:
            # For subsequent groups, merge the previous summary with the new explanation.
            current_summary = run_merge(merge_model, current_summary, current_explanation)

        output_file = os.path.join(run_dir, f"summary_{group_index}-{group_index+2}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(current_summary)
        group_index += 3

    final_summary = current_summary

    # Write the final consolidated summary to a text file
    output_file = os.path.join(run_dir, "final_summary.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print(f"Final summary saved to {output_file}")


if __name__ == "__main__":
    cfg = Config()
    farm_ids = np.arange(4, 6)
    run_id = 1

    for farm_id in farm_ids:
        print(f"Running farm: {farm_id}")
        run_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", f"farm_{farm_id}", "heuristics",
                               f"run_{run_id}")
        main()
