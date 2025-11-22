"""Helpers for instruction-writing and evaluation agent creation."""
from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from echo_mimic.common.models import build_model_client


@dataclass
class DomainDefinition:
    name: str
    task_description: str
    input_schema: str
    objectives: List[str]
    evaluation_criteria: List[str]
    collective_action_problem: str
    model: str = "gemini-flash-lite-latest"
    agent_id: Optional[str] = None


def generate_domain_artifacts(definition: DomainDefinition, output_base_dir: str) -> None:
    """Generates all necessary artifacts for a new domain."""
    domain_slug = definition.name.lower().replace(" ", "_")
    domain_dir = os.path.join(output_base_dir, domain_slug)
    os.makedirs(domain_dir, exist_ok=True)

    client = build_model_client(definition.model)

    # 1. Generate __init__.py with Domain class and Config
    init_content = _generate_init_py(definition, client)
    with open(os.path.join(domain_dir, "__init__.py"), "w") as f:
        f.write(init_content)

    # 2. Generate prompts.py
    prompts_content = _generate_prompts_py(definition, client)
    with open(os.path.join(domain_dir, "prompts.py"), "w") as f:
        f.write(prompts_content)

    # 3. Generate evaluation.py
    eval_content = _generate_evaluation_py(definition, client)
    with open(os.path.join(domain_dir, "evaluation.py"), "w") as f:
        f.write(eval_content)

    # 4. Generate placeholder strategies
    _generate_strategies(definition, domain_dir)

    print(f"Domain artifacts generated in: {domain_dir}")


def _generate_init_py(definition: DomainDefinition, client) -> str:
    prompt = textwrap.dedent(f"""
        You are a Python code generator. Generate the `__init__.py` file for a new domain named "{definition.name}" in the ECHO-MIMIC agent framework.
        
        Context:
        - Task: {definition.task_description}
        - Collective Action Problem: {definition.collective_action_problem}
        
        Requirements:
        1. Define a `class {definition.name}DomainConfig` dataclass.
        2. Define a `class {definition.name}Domain` class.
        3. The Domain class must have a `run()` method.
        4. The `run()` method should dispatch to different strategies based on `config.mode` (e.g., 'local', 'global', 'nudge').
        5. Import necessary modules.
        6. Follow the pattern of existing domains (Farm, Energy).
        
        Return ONLY the Python code.
    """)
    return client.generate(prompt).replace("```python", "").replace("```", "")


def _generate_prompts_py(definition: DomainDefinition, client) -> str:
    prompt = textwrap.dedent(f"""
        You are an instruction-writing agent. Generate the `prompts.py` file for the "{definition.name}" domain.
        
        Context:
        - Task: {definition.task_description}
        - Inputs: {definition.input_schema}
        - Objectives: {', '.join(definition.objectives)}
        - Evaluation criteria: {', '.join(definition.evaluation_criteria)}
        
        Requirements:
        1. Define string constants or functions that return prompts for:
           - SYSTEM_PROMPT
           - AGENT_INSTRUCTIONS
           - EVALUATION_PROMPT
        2. The prompts should be detailed and tailored to the domain.
        
        Return ONLY the Python code.
    """)
    return client.generate(prompt).replace("```python", "").replace("```", "")


def _generate_evaluation_py(definition: DomainDefinition, client) -> str:
    prompt = textwrap.dedent(f"""
        You are an evaluation harness designer. Generate the `evaluation.py` file for the "{definition.name}" domain.
        
        Context:
        - Objectives: {', '.join(definition.objectives)}
        - Evaluation criteria: {', '.join(definition.evaluation_criteria)}
        
        Requirements:
        1. Define a JSON schema for scoring outputs.
        2. Define functions to parse and score agent outputs.
        3. Include a `score_output(output: str) -> dict` function.
        
        Return ONLY the Python code.
    """)
    return client.generate(prompt).replace("```python", "").replace("```", "")


def _generate_strategies(definition: DomainDefinition, domain_dir: str) -> None:
    strategies = ["local_strat.py", "global_strat.py", "nudge_strat.py"]
    for strat in strategies:
        with open(os.path.join(domain_dir, strat), "w") as f:
            f.write(textwrap.dedent(f"""
                # TODO: Implement simulation logic for {definition.name} ({strat.split('_')[0]} mode)
                # This file should contain the specific logic for running the simulation in this mode.
                # It will be called by the {definition.name}Domain class in __init__.py.
                
                def run(**kwargs):
                    raise NotImplementedError("This strategy has not been implemented yet.")
            """))


def guide_integration(definition: DomainDefinition) -> None:
    """Prints instructions for integrating the new domain."""
    domain_slug = definition.name.lower().replace(" ", "_")
    class_name = f"{definition.name}Domain"
    config_name = f"{definition.name}DomainConfig"
    
    print(textwrap.dedent(f"""
        ============================================================
        Integration Guide for "{definition.name}" Domain
        ============================================================
        
        1. Verify the generated files in `echo_mimic/domains/{domain_slug}/`.
           - Check `__init__.py`, `prompts.py`, `evaluation.py`, and strategy files.
           - Implement the simulation logic in the strategy files.
           
        2. Update `echo_mimic/orchestrator.py`:
        
           a. Import the new domain:
              from echo_mimic.domains.{domain_slug} import {class_name}, {config_name}
              
           b. Update the `run()` method in `Orchestrator` class:
              
              elif domain == "{domain_slug}":
                  {domain_slug}_domain = {class_name}(
                      {config_name}(
                          mode=self.config.mode,
                          method=self.config.method,
                          agent_id=self.config.agent_id,
                          model=self.config.model,
                          # Add other config parameters as needed
                      )
                  )
                  {domain_slug}_domain.run()
                  
        3. Run the orchestrator with the new domain:
           python -m echo_mimic.main --domain {domain_slug} ...
           
        ============================================================
    """))

if __name__ == "__main__":
    # Example usage for testing
    print("This module provides domain generation utilities.")
    print("To use, import `generate_domain_artifacts` and `guide_integration`.")

