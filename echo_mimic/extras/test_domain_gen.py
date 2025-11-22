import os
import sys

# Add the current directory to sys.path so we can import echo_mimic
sys.path.append(os.getcwd())

from echo_mimic.prompt_orchestration import DomainDefinition, generate_domain_artifacts, guide_integration

def main():
    definition = DomainDefinition(
        name="Traffic Control",
        task_description="Manage traffic flow in a city grid to minimize congestion.",
        input_schema="Intersection data (cars waiting, light state)",
        objectives=["Minimize average wait time", "Maximize throughput"],
        evaluation_criteria=["Average wait time per car", "Total cars processed"],
        collective_action_problem="Braess's paradox",
        model="gemini-flash-lite-latest"
    )

    output_dir = os.path.join(os.getcwd(), "echo_mimic", "domains")
    
    print("Generating domain artifacts...")
    generate_domain_artifacts(definition, output_dir)
    
    print("\nGuiding integration...")
    guide_integration(definition)

if __name__ == "__main__":
    main()
