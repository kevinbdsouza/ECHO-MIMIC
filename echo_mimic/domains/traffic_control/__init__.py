
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from echo_mimic.domain.domain import Domain
from echo_mimic.domain.simulation_config import SimulationConfig

# --- Configuration ---

TrafficControlMode = Literal["local", "global", "nudge", "braess_test"]


@dataclass
class TrafficControlDomainConfig(SimulationConfig):
    """
    Configuration for the Traffic Control Domain.
    """
    mode: TrafficControlMode = "local"
    city_grid_size: int = 10  # e.g., 10x10 grid
    junction_capacities: Dict[str, int] = field(default_factory=lambda: {"A": 100, "B": 120})
    braess_paradox_enabled: bool = False
    time_step_s: float = 60.0
    initial_traffic_demand_factor: float = 1.0

    def __post_init__(self):
        # Example validation/setup based on mode
        if self.mode == "braess_test":
            print("Configuring environment specifically for Braess's Paradox demonstration.")


# --- Domain Implementation ---

class TrafficControlDomain(Domain):
    """
    Manages traffic flow and congestion minimization in a simulated city grid.
    
    Addresses the collective action problem exemplified by Braess's Paradox,
    where adding infrastructure can sometimes worsen overall travel times.
    """

    def __init__(self, config: TrafficControlDomainConfig):
        super().__init__(config)
        self.config: TrafficControlDomainConfig = config
        self.current_state: Dict[str, Any] = {}
        self.initialize_environment()
        print(f"TrafficControlDomain initialized in mode: {self.config.mode}")

    def initialize_environment(self):
        """Sets up the initial state of the city grid."""
        print(f"Initializing {self.config.city_grid_size}x{self.config.city_grid_size} traffic grid.")
        # Placeholder for complex environment setup (roads, origins, destinations)
        self.current_state["junction_utilization"] = {
            f"J_{i}_{j}": 0.5 for i in range(self.config.city_grid_size) for j in range(self.config.city_grid_size)
        }
        self.current_state["total_delay_s"] = 0.0

    def _run_local_strategy(self) -> Dict[str, Any]:
        """Agents optimize based only on local congestion feedback."""
        print("Executing Local Control Strategy...")
        # Logic for local signal timing updates
        new_delay = self.current_state["total_delay_s"] * 0.95 + 5.0  # Simple update example
        return {"strategy": "local", "updated_delay": new_delay}

    def _run_global_strategy(self) -> Dict[str, Any]:
        """A central authority optimizes flow across the entire network."""
        print("Executing Global Optimization Strategy...")
        # Logic for centralized path planning or signal coordination
        new_delay = self.current_state["total_delay_s"] * 0.85
        return {"strategy": "global", "updated_delay": new_delay}

    def _run_nudge_strategy(self) -> Dict[str, Any]:
        """Applies minor, non-disruptive adjustments based on predicted future state."""
        print("Executing Nudge Strategy...")
        # Logic for proactive, small adjustments
        new_delay = self.current_state["total_delay_s"] * 0.90 + 1.0
        return {"strategy": "nudge", "updated_delay": new_delay}

    def _run_braess_test(self) -> Dict[str, Any]:
        """Specific scenario testing Braess's paradox behavior."""
        print("Executing Braess Paradox Test Scenario...")
        # This mode might deliberately introduce a 'shortcut' road and measure the resulting delay increase.
        if self.config.braess_paradox_enabled:
            print("Warning: Braess shortcut is active. Expect potential counter-intuitive results.")
        new_delay = self.current_state["total_delay_s"] + 10.0 if self.config.braess_paradox_enabled else self.current_state["total_delay_s"] * 0.9
        return {"strategy": "braess_test", "updated_delay": new_delay}

    def run(self, num_steps: int = 100) -> List[Dict[str, Any]]:
        """
        Executes the traffic simulation for a given number of steps based on the configured mode.
        """
        history = []
        print(f"\n--- Starting Traffic Control Simulation ({self.config.mode}) ---")

        for step in range(num_steps):
            observation = {}

            if self.config.mode == "local":
                observation = self._run_local_strategy()
            elif self.config.mode == "global":
                observation = self._run_global_strategy()
            elif self.config.mode == "nudge":
                observation = self._run_nudge_strategy()
            elif self.config.mode == "braess_test":
                observation = self._run_braess_test()
            else:
                raise ValueError(f"Unknown traffic control mode: {self.config.mode}")

            # Update state based on strategy output (simplified)
            if "updated_delay" in observation:
                self.current_state["total_delay_s"] = observation["updated_delay"]

            history.append({
                "step": step,
                "mode": self.config.mode,
                "delay": self.current_state["total_delay_s"],
                "details": observation
            })

            if step % 10 == 0:
                print(f"Step {step}: Current Delay = {self.current_state['total_delay_s']:.2f}s")

        print("--- Simulation Finished ---")
        return history


if __name__ == '__main__':
    # Example Usage:
    local_config = TrafficControlDomainConfig(mode="local", city_grid_size=5)
    local_domain = TrafficControlDomain(local_config)
    local_results = local_domain.run(num_steps=20)
    # print(local_results[-1])

    braess_config = TrafficControlDomainConfig(mode="braess_test", braess_paradox_enabled=True)
    braess_domain = TrafficControlDomain(braess_config)
    braess_results = braess_domain.run(num_steps=20)
    # print(braess_results[-1])
