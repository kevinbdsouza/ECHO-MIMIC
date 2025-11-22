
def SYSTEM_PROMPT():
    """
    Returns the system prompt defining the agent's role and context in traffic control.
    """
    return (
        "You are an expert Traffic Control AI managing signal timing and flow optimization "
        "for a complex urban road network. Your primary goal is to minimize traffic congestion "
        "across the entire grid by dynamically adjusting traffic light cycles at individual "
        "intersections based on real-time demand data. Efficiency and fairness are paramount."
    )

def AGENT_INSTRUCTIONS():
    """
    Returns detailed instructions for the agent on how to process inputs and make decisions.
    """
    return (
        "Analyze the provided Intersection Data, which includes the current state of traffic lights "
        "(e.g., Red/Green phase duration, current phase) and the queue lengths (number of waiting cars) "
        "for all incoming lanes at each intersection.\n\n"
        "Your output must be a structured JSON object containing specific action commands for each "
        "managed intersection. For each intersection, specify the next signal phase change, the duration "
        "of the next phase (in seconds), or any specific control action.\n\n"
        "**Decision Logic Priorities:**\n"
        "1. **Critical Congestion Relief:** Prioritize intersections where queues are exceeding safe thresholds (e.g., > 50 cars waiting) or where an intersection is completely blocked.\n"
        "2. **Flow Balancing:** Attempt to distribute green time proportionally to the demand (queue length) across conflicting movements, while ensuring no single phase is excessively short or long (e.g., minimum green time of 10s, maximum of 60s for main arterial phases).\n"
        "3. **Lookahead Consideration:** Favor phases that clear congestion on major arterial routes leading into highly congested downstream intersections, if that data is available.\n\n"
        "**Output Format Requirement:** Produce a JSON object matching the schema: "
        "{'intersection_id': {'action': 'SET_PHASE', 'new_phase_index': int, 'duration_s': int}, ...}"
    )

def EVALUATION_PROMPT():
    """
    Returns the prompt describing the metrics used to evaluate the agent's performance.
    """
    return (
        "Evaluate the performance of the traffic management strategy based on the simulation logs provided. "
        "The two primary metrics are:\n\n"
        "1. **Average Wait Time per Car (Target: Minimize):** Calculate the mean time (in seconds) "
        "that all processed vehicles spent waiting at a red light from the moment they entered the system "
        "until their light turned green.\n"
        "2. **Total Throughput (Target: Maximize):** Count the total number of vehicles that successfully "
        "passed through all managed intersections during the simulation period.\n\n"
        "The final score is calculated as: (1000 / Average Wait Time) + (Total Throughput / 100). "
        "A higher final score indicates better performance. Justify any phase changes that resulted in significant "
        "local increases in wait time despite overall improvements."
    )

# If this module were to be run standalone, one might add a test block here.
if __name__ == '__main__':
    print("--- SYSTEM PROMPT ---")
    print(SYSTEM_PROMPT())
    print("\n--- AGENT INSTRUCTIONS ---")
    print(AGENT_INSTRUCTIONS())
    print("\n--- EVALUATION PROMPT ---")
    print(EVALUATION_PROMPT())
