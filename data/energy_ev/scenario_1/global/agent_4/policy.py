# Original:
# self.alpha = scenario_data['alpha']
# self.beta = scenario_data['beta']
# self.gamma = scenario_data['gamma']

# Corrected assumption:
params = scenario_data.get('policy_parameters', {})
self.alpha = params.get('alpha', 1.0)  # Default to 1.0 if missing
self.beta = params.get('beta', 1.0)
self.gamma = params.get('gamma', 1.0)