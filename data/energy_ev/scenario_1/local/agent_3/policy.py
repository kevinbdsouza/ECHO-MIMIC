self.location_id = self.scenario['agent_config']['location']
        self.base_demand = np.array(self.scenario['agent_config']['base_demand'])
        self.slot_min_sessions = np.array(self.scenario['slot_min_sessions'])
        self.slot_max_sessions = np.array(self.scenario['slot_max_sessions'])