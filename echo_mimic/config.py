import os
import numpy as np


class Config:
    def __init__(self):
        #cwd = os.getcwd()
        self.src_dir = os.getcwd()
        self.data_dir = os.path.join(self.src_dir, "data")
        self.plot_dir = os.path.join(self.src_dir, "plots")

        # DSPy configuration values
        self.farms_dir = os.path.join(self.data_dir)
        self.lm = "gemini-flash-lite-latest"
        self.auto = "heavy"
        self.trials = 50
        self.seed = 9
        
        # Rate limiting configuration
        self.rate_limit = {
            'max_retries': 5,
            'base_delay': 1.0,
            'max_delay': 60.0,
            'backoff_factor': 2.0,
            'jitter': True,
            'requests_per_minute': 14  # Conservative limit for Gemini free tier
        }

        # Define parameters (crop-dependent)
        self.params = {
            'crops': {
                'Spring wheat': {
                    'margin': {
                        'alpha': 0.05,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.05,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 200 #price in USD/Tonne
                },
                'Barley': {
                    'margin': {
                        'alpha': 0.05,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.05,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 120 #price in USD/Tonne
                },
                'Canola/rapeseed': {
                    'margin': {
                        'alpha': 0.20,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.10,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.20,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.10,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 1100 #price in USD/Tonne
                },
                'Corn': {
                    'margin': {
                        'alpha': 0.05,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.05,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 190 #price in USD/Tonne
                },
                'Oats': {
                    'margin': {
                        'alpha': 0.05,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.05,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 95 #price in USD/Tonne
                },
                'Soybeans': {
                    'margin': {
                        'alpha': 0.10,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.10,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.10,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.10,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 370 #price in USD/Tonne
                }
            },
            'habitats': [
                "Broadleaf", "Coniferous", "Exposed land/barren",
                "Grassland", "Shrubland", "Water", "Wetland"
            ],
            'r': 0.05,  # 5% discount rate
            't': 20,  # 20-year time horizon
            'costs': {
                'margin': {
                    'implementation': 400,  # USD/ha one-time cost
                    'maintenance': 60  # USD/ha/year
                },
                'habitat': {
                    'implementation': 300,  # USD/ha one-time cost
                    'maintenance': 70,  # USD/ha/year
                    'existing_hab': 0 # USD/ha/year
                },
                'agriculture': {
                    'maintenance': 100  # USD/ha/year baseline maintenance cost
                }
            }
        }
