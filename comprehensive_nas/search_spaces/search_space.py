import random


class SearchSpace:
    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters

    def sample_config(self):
        config = []
        for hyperparameter in self.hyperparameters:
            config.append(hyperparameter.sample())

    def mutate(self, config, mutate_probability_per_hyperparameter=0.5):
        for hyperparameter in config:
            if random.random() < mutate_probability_per_hyperparameter:
                hyperparameter.mutate()
