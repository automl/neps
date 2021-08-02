from copy import deepcopy


class Sampler:
    def __init__(self, args, objective):
        self.pool_strategy = args.pool_strategy
        self.objective = objective

    def sample(self, pool_size):

        pool = []
        while len(pool) < pool_size:
            rand_config = deepcopy(self.objective)
            rand_config.sample_random_architecture()
            pool.append(rand_config)

        return pool
