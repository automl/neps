from .base_acq_optimizer import AcquisitionOptimizer


class RandomSampler(AcquisitionOptimizer):
    def __init__(self, args, objective):
        super().__init__(objective=objective)
        self.optimize_arch = args.optimize_arch
        self.optimize_hps = args.optimize_hps
        self.pool_strategy = args.pool_strategy

    def sample(self, pool_size):
        pool = []
        while len(pool) < pool_size:
            rand_config = self.objective.sample(
                optimize_arch=self.optimize_arch, optimize_hps=self.optimize_hps
            )
            pool.append(rand_config)

        return pool
