import json
import logging
import os
import sys
import time
import warnings

from functools import partial

import click

from comprehensive_nas.bo.utils.plot import STRING_TO_PLOTTYPE
from comprehensive_nas.bo.utils.plot import PlotTypes
from comprehensive_nas.bo.utils.plot import make_incumbent_plot


optimizers = [
    "random_search",
    # 'DEHB',
    "bayes_opt",
]

results_dict = dict()
benchmark_name = "nasbench301"
dir = "demo/plots/{}".format(benchmark_name)
seed = 56
max_iters = 100
n_repetitions = 5

for exp in ["hpo", "nas", "nas_hpo"]:

    for name in optimizers:

        res = "{}/{}".format(dir, name)

        results_dict[name] = dict()
        results_dict[name]["incumbent_values"] = list()
        results_dict[name]["runtime"] = list()

        for i in range(n_repetitions):
            try:
                with open(
                    os.path.join(res, "{}/{}:{}.json".format(exp, max_iters, seed + i)),
                    "r",
                ) as f:
                    result = json.load(f)

                results_dict[name]["incumbent_values"].append(result["incumbent_fval"])
                results_dict[name]["runtime"].append(result["runtime"])
                # [-len(result['incumbent_fval']):])
            except:
                pass

    make_incumbent_plot(
        results_dict,
        optimizers,
        dir,
        benchmark_name=benchmark_name,
        title_type=PlotTypes.INCUMBENT_RUNTIME,
        experiment_name=exp,
    )
