import numpy as np

from neps.plot.plot3D import Plotter3D

from .priorband_template import pipeline_space, run_pipeline


ASSUMED_MAX_LOSS = 10


def ifbo_run_pipeline(
    pipeline_directory,  # The directory where the config is saved
    previous_pipeline_directory,  # The directory of the config's immediate lower fidelity
    **config,  # The hyperparameters to be used in the pipeline
) -> dict | float:
    result_dict = run_pipeline(
        pipeline_directory=pipeline_directory,  # NOTE: can only support <=10 HPs and no categoricals
        previous_pipeline_directory=previous_pipeline_directory,
        **config,
    )
    # NOTE: Normalize the loss to be between 0 and 1
    ## crucial for ifBO's FT-PFN surrogate to work as expected
    result_dict["loss"] = np.clip(result_dict["loss"], 0, ASSUMED_MAX_LOSS) / ASSUMED_MAX_LOSS
    return result_dict


if __name__ == "__main__":
    import neps

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space(),
        root_directory="results",
        max_evaluations_total=50,
        searcher="ifbo",
    )
# end of ifbo_run_pipeline