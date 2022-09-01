from .means import ConstantMean, GPMean, LinearMean, MeanComposer, ZeroMean

MEANS_MAPPING = {
    "zero": ZeroMean,
    "constant": ConstantMean,
    "linear": LinearMean,
}
