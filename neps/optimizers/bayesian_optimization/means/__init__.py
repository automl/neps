from .means import ConstantMean, GpMean, LinearMean, MeanComposer, ZeroMean

MEANS_MAPPING = {
    "zero": ZeroMean,
    "constant": ConstantMean,
    "linear": LinearMean,
}
