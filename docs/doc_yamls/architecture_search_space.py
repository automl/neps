
from torch import nn
import neps
from neps.search_spaces.architecture import primitives as ops
from neps.search_spaces.architecture import topologies as topos
from neps.search_spaces.architecture.primitives import AbstractPrimitive


class DownSampleBlock(AbstractPrimitive):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(locals())
        self.conv_a = ReLUConvBN(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.conv_b = ReLUConvBN(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
            ),
        )

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        residual = self.downsample(inputs)
        return residual + basicblock


class ReLUConvBN(AbstractPrimitive):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__(locals())

        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return self.op(x)


class AvgPool(AbstractPrimitive):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.op = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.op(x)


primitives = {
    "Sequential15": topos.get_sequential_n_edge(15),
    "DenseCell": topos.get_dense_n_node_dag(4),
    "down": {"op": DownSampleBlock},
    "avg_pool": {"op": AvgPool},
    "id": {"op": ops.Identity},
    "conv3x3": {"op": ReLUConvBN, "kernel_size": 3, "stride": 1, "padding": 1},
    "conv1x1": {"op": ReLUConvBN, "kernel_size": 1, "stride": 1, "padding": 0},
}


structure = {
    "S": ["Sequential15(C, C, C, C, C, down, C, C, C, C, C, down, C, C, C, C, C)"],
    "C": ["DenseCell(OPS, OPS, OPS, OPS, OPS, OPS)"],
    "OPS": ["id", "conv3x3", "conv1x1", "avg_pool"],
}


def set_recursive_attribute(op_name, predecessor_values):
    in_channels = 16 if predecessor_values is None else predecessor_values["out_channels"]
    out_channels = in_channels * 2 if op_name == "DownSampleBlock" else in_channels
    return dict(in_channels=in_channels, out_channels=out_channels)


pipeline_space = dict(
    architecture=neps.ArchitectureParameter(
        set_recursive_attribute=set_recursive_attribute,
        structure=structure,
        primitives=primitives,
    ),
    optimizer=neps.CategoricalParameter(choices=["sgd", "adam"]),
    learning_rate=neps.FloatParameter(lower=10e-7, upper=10e-3, log=True),
)

