try:
    import torch.nn as nn
except ModuleNotFoundError:
    from neps.utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)

from neps.search_spaces.graph_grammar.primitives import AbstractPrimitive, ReLUConvBN

"""
Code below from NASBench-201 and slightly adapted
@inproceedings{dong2020nasbench201,
  title     = {NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
"""


class ResNetBasicblock(AbstractPrimitive):
    def __init__(self, C_in, C_out, stride, affine=True):  # pylint:disable=W0613
        super().__init__(locals())
        assert stride == 1 or stride == 2, f"invalid stride {stride}"
        self.conv_a = ReLUConvBN(C_in, C_out, 3, stride)
        self.conv_b = ReLUConvBN(C_out, C_out, 3)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x, edge_data):  # pylint: disable=W0613
        basicblock = self.conv_a(x, None)
        basicblock = self.conv_b(basicblock, None)
        residual = self.downsample(x) if self.downsample is not None else x
        return residual + basicblock

    @staticmethod
    def get_embedded_ops():
        return None
