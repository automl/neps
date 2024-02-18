from abc import ABCMeta, abstractmethod

import torch
from torch import nn


class _AbstractPrimitive(nn.Module, metaclass=ABCMeta):
    """
    Use this class when creating new operations for edges.

    This is required because we are agnostic to operations
    at the edges. As a consequence, they can contain subgraphs
    which requires naslib to detect and properly process them.
    """

    def __init__(self, kwargs):
        super().__init__()

        self.init_params = {
            k: v
            for k, v in kwargs.items()
            if k != "self" and not k.startswith("_") and k != "kwargs"
        }

    @abstractmethod
    def forward(self, x):
        """
        The forward processing of the operation.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embedded_ops(self):
        """
        Return any embedded ops so that they can be
        analysed whether they contain a child graph, e.g.
        a 'motif' in the hierachical search space.

        If there are no embedded ops, then simply return
        `None`. Should return a list otherwise.
        """
        raise NotImplementedError

    @property
    def get_op_name(self):
        return type(self).__name__


class AbstractPrimitive(_AbstractPrimitive):
    def forward(self, x):
        raise NotImplementedError

    def get_embedded_ops(self):
        return None


class Identity(AbstractPrimitive):
    """
    An implementation of the Identity operation.
    """

    def __init__(self, **kwargs):
        super().__init__(locals())

    def forward(self, x):
        return x


class Zero(AbstractPrimitive):
    """
    Implementation of the zero operation. It removes
    the connection by multiplying its input with zero.
    """

    def __init__(self, stride, **kwargs):
        """
        When setting stride > 1 then it is assumed that the
        channels must be doubled.
        """
        super().__init__(locals())
        self.stride = int(stride)

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        else:
            return x[:, :, :: self.stride, :: self.stride].mul(0.0)

    def __repr__(self):
        return f"<Zero (stride={self.stride})>"


class Zero1x1(AbstractPrimitive):
    """
    Implementation of the zero operation. It removes
    the connection by multiplying its input with zero.
    """

    def __init__(self, stride, **kwargs):
        """
        When setting stride > 1 then it is assumed that the
        channels must be doubled.
        """
        super().__init__(locals())
        self.stride = int(stride)

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        else:
            x = x[:, :, :: self.stride, :: self.stride].mul(0.0)
            return torch.cat([x, x], dim=1)  # double the channels TODO: ugly as hell

    def __repr__(self):
        return f"<Zero1x1 (stride={self.stride})>"


class SepConv(AbstractPrimitive):
    """
    Implementation of Separable convolution operation as
    in the DARTS paper, i.e. 2 sepconv directly after another.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, **kwargs):
        super().__init__(locals())

        C_in = int(C_in)
        C_out = int(C_out)
        kernel_size = int(kernel_size)
        stride = int(stride)
        padding = int(padding)
        affine = bool(affine)

        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += f"{self.kernel_size}x{self.kernel_size}"
        return op_name


class DilConv(AbstractPrimitive):
    """
    Implementation of a dilated separable convolution as
    used in the DARTS paper.
    """

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, **kwargs
    ):
        super().__init__(locals())

        C_in = int(C_in)
        C_out = int(C_out)
        kernel_size = int(kernel_size)
        stride = int(stride)
        padding = int(padding)
        dilation = int(dilation)
        affine = bool(affine)

        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += f"{self.kernel_size}x{self.kernel_size}"
        return op_name


class Stem(AbstractPrimitive):
    """
    This is used as an initial layer directly after the
    image input.
    """

    def __init__(self, C_out, C_in=3, **kwargs):
        super().__init__(locals())

        C_out = int(C_out)

        self.seq = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False), nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.seq(x)


class Sequential(AbstractPrimitive):
    """
    Implementation of `torch.nn.Sequential` to be used
    as op on edges.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(locals())
        self.primitives = args
        self.op = nn.Sequential(*args)

    def forward(self, x):
        return self.op(x)

    def get_embedded_ops(self):
        return list(self.primitives)


class MaxPool(AbstractPrimitive):
    def __init__(self, kernel_size, stride, **kwargs):
        super().__init__(locals())

        kernel_size = int(kernel_size)
        stride = int(stride)

        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=1)

    def forward(self, x):
        x = self.maxpool(x)
        return x


class MaxPool1x1(AbstractPrimitive):
    """
    Implementation of MaxPool with an optional 1x1 convolution
    in case stride > 1. The 1x1 convolution is required to increase
    the number of channels.
    """

    def __init__(self, kernel_size, stride, C_in, C_out, affine=True, **kwargs):
        super().__init__(locals())

        kernel_size = int(kernel_size)
        stride = int(stride)
        C_in = int(C_in)
        C_out = int(C_out)
        affine = bool(affine)

        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=1)
        if stride > 1:
            assert C_in is not None and C_out is not None
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.maxpool(x)
        if self.stride > 1:
            x = self.conv(x)
            x = self.bn(x)
        return x


class AvgPool(AbstractPrimitive):
    """
    Implementation of Avergae Pooling.
    """

    def __init__(self, kernel_size, stride, **kwargs):  # pylint: disable=W0613
        stride = int(stride)
        super().__init__(locals())
        self.avgpool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    def forward(self, x):
        x = self.avgpool(x)
        return x


class AvgPool1x1(AbstractPrimitive):
    """
    Implementation of Avergae Pooling with an optional
    1x1 convolution afterwards. The convolution is required
    to increase the number of channels if stride > 1.
    """

    def __init__(
        self,
        kernel_size,  # pylint: disable=W0613
        stride,
        C_in,
        C_out,
        affine=True,
        **kwargs,  # pylint: disable=W0613
    ):
        super().__init__(locals())
        stride = int(stride)
        self.stride = int(stride)
        self.avgpool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        if stride > 1:
            assert C_in is not None and C_out is not None
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.avgpool(x)
        if self.stride > 1:
            x = self.conv(x)
            x = self.bn(x)
        return x


class ReLUConvBN(AbstractPrimitive):
    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, **kwargs):
        super().__init__(locals())
        kernel_size = int(kernel_size)
        stride = int(stride)

        self.kernel_size = kernel_size
        pad = 0 if int(stride) == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += f"{self.kernel_size}x{self.kernel_size}"
        return op_name


class ConvBnReLU(AbstractPrimitive):
    """
    Implementation of 2d convolution, followed by 2d batch normalization and ReLU activation.
    """

    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += f"{self.kernel_size}x{self.kernel_size}"
        return op_name


class ConvBn(AbstractPrimitive):
    """
    Implementation of 2d convolution, followed by 2d batch normalization and ReLU activation.
    """

    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += f"{self.kernel_size}x{self.kernel_size}"
        return op_name


class Concat1x1(AbstractPrimitive):
    """
    Implementation of the channel-wise concatination followed by a 1x1 convolution
    to retain the channel dimension.
    """

    def __init__(
        self, num_in_edges, C_out, affine=True, **kwargs  # pylint: disable=W0613
    ):
        super().__init__(locals())
        self.conv = nn.Conv2d(
            num_in_edges * C_out, C_out, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        """
        Expecting a list of input tensors. Stacking them channel-wise
        and applying 1x1 conv
        """
        x = torch.cat(x, dim=1)
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(AbstractPrimitive):
    def __init__(
        self, C_in, C_out, stride, affine=True, **kwargs
    ):  # pylint:disable=W0613
        super().__init__(locals())
        assert stride == 1 or stride == 2, f"invalid stride {stride}"
        self.conv_a = ReLUConvBN(C_in, C_out, 3, stride)
        self.conv_b = ReLUConvBN(C_out, C_out, 3)
        if stride == 2:
            self.downsample = nn.Sequential(
                # nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(C_out),
            )
        else:
            self.downsample = None

    def forward(self, x):
        basicblock = self.conv_a(x)
        basicblock = self.conv_b(basicblock)
        residual = self.downsample(x) if self.downsample is not None else x
        return residual + basicblock
