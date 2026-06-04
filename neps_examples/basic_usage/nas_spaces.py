from functools import partial
import itertools

import neps
import torch.nn as nn
import numpy as np




class Mul(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor

class ReLUConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine, track_running_stats=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not affine)
        self.bn = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class Normalization(nn.Module):
    def __init__(self, num_features, norm_type, affine=True):
        super().__init__()
        self.affine = affine
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(num_features, affine=affine)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(num_features, affine=affine)
        elif norm_type == "layer":
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def forward(self, x):
        if self.norm is None:
            self.norm = nn.LayerNorm(x.size()[1:], elementwise_affine=self.affine)
            if x.is_cuda:
                self.norm = self.norm.cuda()
        return self.norm(x)

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.conv(x)

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, affine=True, track_running_stats=True):
        super().__init__()
        self.conv1 = ReLUConvBN(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, affine=affine, track_running_stats=track_running_stats)
        self.conv2 = ReLUConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, affine=affine, track_running_stats=track_running_stats)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(2, stride=2, padding=0),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            )
        elif in_channels != out_channels:
            self.downsample = ReLUConvBN(in_channels, out_channels, kernel_size=1, stride=1, padding=0, affine=affine, track_running_stats=track_running_stats)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity

class Activation(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        if act_type == "relu":
            self.act = nn.ReLU(inplace=False)
        elif act_type == "hardswish":
            self.act = nn.Hardswish(inplace=False)
        elif act_type == "mish":
            self.act = nn.Mish(inplace=False)
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

    def forward(self, x):
        return self.act(x)


class Residual2(nn.Module):
    def __init__(self, op1, op2, op3):
        super().__init__()
        self.op1 = op1
        self.op2 = op2
        self.op3 = op3
    def forward(self, x):
        out = self.op1(x)
        out = self.op3(out)
        return self.op2(x) + out

class Residual3(nn.Module):
    def __init__(self, op1, op2, op3, op4):
        super().__init__()
        self.op1 = op1
        self.op2 = op2
        self.op3 = op3
        self.op4 = op4
    def forward(self, x):
        out = self.op1(x)
        out = self.op2(out)
        out = self.op4(out)
        return self.op3(x) + out

class NASBench201Cell(nn.Module):
    def __init__(self, op1, op2, op3, op4, op5, op6):
        super().__init__()
        # Operations corresponding to the edge_list:
        self.op_1to2 = op1
        self.op_1to3 = op2
        self.op_2to3 = op3
        self.op_1to4 = op4
        self.op_2to4 = op5
        self.op_3to4 = op6

    def forward(self, x):
        # Node 1 is the input: x
        # Node 2: only has one incoming edge from Node 1
        node2 = self.op_1to2(x)
        # Node 3: sum of edges from Node 1 and Node 2
        node3 = self.op_1to3(x) + self.op_2to3(node2)
        # Node 4 (Output): sum of edges from Node 1, Node 2, and Node 3
        node4 = self.op_1to4(x) + self.op_2to4(node2) + self.op_3to4(node3)
        return node4


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, out_channels)
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.pool(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out

class HNASSpace(neps.PipelineSpace):
    def __init__(self, fidelities=None, in_channels=3, out_channels=10, base_channels=16, channel_factor=2, lr=False, wd=False):
        self._in_channels = in_channels
        self._start_channels = base_channels
        self._early_channels = base_channels * channel_factor
        self._late_channels = base_channels * (channel_factor ** 2)
        self._out_channels = out_channels
        
        if fidelities is not None:
            self.fidelity = neps.IntegerFidelity(lower=fidelities[0], upper=fidelities[1])

        if lr:
            self.learning_rate = neps.Float(lower=0.001, upper=0.3, log=True)

        if wd:
            self.weight_decay = neps.Float(lower=1e-4, upper=1e-2, log=True)

        self._ACT = neps.Categorical(
            choices=(
                Activation(act_type="relu"),
                Activation(act_type="hardswish"),
                Activation(act_type="mish"),
            ),
        )
        
        self._CONV_A = neps.Categorical(
            choices=(
                ReLUConvBN(in_channels=self._start_channels, out_channels=self._start_channels, kernel_size=1, stride=1, padding=0, affine=True),
                ReLUConvBN(in_channels=self._start_channels, out_channels=self._start_channels, kernel_size=3, stride=1, padding=1, affine=True),
                DepthwiseConv(in_channels=self._start_channels, out_channels=self._start_channels, kernel_size=3),
            ),
        )

        self._CONV_B = neps.Categorical(
            choices=(
                ReLUConvBN(in_channels=self._early_channels, out_channels=self._early_channels, kernel_size=1, stride=1, padding=0, affine=True),
                ReLUConvBN(in_channels=self._early_channels, out_channels=self._early_channels, kernel_size=3, stride=1, padding=1, affine=True),
                DepthwiseConv(in_channels=self._early_channels, out_channels=self._early_channels, kernel_size=3),
            ),
        )

        self._CONV_C = neps.Categorical(
            choices=(
                ReLUConvBN(in_channels=self._late_channels, out_channels=self._late_channels, kernel_size=1, stride=1, padding=0, affine=True),
                ReLUConvBN(in_channels=self._late_channels, out_channels=self._late_channels, kernel_size=3, stride=1, padding=1, affine=True),
                DepthwiseConv(in_channels=self._late_channels, out_channels=self._late_channels, kernel_size=3),
            ),
        )

        self._NORM_A = neps.Categorical(
            choices=(
                Normalization(num_features=self._start_channels, norm_type="batch"),
                Normalization(num_features=self._start_channels, norm_type="instance"),
                Normalization(num_features=self._start_channels, norm_type="layer"),
            ),
        )
        
        self._NORM_B = neps.Categorical(
            choices=(
                Normalization(num_features=self._early_channels, norm_type="batch"),
                Normalization(num_features=self._early_channels, norm_type="instance"),
                Normalization(num_features=self._early_channels, norm_type="layer"),
            ),
        )
        
        self._NORM_C = neps.Categorical(
            choices=(
                Normalization(num_features=self._late_channels, norm_type="batch"),
                Normalization(num_features=self._late_channels, norm_type="instance"),
                Normalization(num_features=self._late_channels, norm_type="layer"),
            ),
        )

        self._CONVBLOCK_A = neps.Operation(
            operator=self.sequentializer,
            args=(
                self._ACT.resample(),
                self._CONV_A.resample(),
                self._NORM_A.resample(),
            ),
        )

        self._CONVBLOCK_B = neps.Operation(
            operator=self.sequentializer,
            args=(
                self._ACT.resample(),
                self._CONV_B.resample(),
                self._NORM_B.resample(),
            ),
        )

        self._CONVBLOCK_C = neps.Operation(
            operator=self.sequentializer,
            args=(
                self._ACT.resample(),
                self._CONV_C.resample(),
                self._NORM_C.resample(),
            ),
        )

        self._OP_options = ["zero", "id", "convblock", "avgpool"]
        self._ALL_OPS = itertools.product(self._OP_options, repeat=6)
        self._VALID_OPS = []
        for ops in self._ALL_OPS:
            if ops[3] != "zero" or not "zero" in [ops[0], ops[4]] or not "zero" in [ops[1], ops[5]] or not "zero" in [ops[0], ops[2], ops[5]]:
                self._VALID_OPS.append(ops)

        self._CL_A = neps.Operation(
            operator=self.cell_builder,
            args=[
                neps.Categorical(choices=tuple(self._VALID_OPS)),
                self._CONVBLOCK_A.resample(),
            ],
        )

        self._CL_B = neps.Operation(
            operator=self.cell_builder,
            args=[
                neps.Categorical(choices=tuple(self._VALID_OPS)),
                self._CONVBLOCK_B.resample(),
            ],
        )

        self._CL_C = neps.Operation(
            operator=self.cell_builder,
            args=[
                neps.Categorical(choices=tuple(self._VALID_OPS)),
                self._CONVBLOCK_C.resample(),
            ],
        )

        self._C_A = neps.Categorical(
            choices=(
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._CL_A.resample(),
                        self._CL_A.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._CL_A.resample(),
                        self._CL_A.resample(),
                        self._CL_A.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.residualizer,
                    kwargs={"type": 2,
                        "op1": self._CL_A.resample(),
                        "op2": self._CL_A.resample(),
                        "op3": self._CL_A.resample(),
                    },
                ),
            ),
        )

        self._C_B = neps.Categorical(
            choices=(
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._CL_B.resample(),
                        self._CL_B.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._CL_B.resample(),
                        self._CL_B.resample(),
                        self._CL_B.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.residualizer,
                    kwargs={"type": 2,
                        "op1": self._CL_B.resample(),
                        "op2": self._CL_B.resample(),
                        "op3": self._CL_B.resample(),
                    },
                ),
            ),
        )

        self._C_C = neps.Categorical(
            choices=(
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._CL_C.resample(),
                        self._CL_C.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._CL_C.resample(),
                        self._CL_C.resample(),
                        self._CL_C.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.residualizer,
                    kwargs={"type": 2,
                        "op1": self._CL_C.resample(),
                        "op2": self._CL_C.resample(),
                        "op3": self._CL_C.resample(),
                    },
                ),
            ),
        )

        self._DOWN_AB = ResNetBasicBlock(in_channels=self._start_channels, out_channels=self._early_channels, stride=2)

        self._DOWN_BC = ResNetBasicBlock(in_channels=self._early_channels, out_channels=self._late_channels, stride=2)

        self._D_AB = neps.Categorical(
            choices=(
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._CL_A.resample(),
                        self._DOWN_AB,
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._CL_A.resample(),
                        self._CL_A.resample(),
                        self._DOWN_AB,
                    ),
                ),
                neps.Operation(
                    operator=self.residualizer,
                    kwargs={"type": 2,
                        "op1": self._CL_A.resample(),
                        "op2": self._DOWN_AB,
                        "op3": self._DOWN_AB,
                    },
                ),
            ),
        )

        self._D_BC = neps.Categorical(
            choices=(
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._CL_B.resample(),
                        self._DOWN_BC,
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._CL_B.resample(),
                        self._CL_B.resample(),
                        self._DOWN_BC,
                    ),
                ),
                neps.Operation(
                    operator=self.residualizer,
                    kwargs={"type": 2,
                        "op1": self._CL_B.resample(),
                        "op2": self._DOWN_BC,
                        "op3": self._DOWN_BC,
                    },
                ),
            ),
        )

        self._D0_A = neps.Categorical(
            choices=(
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._C_A.resample(),
                        self._C_A.resample(),
                        self._CL_A.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._C_A.resample(),
                        self._C_A.resample(),
                        self._C_A.resample(),
                        self._CL_A.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._C_A.resample(),
                        self._C_A.resample(),
                        self._CL_A.resample(),
                        self._CL_A.resample(),
                    ),
                ),
            ),
        )
        
        self._D0_C = neps.Categorical(
            choices=(
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._C_C.resample(),
                        self._C_C.resample(),
                        self._CL_C.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._C_C.resample(),
                        self._C_C.resample(),
                        self._C_C.resample(),
                        self._CL_C.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._C_C.resample(),
                        self._C_C.resample(),
                        self._CL_C.resample(),
                        self._CL_C.resample(),
                    ),
                ),
            ),
        )
        
        self._D1_AB = neps.Categorical(
            choices=(
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._C_A.resample(),
                        self._C_A.resample(),
                        self._D_AB.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._C_A.resample(),
                        self._C_A.resample(),
                        self._C_A.resample(),
                        self._D_AB.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.residualizer,
                    kwargs={"type": 3,
                            "op1": self._C_A.resample(),
                            "op2": self._C_A.resample(),
                            "op3": self._D_AB.resample(),
                            "op4": self._D_AB.resample(),
                    },
                ),
            ),
        )

        self._D1_BC = neps.Categorical(
            choices=(
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._C_B.resample(),
                        self._C_B.resample(),
                        self._D_BC.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._C_B.resample(),
                        self._C_B.resample(),
                        self._C_B.resample(),
                        self._D_BC.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.residualizer,
                    kwargs={"type": 3,
                            "op1": self._C_B.resample(),
                            "op2": self._C_B.resample(),
                            "op3": self._D_BC.resample(),
                            "op4": self._D_BC.resample(),
                    },
                ),
            ),
        )

        self._D2 = neps.Categorical(
            choices=(
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._D1_AB.resample(),
                        self._D1_BC.resample(),
                        self._D0_C.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._D0_A.resample(),
                        self._D1_AB.resample(),
                        self._D1_BC.resample(),
                    ),
                ),
                neps.Operation(
                    operator=self.sequentializer,
                    args=(
                        self._D1_AB.resample(),
                        self._D1_BC.resample(),
                        self._D0_C.resample(),
                        self._D0_C.resample(),
                    ),
                ),
            ),
        )

        self.architecture = neps.Operation(
            operator=self.sequentializer,
            args=(
                Stem(in_channels=self._in_channels, out_channels=self._start_channels),
                self._D2,
                Head(in_channels=self._late_channels, out_channels=self._out_channels),
            ),
        )


    @staticmethod
    def sequentializer(*args):
        return nn.Sequential(*args)
    
    @staticmethod
    def cell_builder(ops, convblock):
        op_dict = {
            "zero": Mul(factor=0),
            "id": nn.Identity(),
            "convblock": convblock,
            "avgpool": nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
        }
        return NASBench201Cell(op_dict[ops[0]], op_dict[ops[1]], op_dict[ops[2]], op_dict[ops[3]], op_dict[ops[4]], op_dict[ops[5]])

    @staticmethod
    def residualizer(type:int, op1, op2, op3, op4=None):
        if type == 2:
            return Residual2(op1, op2, op3)
        elif type == 3:
            return Residual3(op1, op2, op3, op4)
        else:
            raise ValueError(f"Unsupported residual type: {type}")

space_dict = {
    "Base_CFG": HNASSpace,
    "Base_CFG_LR_WD": partial(HNASSpace, lr=True, wd=True),
}
