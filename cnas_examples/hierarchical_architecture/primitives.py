from comprehensive_nas.search_spaces.graph_grammar.primitives import AbstractPrimitive


class OP(AbstractPrimitive):
    def __init__(self, op: int):
        super().__init__(locals())
        self.op = op

    def forward(self, x=None, edge_data=None):  # pylint: disable=unused-argument
        return self.op

    @staticmethod
    def get_embedded_ops():
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        return f"{op_name}{self.op}"
