from __future__ import annotations

from abc import abstractmethod


class Kernel:
    def __init__(
        self,
        active_hps: None | list = None,
        kernel_kwargs=None,
    ):
        self.active_hps = active_hps
        self.kernel_kwargs = kernel_kwargs or {}

    @abstractmethod
    def does_apply_on(self, hp):
        raise NotImplementedError

    @abstractmethod
    def build(self, hp_shapes):
        raise Exception()

    def assign_hyperparameters(self, hyperparameters):
        if self.active_hps is None:
            self.active_hps = [
                hp_name
                for hp_name, hp in hyperparameters.items()
                if self.does_apply_on(hp)
            ]
        if not self.active_hps:
            raise Exception("Can't build a kernel without hyperparameters to apply on")
        return set(self.active_hps)

    @staticmethod
    def get_active_dims(hp_shapes):
        active_dims = []
        for shape in hp_shapes.values():
            active_dims.extend(shape.active_dims)
        return tuple(active_dims)

    @staticmethod
    def get_tensor_length(hp_shapes):
        return sum(shape.length for shape in hp_shapes.values())
