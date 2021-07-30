class Optimizer:
    def __init__(self):
        pass

    def generate_new_pool(self):
        pass

    def evaluate(self):
        pass

    def initialize_model(self, **kwargs):
        raise NotImplementedError

    def update_model(self, **kwargs):
        raise NotImplementedError

    def propose_new_location(self, **kwargs):
        raise NotImplementedError
