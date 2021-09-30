class Optimizer:
    def new_result(self, job):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError
