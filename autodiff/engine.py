""" Core functionality during differentiation """


class GradEngine:
    __slots__ = {'_num_dep', '_acc', '_working'}

    def __init__(self):
        """
        The Engine to solve graph dependencies and accumulate gradients.

        Can be activated and deactivated. Working only during activated status
        """
        self._num_dep = 0
        self._acc = None
        self._working = False

    def reset(self):
        self._num_dep = 0
        self._acc = None
        self._working = False

    def activate(self):
        if self._working:
            raise RuntimeError("Cannot reactivate when backward engine is working.")
        self._working = True

    def is_working(self):
        return self._working

    def _check_working(self):
        if not self._working:
            raise RuntimeError("Cannot accumulate gradient or add dependency if engine is not alive.")

    def accumulate(self, grad):
        self._check_working()
        if self._acc is None:
            self._acc = 0
        self._acc += grad

    def add_dependency(self):
        self._num_dep += 1

    def del_dependency(self):
        if self._num_dep <= 0:
            raise ValueError("Cannot call del_dependency since engine has already no dependencies.")
        self._num_dep -= 1

    def zero_dependency(self):
        return self._num_dep == 0

    def ready_backward(self):
        return self._working and self._num_dep == 0

    def get_grad(self):
        return self._acc
