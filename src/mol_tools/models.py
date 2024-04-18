import torch
from autosklearn.regression import AutoSklearnRegressor
from abc import ABC, abstractmethod

MODELS = {}


def register_model(name):
    def wrapper(cls):
        if name in MODELS:
            raise ValueError(f"Model {name} already exists")
        MODELS[name] = cls
        return cls

    return wrapper


class Model(ABC):
    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs):
        pass


@register_model("AutoSklearnRegressor")
class AutoSklearnRegressorModel(AutoSklearnRegressor, Model):
    def __init__(
        self, memory_limit=24576, time_left_for_this_task=180, n_jobs=1, seed=1, **kwargs
    ):
        super().__init__(
            memory_limit=memory_limit,
            time_left_for_this_task=time_left_for_this_task,
            n_jobs=n_jobs,
            seed=seed,
        )


