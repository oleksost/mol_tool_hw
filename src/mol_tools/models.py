import torch
from sklearn.linear_model import LinearRegression
from autosklearn.regression import AutoSklearnRegressor
from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

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
        self,
        memory_limit=24576,
        time_left_for_this_task=180,
        n_jobs=1,
        seed=1,
        **kwargs,
    ):
        super().__init__(
            memory_limit=memory_limit,
            time_left_for_this_task=time_left_for_this_task,
            n_jobs=n_jobs,
            seed=seed,
        )


@register_model("linear_regression")
class SkLearnLinearRegression(LinearRegression, Model):
    def __init__(self,  fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None, positive=False):
        super().__init__(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
                         n_jobs=n_jobs, positive=positive)


    def fit(self, X, y):
        super().fit(X, y)

    def predict(self, X):
        return super().predict(X)


@register_model("tree_regression")
class SkLearnTreeRegression(DecisionTreeRegressor, Model):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        super().fit(X, y)

    def predict(self, X):
        return super().predict(X)


@register_model("mlp_regression")
class SkLearnMLPRegression(MLPRegressor, Model):
    def __init__(self):
        super().__init__(
            hidden_layer_sizes=(3,), activation="relu", solver="lbfgs"
        )

    def fit(self, X, y):
        super().fit(X, y)

    def predict(self, X):
        return super().predict(X)
