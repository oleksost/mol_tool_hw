import os
import pickle
from sklearn.metrics import mean_absolute_error
from mol_tools.models import MODELS, Model
import logging
from mol_tools.data_utils import get_freesolv_data


def train(X_train, y_train, model="AutoSklearnRegressor", model_path="storage/"):
    """
    Train a model on freesolv dataset using the specified model and save it to the given model path.

    Args:
        model (str): The name of the model to use. Defaults to "AutoSklearnRegressor". See list on the models in mol_tools/models.py.
        model_path (str): The path where the trained model will be saved. Defaults to "storage/".

    Returns:
        None
    """
    model_path = os.path.join(model_path, f"_{model}.pkl")
    logging.info(f"Data loaded with {X_train.shape[0]} samples")
    assert model in MODELS, f"Model {model} not found in MODELS"
    model: Model = MODELS[model](random_state=0)
    logging.info(f"Training {model.__class__} model")

    model.fit(X_train, y_train)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Model saved at {model_path}")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X_train, X_test, y_train, y_test = get_freesolv_data()

    model = train(X_train, y_train, "mlp_regression")
    y_hat = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_hat)
    logging.info(f"Mean Absolute Test Error: {mae}")
