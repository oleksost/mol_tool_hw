import os
import datamol as dm
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
from autosklearn.regression import AutoSklearnRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score

smiles_column = "smiles"


def _preprocess(row):

    dm.disable_rdkit_log()

    mol = dm.to_mol(row[smiles_column], ordered=True)
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
    mol = dm.standardize_mol(
        mol,
        disconnect_metals=False,
        normalize=True,
        reionize=True,
        uncharge=False,
        stereo=True,
    )

    row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
    row["selfies"] = dm.to_selfies(mol)
    row["inchi"] = dm.to_inchi(mol)
    row["inchikey"] = dm.to_inchikey(mol)
    return row


def get_data(dest="storage/_freesolv_encoded.csv"):
    if os.path.exists(dest):
        data = pd.read_csv(dest)
        X_encoded, y = data.drop("expt", axis=1), data["expt"]
    else:
        df = dm.data.freesolv()
        # 1. clean the data
        data_clean = df.apply(_preprocess, axis=1)
        data_clean = pd.DataFrame(data_clean)
        X, y = data_clean["standard_smiles"], data_clean["expt"]
        # featurize
        calc = FPCalculator("ecfp")
        mol_transf = MoleculeTransformer(calc, n_jobs=-1)
        X_encoded = mol_transf(X)
        X_encoded = pd.DataFrame(X_encoded)
        # save on disk
        pd.concat([X_encoded, y], axis=1).to_csv(dest, index=False)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=0)
    return X_train, X_test, y_train, y_test


def main(model_path="storage/_autosklearn_model.pkl"):
    X_train, X_test, y_train, y_test = get_data()
    print(
        "train sahpes:",
        X_train.shape,
        y_train.shape,
        "\n Test shapes:",
        X_test.shape,
        y_test.shape,
    )

    automl = AutoSklearnRegressor(
        memory_limit=24576,
        # For practicality’s sake, limit this to 5 minutes!
        # (x3 = 15 min in total)
        time_left_for_this_task=180,
        n_jobs=1,
        seed=1,
    )
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    mae = mean_absolute_error(y_test, y_hat)
    print("MAE:", mae)
    # save the model with pickle
    import pickle

    with open(model_path, "wb") as f:
        pickle.dump(automl, f)
    print("Model saved at:", model_path)


if __name__ == "__main__":
    main()
