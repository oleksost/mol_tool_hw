import os
import datamol as dm
import pandas as pd
from sklearn.model_selection import train_test_split
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
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


def get_freesolv_data(dest="storage/_freesolv_encoded.csv"):
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