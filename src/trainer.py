import datamol as dm
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
from molfeat.store.modelstore import ModelStore
import autosklearn.regression

smiles_column = "smiles"

def _preprocess(i, row):

    dm.disable_rdkit_log()

    mol = dm.to_mol(row[smiles_column], ordered=True)
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
    mol = dm.standardize_mol(
        mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True
    )

    row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
    row["selfies"] = dm.to_selfies(mol)
    row["inchi"] = dm.to_inchi(mol)
    row["inchikey"] = dm.to_inchikey(mol)
    return row


def get_data():
    df = dm.data.freesolv()
    # 1. clean the data
    data_clean = dm.parallelized(_preprocess, df.iterrows(), arg_type="args", progress=True, total=len(df))
    data_clean = pd.DataFrame(data_clean)
    X, y = data_clean["standard_smiles"], data_clean["expt"]
    # featurize
    calc = FPCalculator("ecfp")
    mol_transf = MoleculeTransformer(calc, n_jobs=-1)
    X_encoded = mol_transf(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=0)
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = get_data()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    automl = AutoSklearnRegressor(
        memory_limit=24576, 
        # For practicality’s sake, limit this to 5 minutes! 
        # (x3 = 15 min in total)
        time_left_for_this_task=180,  
        n_jobs=1,
        seed=1,
    )
    automl.fit(X_train, y_train)
    
    
    

if __name__ == "__main__":
    main()