from rdkit.Chem import MolFromSmiles, Descriptors, MolFromSmarts
import pandas as pd
from pycaret.regression import load_model


def create_esol_descriptors(smiles: str):

    mol = MolFromSmiles(smiles)

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    arom_proportion = len(mol.GetSubstructMatches(MolFromSmarts("a"))) / Descriptors.HeavyAtomCount(mol)

    # other descriptors in Delaney's publication
    # hbd = Descriptors.NHOHCount(mol)
    # hba = Descriptors.NOCount(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    non_carbon_proportion = len(mol.GetSubstructMatches(MolFromSmarts("[!#6]"))) / Descriptors.HeavyAtomCount(mol)
    psa = Descriptors.TPSA(mol, includeSandP=True)
    # plus some extra one
    fsp3 = Descriptors.FractionCSP3(mol)

    return mw, logp, rotb, arom_proportion, hbd, hba, non_carbon_proportion, psa, fsp3


def run_prediction(smiles: str):

    features = [create_esol_descriptors(smiles)]
    col_names = ['mw', 'logp', 'rotb', 'ap', 'hbd', 'hba', 'non_cp', 'psa', 'fsp3']
    df = pd.DataFrame(features, columns=col_names)

    model = load_model('sol_pred_model_delaney')
    prediction = model.predict(df)

    return prediction


if __name__ == '__main__':

    smiles = 'CC(=O)OC1=CC=CC=C1C(O)=O'
    prediction = run_prediction(smiles)
    print(prediction)
