import streamlit as st

# Data loading and management
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


# Basic chemistry packages
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem import Descriptors

# Non-standard metrics
from sklearn.metrics import matthews_corrcoef as MCC

from xgboost import XGBClassifier

from streamlit_ketcher import st_ketcher


# Fingerprints functions
fp_size = [2048]
fp_radius = [2]


def smiles_to_fp(smiles_list):
    fp_list = []
    for smile in smiles_list:
        fp_list.append(fp_as_array(smile))
    return fp_list


def fp_as_array(smile):
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius[0], nBits=fp_size[0])
    arr = np.zeros((1,), int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
'''
## Loading models
@st.cache_resource
def load_models():
    return pickle.load(open("model_fp2fp_Ridge_7.5.mlpickle", "rb"))


fp2fp_model = load_models()
'''

st.markdown("# Draw your molecule")
st.markdown("Draw your molecule and click on the button below to get the SMILES code")
smile_code = st_ketcher()
#st.markdown(f"Smile code: ``{smile_code}``")
smiles = "'" + smile_code + "'"
st.text(smiles)

#compound = Chem.MolFromSmiles(smiles_code)
compound_FP = smiles_to_fp(smiles)

st.markdown(compound_FP)


