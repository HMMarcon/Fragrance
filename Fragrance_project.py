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
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits = 2048)
    arr = np.zeros((1,), int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


import os
import pickle

#@st.cache_resource
def load_pickles_from_folder():
    """Load all pickle files from a given folder into a list."""
    # List all files in the folder
    files = [f for f in os.listdir()]
    st.write(files)
    # Filter for .pickle files
    pickle_files = [f for f in files if f.endswith('.pkl')]

    data_list = []

    # Load each pickle file and append its content to data_list
    for pkl_file in pickle_files:
        with open(pkl_file, 'rb') as file:
            data = pickle.load(file)
            data_list.append(data)

    return data_list

# Example usage
# folder_path = '/path/to/your/folder'

all_pickles = load_pickles_from_folder()
st.write(all_pickles)

st.markdown("# Draw your molecule")
st.markdown("Draw your molecule and click on the button below to get the SMILES code")
smile_code = st_ketcher()
#st.markdown(f"Smile code: ``{smile_code}``")
#smiles = "'" + smile_code + "'"

#compound = Chem.MolFromSmiles(smiles_code)
compound_FP = smiles_to_fp([smile_code])

if compound_FP[0] is None:
    st.markdown("Invalid SMILES or unable to generate fingerprint.")

else:
    st.markdown(compound_FP[0])

