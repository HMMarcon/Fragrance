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
import xgboost as xgb
from xgboost import XGBClassifier

from streamlit_ketcher import st_ketcher

from joblib import load


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

@st.cache_resource
def load_pickles_from_folder(subfolder = 'models'):
    """Load all pickle files from a given folder into a list."""
    # List all files in the folder
    folder_path = os.path.join(os.getcwd(), subfolder)
    files = [f for f in os.listdir(folder_path)]

    # Filter for .pickle files
    pickle_files = sorted([f for f in files if f.endswith('.pkl')])

    #st.write(pickle_files)
    models_list = []

    # Load each pickle file and append its content to data_list
    for pkl_file in pickle_files:
        st.write('loading')
        with open(pkl_file, 'rb') as file:
            model = xgb.XGBClassifier()
            model.load_model(pkl_file)
            st.write('model loaded')
            #data = load(file)
            models_list.append((pkl_file[:-4], model))
            #data_list.append(model)

    return models_list


all_models = load_pickles_from_folder()
#st.write(len(all_models))

st.markdown("# From Molecule to smell")
st.markdown("Draw your molecule and click on the button below to find out how it might smell like.")
st.markdown("Using cutting-edge artifical intelligence technology we are capable of predicting how molecules smell. Try it out and let us know how it performs for you!")
smile_code = st_ketcher()
#st.markdown(f"Smile code: ``{smile_code}``")
#smiles = "'" + smile_code + "'"

#compound = Chem.MolFromSmiles(smiles_code)
compound_FP = smiles_to_fp([smile_code])

if compound_FP[0] is None:
    st.markdown("Invalid SMILES or unable to generate fingerprint.")

else:
    #st.markdown(compound_FP[0])
    #smells = []
    #for model in all_models:
        #st.write(model)
        #st.write(compound_FP)
        #smell = model.predict(compound_FP)
        #st.write(smell)
        #smells.append(smell)

    positive_models = []  # List to store names of models that predict 1

    for model_name, model_instance in all_models:
        prediction = model_instance.predict(compound_FP)
        if prediction == 1:
            positive_models.append(model_name)

    output = ""
    for positive_model in positive_models:
        output = output + positive_model
        if positive_model != positive_models[-1]:
            output = output + + ", "
        
    # Display the models that predicted 1
    st.write("Molecule smells like:", output)



