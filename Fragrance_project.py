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


st.markdown("# Draw your molecule")
st.markdown("Draw your molecule and click on the button below to get the SMILES code")
smile_code = st_ketcher()
st.markdown(f"Smile code: ``{smile_code}``")

