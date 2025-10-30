# --------------- Model Data (Joel Chaple) ---------------
# # ---------- Import Necessary Libraries ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snsfrom 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics

# ---------- Load and Preprocess Data ----------

def clean_load_data(path):
    data = pd.read_csv(path) # loads the data
    data = data.dropna() # makes sure there are no missing values
    return data

# ---------- Split Data into Training and Testing Sets ----------
# Temporary
clean_load_data(recruitment_data.csv)