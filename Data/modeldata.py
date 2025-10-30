# --------------- Model Data (Joel Chaple) ---------------
# # ---------- Import Necessary Libraries ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import accuracy_score

# ---------- Load and Preprocess Data ----------

def clean_load_data(path):
    """
    Loads and cleans the recruitment data from kaggle
    Data Source:https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data/data
    Input:
    path = path to the data file (string)
    Returns:
    data = cleaned pandas dataframe 
    Column information about the data:
    Age = Age of the candidate, from 20 to 50 (int)
    Gender = Gender of the candidate, 0 for Male, 1 for Female (int)
    Education Level = Education level of the canditate, 1 for Bachelors(Type1), 2 for Bachelors(Type2), 
    3 for Masters, 4 for PhD (int)
    Experience Years = Years of experience of the candidate, from 0 to 15 (int)
    Previous Companies Worked = Number of previous companies the candidate has worked for, from 0 to 5 (int)
    Distance from  Company = Distance in kilometers from the candidate's residence to the hiring company, from 1 to 50 (int)
    Interview Score = Score obtained by the candidate in the interview, from 1 to 100 (int)
    Skill Score = Score obtained by the candidate in the skill test, from 1 to 100 (int)
    Personality Score = Score obtained by the candidate in the personality test, from 1 to 100 (int)
    Recuitment Strategy = Strategy used for recruitment, 1 for Aggresive, 2 for Moderate, 3 for Conservative (int)
    Hiring Decision = 0 for Not Hired, 1 for Hired (int)
    Extra Information:
    1500 entries
    10 features
    """
    data = pd.read_csv(path) # loads the data
    data = data.dropna() # makes sure there are no missing values
    print(data) # prints the cleaned data
    return data

# ---------- Split Data into Training and Testing Sets ----------
def split_data(data, test_size=0.2, random_state=42):
    """
    Splits the recruitment data into training and testing sets
    Inputs:
    data = cleaned pandas dataframe
    test_size = amount of the data to be used for testing (float)
    random_state = random seed (int)
    Returns:
    X_train, X_test, y_train, y_test = training and testing sets 
    """
    X = data.drop('HiringDecision', axis=1) # feature(s)
    y = data['HiringDecision'] # target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# ---------- Temporary Test Code ----------
data = clean_load_data('Data/recruitment_data.csv')
X_train, X_test, y_train, y_test = split_data(data)
logit_model = sm.Logit(y_train, sm.add_constant(X_train))
result = logit_model.fit()
print(result.summary())

accuracy = accuracy_score(y_test, result.predict(sm.add_constant(X_test)).round())
print(f'The Linear Regression Model has accuracy: {accuracy} %')