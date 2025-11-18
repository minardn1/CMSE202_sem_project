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

def clean_data(path):
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
    data = clean_data('Data/recruitment_data.csv') # loads and cleans the data
    X = data.drop('HiringDecision', axis=1) # feature(s)
    y = data['HiringDecision'] # target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# ---------- Test Data ----------

def test_data():
    """
    Briefly tests the data by fitting it into a logistic regression model and evaluating the accuracy score
    Purpose:
    To make sure that the data is properly cleaned and can be used for models
    To check if the features have a significant relationship with the target variable (Hiring Decision)
    """
    data = clean_data('Data/recruitment_data.csv')
    X_train, X_test, y_train, y_test = split_data(data)
    logit_model = sm.Logit(y_train, sm.add_constant(X_train))
    result = logit_model.fit()
    print(result.summary())

    accuracy = accuracy_score(y_test, result.predict(sm.add_constant(X_test)).round())
    print(f'The Linear Regression Model has accuracy: {accuracy}')

# ---------- Data Visualization ----------

def see_data(data):
    """
    Visualizes data to better understand the nature and distribution of the features and target variable
    Purpose:
    Identify patterns and relationships between features and target variable(s)
    Potentially see if data is linearly separable for model selection
    Input:
    data = cleaned pandas dataframe
    Returns:
    None (plots)
    """
    data = clean_data('Data/recruitment_data.csv') # loads and cleans the data
    data.head() # returns head of data
    data.describe() # describes data
    data.info() # info about data
    
    # Identify correlations between features and target variable
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='magma')
    plt.title('Correlation Heatmap of Recruitment Data')
    plt.show()

    # Distribution of target variable (Hiring Decision)
    plt.figure(figsize=(10, 8))
    sns.countplot(x='HiringDecision', data=data, color='crimson')
    plt.title('Distribution of Hiring Decisions')
    plt.xlabel('Hiring Decision (0 = Not Hired, 1 = Hired)')
    plt.ylabel('Count')
    plt.show()

    # Distribution of Age
    plt.figure(figsize=(10, 8))
    sns.histplot(data['Age'], bins=30, kde=True, color='blue')
    plt.title('Applicant Age Distribution')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.show()

    # Distribution of Experience Years
    plt.figure(figsize=(10, 8))
    sns.histplot(data['ExperienceYears'], bins=30, kde=True, color='blue')
    plt.title('Applicant Experience Level Distribution')
    plt.xlabel('Experience Years')
    plt.ylabel('Count')
    plt.show()

    # Hiring Decision by Education Level
    plt.figure(figsize=(10, 8))
    sns.countplot(x='EducationLevel', hue='HiringDecision', data=data)
    plt.title('Hiring Decision by Education Level')
    plt.xlabel('Education Level (1 = Bachelors(Type1), 2 = Bachelors(Type2), 3 = Masters, 4 = PhD)')
    plt.ylabel('Count')
    plt.legend(title='Hiring Decision', labels=['Not Hired', 'Hired'])
    plt.show()

# ---------- Run Code ----------
data = clean_data('Data/recruitment_data.csv') # loads and cleans the data
see_data(data)