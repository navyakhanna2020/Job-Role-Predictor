import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('data sets/Resume dataset.csv')

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())
print(data['category'].value_counts())
print(data.columns)
print("Duplicate Rows:", data.duplicated().sum())

url_count = data["Text"].str.contains(r"http|www", case=False, regex=True).sum()

print("Number of resumes containing URLs:", url_count)
