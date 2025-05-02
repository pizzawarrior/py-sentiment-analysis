import re
import pandas as pd
from IPython.display import display
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linspace, floor, ceil
from matplotlib.pyplot import scatter, xlabel, ylabel, title, plot
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

'''
    A note on the command %matlibplot inline:
    In essence, if you're working in a Jupyter Notebook environment, the %matplotlib inline command is the preferred way to handle Matplotlib plots. If you are using a standard Python script, you should use matplotlib.pyplot.ion() and matplotlib.pyplot.ioff() to control how plots are displayed.

'''

# link to dataset, with information: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

data_file = 'data.csv'

def read_data(data_file):
    df = pd.read_csv(data_file)
    return df

df = read_data(data_file)

# print(df.describe())
# display(df)
