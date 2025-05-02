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

from data_loader import read_data, data_file

'''
    A note on the command %matlibplot inline:
    In essence, if you're working in a Jupyter Notebook environment, the %matplotlib inline command is the preferred way to handle Matplotlib plots. If you are using a standard Python script, you should use matplotlib.pyplot.ion() and matplotlib.pyplot.ioff() to control how plots are displayed.

'''

df = read_data(data_file)
df = df.drop(['Unnamed: 0', 'Clothing ID', 'Division Name', 'Department Name'], axis=1)
new_col_names = ['age', 'review_title', 'review', 'rating', 'is_recommended', 'review_helpful_count', 'item_class']
df = df.rename(columns=dict(zip(df.columns, new_col_names))).fillna('UNKNOWN') # rename columns, fill NaN values with 'UNKNOWN'
