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

def clean_data(df):
    df = df.drop(['Unnamed: 0', 'Clothing ID', 'Division Name', 'Department Name'], axis=1)
    new_col_names = ['age', 'review_title', 'review', 'rating', 'is_recommended', 'review_helpful_count', 'item_class']
    df = df.rename(columns=dict(zip(df.columns, new_col_names))).fillna('UNKNOWN')  # rename columns, fill NaN values with 'UNKNOWN'
    return df

cleaned_df = clean_data(df)

def garment_class(cleaned_df):
    '''
        Takes a cleaned df and drops rows where garment class is not known
        Adds a new binary column where reviews with ratings > 3 get a score of 1 for favorable, else 0
    '''
    garment_df = cleaned_df[cleaned_df['item_class'] != 'UNKNOWN'].copy()
    garment_df.loc[:, 'review_favorable'] = np.where(garment_df['rating'] > 3, 1, 0)
    return garment_df

garment_df = garment_class(cleaned_df)

def aggregate_ratings(garment_df):
    '''
        Takes a garment_df dataframe and groups all product types together and counts how many favorable reviews each has.
        We then calculate the weighted average of the favorable reviews for each garment type.
        Returns a new table sorted by items that have the highest favorable ratings first.
    '''
    total_reviews = garment_df.groupby('item_class')['review_favorable'].count()
    favorable_reviews = garment_df.groupby('item_class')['review_favorable'].sum()
    weighted_avg_rating = favorable_reviews / total_reviews

    weighted_ratings = pd.DataFrame({
        'total_reviews': total_reviews,
        'favorable_reviews': favorable_reviews,
        'weighted_avg_rating': weighted_avg_rating
    })

    weighted_ratings = weighted_ratings.sort_values(by=['weighted_avg_rating', 'total_reviews'], ascending=False).reset_index()
    return weighted_ratings

weighted_ratings_df = aggregate_ratings(garment_df)

def reviews_lin_reg(x, y):
    '''
        Takes a weighted_ratings_df and performs a linear regression by way of ordinary least squares.
        We want to predict the ratings of each product based on the number of total reviews.
        We use a univariate model with the independent variable as the number of reviews, and
        a dependent variable of ratings.
        We calculate a, B in the formula y ~ a * x + B
        Note that the function uses a vector "u" which is a vector of all ones. This represents the
        bias term, Beta.
    '''
    m = len(x)
    assert len(y) == m
    u = np.ones(m)
    alpha = (x.dot(y) - (u.dot(x) * u.dot(y) / m)) / (x.dot(x) - (u.dot(x) ** 2 / m))
    beta = (u.dot(y - alpha * x)) / m

    return (alpha, beta)

alpha, beta = reviews_lin_reg(weighted_ratings_df["total_reviews"], weighted_ratings_df["weighted_avg_rating"])
print(alpha, beta)
