from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from utils import generate_classification_report, positive_text, negative_text
from sklearn.metrics import roc_auc_score
from processing import garment_df


def clean_reviews(df):
    # drop reviews that have no body
    return df[df['review'] != 'UNKNOWN'][['review', 'review_favorable']]


def split_data(df):
    x_train, x_test, y_train, y_test = train_test_split(df['review'],
                                                        df['review_favorable'],
                                                        random_state=6040)
    return x_train, x_test, y_train, y_test


def vectorize_text(x_train, x_test, min_df):
    # we only allow words with a freq >= 3, and use a range of 1-3 ngrams
    vect = CountVectorizer(analyzer='word', min_df=min_df, ngram_range=(1, 3))
    x_train_vect = vect.fit_transform(x_train)
    x_test_vect = vect.transform(x_test)
    return vect, x_train_vect, x_test_vect


def fit_model(x_train_vect, y_train):
    model = LogisticRegression()
    model.fit(x_train_vect, y_train)
    return model


def make_prediction(model, x_test_vect):
    return model.predict(x_test_vect)


def print_auc_score(y_test, pred):
    print(f'AUC: {roc_auc_score(y_test, pred)}')


def test_sentiment_excerpt(model, vect, positive_text, negative_text):
    # test prediction accuracy: how does the model respond to nuanced language?
    # should return (1, 0)
    return model.predict(vect.transform([positive_text, negative_text]))


def analyse_coefficients(model, vect):
    feature_names = np.array(vect.get_feature_names_out())
    sorted_coefs = model.coef_[0].argsort()
    # words connected to negative reviews (smallest weights), words connected to positive reviews, rated highest first
    return f'Smallest Coefs: {feature_names[sorted_coefs[:10]]}', f'Largest Coefs: {feature_names[sorted_coefs[:-11:-1]]}'


df = clean_reviews(garment_df)
x_train, x_test, y_train, y_test = split_data(df)
vect, x_train_vect, x_test_vect = vectorize_text(x_train, x_test, min_df=2)
model = fit_model(x_train_vect, y_train)
pred = make_prediction(model, x_test_vect)
auc = print_auc_score(y_test, pred)
sentiment_score = test_sentiment_excerpt(model, vect, positive_text, negative_text)
coeffs = analyse_coefficients(model, vect)

print(f'Classification report: {(generate_classification_report(y_test, pred))}')
print(coeffs)
print(sentiment_score)
