from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from processing import garment_df


'''
    Perform a logistic regression using the review titles to predict the rating.
    Review titles consist of a maximum of 12 words
    We use n-grams to group words together
    Link to article on n-grams: https://medium.com/@abhishekjainindore24/n-grams-in-nlp-a7c05c1aff12
'''

def clean_titles(df):
    # filter out rows where title == 'UNKNOWN'
    review_titles = df[df['review_title'] != 'UNKNOWN'][['review_title', 'review_favorable']]
    return review_titles

review_titles = clean_titles(garment_df)

def split_data(df):
    x_train, x_test, y_train, y_test = train_test_split(df['review_title'],
                                                        df['review_favorable'],
                                                        random_state=6040)
    return x_train, x_test, y_train, y_test


def vectorize_text(x_train, x_test):
    # we only allow words with a freq >= 3, and use a range of 1-3 ngrams
    vect = CountVectorizer(analyzer='word', min_df=3, ngram_range=(1, 3))
    x_train_vect = vect.fit_transform(x_train)
    x_test_vect = vect.transform(x_test)
    return vect, x_train_vect, x_test_vect

def fit_model(x_train_vect, y_train):
    model = LogisticRegression()
    model.fit(x_train_vect, y_train)
    return model

def predict(model, x_test_vect, y_test):
    pred = model.predict(x_test_vect)
    return roc_auc_score(y_test, pred)

x_train, x_test, y_train, y_test = split_data(review_titles)
vect, x_train_vect, x_test_vect = vectorize_text(x_train, x_test)
model = fit_model(x_train_vect, y_train)
pred = predict(model, x_test_vect, y_test)

print(f'AUC: {predict(model, x_test_vect, y_test)}')
