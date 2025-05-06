from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


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
