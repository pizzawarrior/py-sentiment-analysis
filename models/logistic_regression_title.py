from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from processing import garment_df
from utils import vectorize_text, fit_model, make_prediction, generate_classification_report, analyse_coefficients

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


def split_data(df):
    x_train, x_test, y_train, y_test = train_test_split(df['review_title'],
                                                        df['review_favorable'],
                                                        random_state=6040)
    return x_train, x_test, y_train, y_test


review_titles = clean_titles(garment_df)
x_train, x_test, y_train, y_test = split_data(review_titles)
vect, x_train_vect, x_test_vect = vectorize_text(x_train, x_test, min_df=3)
model = fit_model(x_train_vect, y_train)
pred = make_prediction(model, x_test_vect)
coeffs = analyse_coefficients(model, vect)

print(generate_classification_report(y_test, pred))
# print(coeffs)
# print(test_sentiment_excerpt(model, vect))
