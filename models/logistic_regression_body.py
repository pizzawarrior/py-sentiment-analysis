from sklearn.model_selection import train_test_split
from processing import garment_df
from utils import (
    vectorize_text,
    fit_model,
    make_prediction,
    print_auc_score,
    generate_classification_report,
    analyse_coefficients
)


def clean_reviews(df):
    # drop reviews that have no body
    return df[df['review'] != 'UNKNOWN'][['review', 'review_favorable']]


def split_data(df):
    x_train, x_test, y_train, y_test = train_test_split(df['review'],
                                                        df['review_favorable'],
                                                        random_state=6040)
    return x_train, x_test, y_train, y_test


df = clean_reviews(garment_df)
x_train, x_test, y_train, y_test = split_data(df)
vect, x_train_vect, x_test_vect = vectorize_text(x_train, x_test, min_df=2)
model = fit_model(x_train_vect, y_train)
pred = make_prediction(model, x_test_vect)
coeffs = analyse_coefficients(model, vect)
auc = print_auc_score(y_test, pred)

print(generate_classification_report(y_test, pred))
print(coeffs)
