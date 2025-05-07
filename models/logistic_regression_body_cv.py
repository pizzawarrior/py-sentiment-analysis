import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from utils import positive_text, negative_text, generate_classification_report
from processing import garment_df
from models.logistic_regression_body import clean_reviews, split_data


'''
    Let's try a different clasifier with 5-fold cross validation,
    and some preprocessing to try to boost the weights of the negative ratings
'''


def text_negation(text):
    text = re.sub(r'not (\w+)', r'not_\1', text)  # join words that are preceded by 'not' into one ngram
    text = re.sub(r"n't (\w+)", r'not_\1', text)  # convert words suffixed with "n't" into "not_", then join with next word
    return text


def build_pipeline():
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1, 3), min_df=2, preprocessor=text_negation)),
        ('clf', LogisticRegression(max_iter=200))
    ])
    return pipeline


def get_param_grid():
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'clf__penalty': ['l2'],             # L2 is default; L1 requires liblinear solver
        'clf__solver': ['liblinear']        # Required for L1 penalty or smaller datasets
    }
    return param_grid


def train_model(x_train, y_train):
    pipeline = build_pipeline()
    param_grid = get_param_grid()
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=1)
    grid.fit(x_train, y_train)
    return grid


def evaluate_model(grid, x_test, y_test):
    print("Best parameters:", grid.best_params_)
    auc_score = grid.score(x_test, y_test)
    print("AUC score on test:", auc_score)
    return auc_score


def predict_excerpt(grid, excerpts):
    predictions = grid.predict([text_negation(text) for text in excerpts])
    print("Prediction:", predictions)
    return predictions


def train_evaluate_predict(x_train, x_test, y_train, y_test, positive_text, negative_text):
    grid = train_model(x_train, y_train)
    evaluate_model(grid, x_test, y_test)
    predict_excerpt(grid, [positive_text, negative_text])
    pred = grid.predict(x_test)
    generate_classification_report(y_test, pred)


df = clean_reviews(garment_df)
x_train, x_test, y_train, y_test = split_data(df)
train_evaluate_predict(x_train, x_test, y_train, y_test, positive_text, negative_text)
