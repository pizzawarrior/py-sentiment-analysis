
from sklearn.metrics import classification_report


def generate_classification_report(y_test, pred):
    report = classification_report(y_test, pred, target_names=["Negative", "Positive"])
    print(f'Classification report:\n{report}')
    return report


# Excerpt text to use to test model ability to detect nuance
positive_text = 'not an issue, dress is great'
negative_text = 'an issue, dress is not great'
