import numpy as np
import matplotlib.pyplot as plt

from processing import weighted_ratings_df


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


x, y = weighted_ratings_df["total_reviews"], weighted_ratings_df["weighted_avg_rating"]
alpha, beta = reviews_lin_reg(x, y)
# print(alpha, beta)

def make_residuals(alpha, x, beta):
    y_hat = np.array(alpha * x * beta)
    e = np.array(y) - y_hat
    return y_hat, e

def scatterplot_residuals(y_hat, e):
    plt.scatter(y_hat, e, marker='o')  # TODO: set this to variable?
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Residuals (Ratings)')
    plt.title('Residuals Scatterplot')
    plt.show()
    # what do we return here? (variable?)

def hist_residuals(e):
    _, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Residuals (Product Rating)')
    plt.hist(e)
    plt.show()


# TODO: call the plot functions above
