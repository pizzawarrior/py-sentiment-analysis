import numpy as np
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

alpha, beta = reviews_lin_reg(weighted_ratings_df["total_reviews"], weighted_ratings_df["weighted_avg_rating"])
# print(alpha, beta)

# TODO: add in an inspection of the residuals
