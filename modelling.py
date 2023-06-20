import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm


# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 30

regression_features = ['len_of_query', 'len_of_doc', 'ratio_in_common', 'okapiBM25', 'embedding_cos_sim']

def read_data() -> pd.DataFrame:
    return pd.read_csv('./data/data.csv', encoding='latin1')


def train(df: pd.DataFrame):
    X = df[regression_features].to_numpy()
    y = df['relevance'].to_numpy().reshape(-1)
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())


def _train(df: pd.DataFrame):
    X = df['ratio_in_common'].to_numpy()
    y = df['relevance'].to_numpy()

    X = X.reshape(-1, 1)

    regressor = LinearRegression()
    regressor.fit(X, y)

    predictions = regressor.predict(X)

    plt.scatter(X, y, color='b', label='Actual Data')
    plt.plot(X, predictions, color='r', label='Linear Regression')
    plt.xlabel('ratio_in_common')
    plt.ylabel('relevance')
    plt.title('Linear Regression - Home Depot')
    plt.legend()
    plt.show()

    explained_variation = r2_score(y, predictions)

    print(f"Explained Variation (R-squared): {explained_variation:.4f}")


def model():
    df = read_data()
    train(df)


if __name__ == '__main__':
    model()
