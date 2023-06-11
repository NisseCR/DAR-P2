import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 30


def read_data() -> pd.DataFrame:
    return pd.read_csv('./data/data.csv', encoding='latin1')


def train(df: pd.DataFrame):
    X = df['complete_ratio'].to_numpy()
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
