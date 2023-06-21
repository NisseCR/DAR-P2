import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy import stats
import mord



# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 30

regression_features = ['len_of_query', 'len_of_doc', 'ratio_in_common', 'okapiBM25', 'embedding_cos_sim']

def read_data() -> pd.DataFrame:
    return pd.read_csv('./data/data.csv', encoding='latin1')


def train_multinomial(df: pd.DataFrame):
    df['relevance2'] = df.apply(lambda r: int(r['relevance']), axis=1)
    df = df[(df['relevance'] == 1) | (df['relevance'] == 2)|(df['relevance'] == 3)] #yes this is very clean code dont ask
    X = df[regression_features].to_numpy()
    y = df['relevance2'].to_numpy().reshape(-1)
    multinomial_model = LogisticRegression(multi_class='multinomial')
    multinomial_model.max_iter = 1000
    multinomial_model.fit(X, y)
    for clas, coefficients, intercept in zip(multinomial_model.classes_, multinomial_model.coef_, multinomial_model.intercept_):
        print(f"class: {clas}")
        print(f"intercept: {intercept}")
        for feature, coefficient in zip(regression_features, coefficients):
            print(f"\t{feature} has intercept:{intercept}")

    print("")


def train_ordinal(df: pd.DataFrame):
    df['relevance2'] = df.apply(lambda r: int(r['relevance']), axis=1)
    df = df[(df['relevance'] == 1) | (df['relevance'] == 2)|(df['relevance'] == 3)] #yes this is very clean code dont ask
    X = df[regression_features].to_numpy()
    y = df['relevance2'].to_numpy().reshape(-1)
    ord_reg = mord.LogisticIT()
    model = ord_reg.fit(X, y)
    for feature, coefficient in zip(regression_features, model.coef_):
        print(f"{feature} has coefficient: {coefficient}")
    for num, boundary in zip(range(1,len(model.theta_)+1), model.theta_):
        print(f"{num} has boundary: {boundary}")
    print("")

def train_multilinear(df: pd.DataFrame):
    X = df[regression_features].to_numpy()
    y = df['relevance'].to_numpy().reshape(-1)



    regressor = LinearRegression()
    regressor.fit(X, y)

    predictions = regressor.predict(X)

    # wizard shit om p te berekenen, gekopieerd van stack overflow------
    # werkt nog niet met p waarde?
    params = np.append(regressor.intercept_, regressor.coef_)
    # newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X.reset_index(drop=True)))
    # newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    # MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

    # Note if you don't want to use a DataFrame replace the two lines above with
    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]

    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    #p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3["feature"] = ["constant"] + regression_features
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilities"] = [params, sd_b, ts_b,
                                                                                                  p_values]
    print(myDF3)
    # ------------------------------------------------------------------


    explained_variation = r2_score(y, predictions)

    print(f"Explained Variation (R-squared): {explained_variation:.4f}")


def train_single_linear(df: pd.DataFrame):
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
    train_multinomial(df)


if __name__ == '__main__':
    model()
