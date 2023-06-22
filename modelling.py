import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy import stats
import mord
import math
import statsmodels.api as sm
import statsmodels


# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 30

regression_features = [
        'all_words_in_common_query_title',
        'all_words_in_common_query_description',
        'count_in_common_query_title',
        'count_in_common_query_description',
        'count_not_in_common_query_title',
        'count_not_in_common_query_description',
        'numbers_in_common_query_title',
        'tf_idf_query_description',
        'glove_cos_sim_query_title',
        'glove_cos_sim_query_description'
    ]


def read_data(file_name: str) -> pd.DataFrame:
    return pd.read_csv(f"./data/{file_name}", encoding='latin1')


def train_multinomial(df: pd.DataFrame):
    X = df[regression_features].to_numpy()
    y = df['relevance2'].to_numpy().reshape(-1)
    multinomial_model = LogisticRegression(multi_class='multinomial')
    multinomial_model.max_iter = 1000
    multinomial_model.fit(X, y)
    for clas, coefficients, intercept in zip(multinomial_model.classes_, multinomial_model.coef_, multinomial_model.intercept_):
        print(f"class: {clas}")
        print(f"intercept: {intercept}")
        for feature, coefficient in zip(regression_features, coefficients):
            print(f"\t{feature} has coefficient:{coefficient}")
    return multinomial_model


def train_ordinal(df: pd.DataFrame):
    X = df[regression_features].to_numpy()
    y = df['relevance2'].to_numpy().reshape(-1)
    ord_reg = mord.LogisticIT()
    model = ord_reg.fit(X, y)
    for feature, coefficient in zip(regression_features, model.coef_):
        print(f"{feature} has coefficient: {coefficient}")
    for num, boundary in zip(range(1,len(model.theta_)+1), model.theta_):
        print(f"{num} has boundary: {boundary}")
    return model


def train_multilinear2(df: pd.DataFrame):
    X = df[regression_features]
    y = df['relevance']
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    return est2



def train_multilinear(df: pd.DataFrame):
    X = df[regression_features].to_numpy()
    y = df['relevance'].to_numpy().reshape(-1)

    regressor = LinearRegression()
    regressor.fit(X, y)

    predictions = regressor.predict(X)

    # wizard shit om p te berekenen, gekopieerd van stack overflow------
    # werkt nog niet met p waarde?
    #"""
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
    #"""

    explained_variation = r2_score(y, predictions)

    print(f"Explained Variation (R-squared): {explained_variation:.4f}")
    return regressor


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
    df_train = read_data('train.csv')
    df_train['relevance2'] = df_train.apply(lambda r: int(r['relevance']), axis=1)
    # yes this is very clean code don't ask
    df_train = df_train[(df_train['relevance'] == 1) | (df_train['relevance'] == 2) | (df_train['relevance'] == 3)]
    mul_model = train_multinomial(df_train)
    ord_model = train_ordinal(df_train)
    df_test = read_data('train.csv')
    df_test = df_test[(df_test['relevance'] == 1) | (df_test['relevance'] == 2) | (df_test['relevance'] == 3)]
    df_test['mul_results'] = df_test.apply(lambda r: mul_model.predict(r[regression_features].to_numpy().reshape(1, -1))[0], axis=1)
    df_test['ord_results'] = df_test.apply(lambda r: ord_model.predict(r[regression_features].to_numpy().reshape(1, -1))[0], axis=1)
    res = df_test[['relevance', 'mul_results', 'ord_results']]
    res['correct_mul'] = res.apply(lambda r: r['relevance'] == r['mul_results'], axis=1)
    res['correct_ord'] = res.apply(lambda r: r['relevance'] == r['ord_results'], axis=1)
    correct_ratio_mul = len(res[res['correct_mul']])/len(res)
    correct_ratio_ord = len(res[res['correct_ord']]) / len(res)
    print(f"ratio mul {correct_ratio_mul*100:.2f}%")
    print(f"ratio ord {correct_ratio_ord*100:.2f}%")
    confusion_matrix_mul = create_confusion_matrix(res, 'relevance', 'mul_results')
    confusion_matrix_ord = create_confusion_matrix(res, 'relevance', 'ord_results')
    print_confusion_matrix(confusion_matrix_mul, "multinomial")
    print_confusion_matrix(confusion_matrix_ord, "ordinal")
    print("")


def create_confusion_matrix(df: pd.DataFrame, category_actual: str, category_predicted: str):
    #this function assumes values of 1,2 or 3
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for _,r in df.iterrows():
        actual = int(r[category_actual])
        predicted = int(r[category_predicted])
        matrix[actual-1][predicted-1] += 1
    return matrix


def print_confusion_matrix(matrix: [[int]], name: str):
    # function assumes values of 1, 2 or 3
    print(f"confusion matrix of {name}")
    print("\tpredicted values")
    print("\t1\t\t2\t3")
    for (index, row) in zip(range(1, 4), matrix):
        result = f"{index}"
        for val in row:
            amount_of_chars = 1 if val == 0 else round(math.log(val, 10))
            result += f"\t{val}"+" "*(4 - amount_of_chars)
        print(result)



def model2():
    df_train = read_data('train.csv')
    model = train_multilinear(df_train)

    """
    df_test = read_data('test.csv')
    X = df_test[regression_features].to_numpy()
    y = df_test['relevance'].to_numpy()
    predictions = model.predict(X)

    explained_variation = r2_score(y, predictions)
    print(f"Explained Variation (R-squared): {explained_variation:.4f}")"""


def model3():
    df_train = read_data('train.csv')
    model = train_multilinear2(df_train)
    print(model.summary2())
    df_test = read_data('test.csv')
    X = df_test[regression_features]
    X.insert(0, 'const', 1)  # statsmodels is weird, const is treated as a feature
    y = df_test['relevance']
    print("")
    predictions = model.predict(X)
    print(f"root mean sqaure error: {statsmodels.tools.eval_measures.rmse(y, predictions)}")
    print("")


if __name__ == '__main__':
    model3()
