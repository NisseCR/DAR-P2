import pandas as pd


# Pandas settings
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 30


def read_data() -> pd.DataFrame:
    return pd.read_csv('./data/data.csv', encoding='latin1')


def model():
    df = read_data()
    print(df.info())
    print(df)


if __name__ == '__main__':
    model()
