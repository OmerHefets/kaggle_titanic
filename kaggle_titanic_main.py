import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_titanic_data():
    return pd.read_csv('titanic_train.csv')


def extract_y_from_data(data, y_label):
    if y_label not in list(data):
        exit("Label not found in extract_y_from_data")
    y = data[y_label].copy()
    X = data.drop(y_label, axis=1)
    return X, y


def titanic_data_cleaning(data, fill_na_median):
    # not enough cabins data
    data = data.drop("Cabin", axis=1)
    # not useful
    data = data.drop("Name", axis=1)
    # not needed since pandas defines indexes
    data = data.drop("PassengerId", axis=1)
    # might be useful, but this is a basic exercise
    data = data.drop("Ticket", axis=1)
    # 2 NA values in 'Embarked', removed them
    data = data.dropna(subset=["Embarked"])
    if fill_na_median:
        age_median = data["Age"].median()
        data["Age"] = data["Age"].fillna(age_median)
    return data


titanic_data = load_titanic_data()
X_train, y_train = extract_y_from_data(titanic_data, "Survived")
X_train = titanic_data_cleaning(X_train, True)
print(X_train.info())
