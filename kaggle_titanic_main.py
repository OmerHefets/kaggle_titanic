import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import titanic_tests


def load_titanic_data():
    return pd.read_csv('titanic_train.csv')


def load_titanic_test():
    return pd.read_csv('titanic_test.csv')


def extract_y_from_data(data, y_label):
    if y_label not in list(data):
        exit("Label not found in extract_y_from_data")
    y = data[y_label]
    X = data.drop(y_label, axis=1)
    return X, y


def titanic_data_cleaning(data, test, embarked, fill_na_median, feature_scaling):
    # not enough cabins data
    data = data.drop("Cabin", axis=1)
    # not useful
    data = data.drop("Name", axis=1)
    # not needed since pandas defines indexes
    data = data.drop("PassengerId", axis=1)
    # might be useful, but this is a basic exercise
    data = data.drop("Ticket", axis=1)
    # 2 NA values in 'Embarked', removed them
    if embarked:
        na_indexes = np.where(data['Embarked'].isnull())[0]
        for index in na_indexes:
            test = test.drop(index)
        data = data.dropna(subset=["Embarked"])
    # using median values for the missing ages
    if fill_na_median:
        age_median = data["Age"].median()
        data["Age"] = data["Age"].fillna(age_median)
    # one-hot for the binary classes
    data = oneHotEncoding(data, data["Sex"])
    data = oneHotEncoding(data, data["Embarked"])
    # Feature scaling
    if feature_scaling:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data, test


def oneHotEncoding(data, vector):
    encoder = LabelEncoder()
    vector_encoded = encoder.fit_transform(vector)
    class_number = encoder.classes_
    # save class names in 1D array
    class_names = []
    for name in encoder.classes_:
        class_names.append(name)
    # making 1Hot 2D array
    encoder = OneHotEncoder(categories='auto')
    vector_one_hot = encoder.fit_transform(vector_encoded.reshape(-1, 1)).toarray()
    # adding the 2D array to the data with the correct labels
    for i in range(len(vector_one_hot[0])):
        data[class_number[i]] = pd.Series(vector_one_hot[:, i], index=data.index)
    data = data.drop(vector.name, axis=1)
    # return the new DataFrame
    return data


titanic_data = load_titanic_data()
titanic_test = load_titanic_test()
X_train, y_train = extract_y_from_data(titanic_data, "Survived")
# clean data
X_train, y_train = titanic_data_cleaning(X_train, y_train, embarked=True, fill_na_median=True, feature_scaling=True)
# clean test set
titanic_test, _ = titanic_data_cleaning(titanic_test, 42, embarked=False, fill_na_median=True, feature_scaling=True)
print(titanic_test.info())
print(titanic_test.head())

SGD_classifier = SGDClassifier(random_state=42)
SGD_classifier.fit(X_train, y_train)
results = SGD_classifier.predict(titanic_test)
results = results.reshape(-1, 1)
np.savetxt("results.csv", results, delimiter=",")
# cross_val_score(SGD_classifier, X_train, y_train, cv=5, scoring="accuracy")
