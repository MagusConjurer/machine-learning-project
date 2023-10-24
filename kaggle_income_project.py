#!/usr/bin/env python
# kaggle_income_project.py

"""
Description:
Author: Cameron Davis
Date Created: 24 October 2023
Modified : 24 October 2023
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer


def load_dataset(filename):
    try:
        if filename.endswith('.csv'):
            with open(filename, 'r') as f:
                data = []
                labels = []
                attributes = []
                vectorizer = DictVectorizer()
                for line in f:
                    entry = line.strip().split(',')
                    if len(attributes) < 1:
                        attributes = entry
                    else:
                        row = {}
                        for i in range(len(entry)):
                            # See if it can be converted to an int before adding to row
                            try:
                                value = float(entry[i])
                            except ValueError:
                                value = entry[i]

                            if "income>50K" in attributes[i]:
                                labels.append(value)
                            else:
                                row[attributes[i]] = value

                        data.append(vectorizer.fit_transform(row).toarray())
            return data, labels
        else:
            print('Provided file', filename, 'is not a csv.')
    except OSError as e:
        print('{0} : {1}'.format(e.strerror, e.filename))
        return None
    return None


def load_data():

    train_X, train_y = load_dataset("dataset/train.csv")
    test_X = load_dataset("dataset/test.csv")

    return train_X, train_y, test_X


class KaggleProject:
    def __init__(self):
        self.results = []
        self.training_X, self.training_y, self.testing_X = load_data()

    def run_midterm_progress(self):
        adaBoost = AdaBoostClassifier()
        adaBoost.fit(self.training_X, self.training_y)
        print("ID,Prediction")
        for i in range(1, len(self.testing_X)):
            row = self.testing_X[i]
            prediction = adaBoost.predict(row)
            print("{}, {}".format(i, prediction))


project = KaggleProject()
project.run_midterm_progress()