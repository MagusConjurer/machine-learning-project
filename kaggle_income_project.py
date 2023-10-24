#!/usr/bin/env python
# kaggle_income_project.py

"""
Description:
Author: Cameron Davis
Date Created: 24 October 2023
Modified : 24 October 2023
"""

import sklearn


def load_data_description(filename):
    try:
        attributes = []
        if filename.endswith('.txt'):
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith("| attributes"):
                        next(f)
                        attr_line = next(f)
                        attributes = attr_line.replace("\n", "").split(", ")
                attributes.append("label")
                attributes.append("weight")
                return attributes
        else:
            print('Provided file', filename, 'is not a txt.')
    except OSError as e:
        print('{0} : {1}'.format(e.strerror, e.filename))


def load_dataset(filename, attr_list):
    try:
        if filename.endswith('.csv'):
            with open(filename, 'r') as f:
                data = []
                for line in f:
                    row = {}
                    entry = line.strip().split(',')
                    for i in range(len(entry)):
                        # See if it can be converted to an int before adding to row
                        try:
                            value = float(entry[i])
                        except ValueError:
                            value = entry[i]
                        row[attr_list[i]] = value
                    data.append(row)
            return data
        else:
            print('Provided file', filename, 'is not a csv.')
    except OSError as e:
        print('{0} : {1}'.format(e.strerror, e.filename))
        return None
    return None


def load_data():
    attr_list = load_data_description("dataset/data-desc.txt")
    if len(attr_list) > 0:
        train_data = load_dataset("dataset/train.csv", attr_list)
        test_data = load_dataset("dataset/test.csv", attr_list)

        return train_data, test_data


class KaggleProject:
    def __init__(self):
        self.results = []

    def __load_data(self):
        self.__load_dataset("dataset/train.csv")
        self.__load_dataset("dataset/test.csv")

