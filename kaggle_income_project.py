#!/usr/bin/env python
# kaggle_income_project.py

"""
Description:
Author: Cameron Davis
Date Created: 24 October 2023
Modified : 24 October 2023
"""

import sys
import datetime
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


def load_dataset(filename):
    try:
        if filename.endswith('.csv'):
            # Data we need to return
            data = {}
            labels = []
            label_counts = {}
            attribute_counts = {}
            missing_data_indices = []
            with open(filename, 'r') as f:
                attributes = []
                row_id = 0
                for line in f:
                    entry = line.strip().split(',')
                    if len(attributes) < 1:
                        attributes = entry
                        for attr in attributes:
                            attribute_counts[attr] = {}
                    else:
                        row = {}
                        for i in range(len(entry)):
                            # See if it can be converted to an int before adding to row
                            try:
                                value = float(entry[i])
                            except ValueError:
                                value = entry[i]

                            # Missing values seem to be indicated by ?
                            if isinstance(value, str) and "?" in value:
                                missing_data_indices.append((row_id, attributes[i]))

                            # If we have reached the label column, add to that list instead of the row data
                            if "income>50K" in attributes[i]:
                                labels.append(value)
                                label_counts[value] = label_counts.get(value, 0) + 1
                            else:
                                row[attributes[i]] = value
                                if value in attribute_counts[attributes[i]]:
                                    attribute_counts[attributes[i]][value] += 1
                                else:
                                    attribute_counts[attributes[i]][value] = 1
                        data[row_id] = row
                    row_id += 1

            # Identify which label has more identified rows
            majority_label = 0
            if len(label_counts) > 0:
                if label_counts[0] > label_counts[1]:
                    majority_label = 0
                else:
                    majority_label = 1
            return {"dataset": data,
                    "labels": labels,
                    "majority_label": majority_label,
                    "attribute_counts": attribute_counts,
                    "missing_data_indices": missing_data_indices}
        else:
            print('Provided file', filename, 'is not a csv.')
    except OSError as e:
        print('{0} : {1}'.format(e.strerror, e.filename))
        return None
    return None


def write_results_to_file(results):
    date_data = datetime.datetime.now()
    day = date_data.strftime("%d%b")
    time = date_data.strftime("%H%M")
    filename = "predictions/prediction-{}-{}.csv".format(day, time)
    try:
        with open(filename, "w") as f:
            print("Writing predictions to file:", filename)
            f.write("ID,Prediction")
            for i in range(len(results)):
                f.write("\n{},{}".format(i + 1, results[i]))
    except Exception as e:
        raise Exception("Unable to write predictions to file: {}".format(e))


class KaggleProject:
    def __init__(self):
        self.results = []
        self.training_X = None
        self.training_y = None
        self.testing_X = None

    def load_data(self):
        training_unprocessed_data = load_dataset("dataset/train.csv")
        print("\nProcessing training data")
        self.__pre_process_data(training_unprocessed_data)
        test_unprocessed_data = load_dataset("dataset/test.csv")
        print("\nProcessing test data")
        self.__pre_process_data(test_unprocessed_data)
        return 0

    def __pre_process_data(self, unprocessed_data):
        """
        Handle all missing (?) attribute values and convert the dictionary rows to arrays
        that the sklearn trees can handle.
        :return:
        """
        start_time = datetime.datetime.now()

        in_process_data = unprocessed_data["dataset"]
        in_process_labels = unprocessed_data["labels"]

        attribute_majorities = {}
        # Update all rows with missing values to use the attribute majority
        for missing in unprocessed_data["missing_data_indices"]:
            missing_time = datetime.datetime.now()
            sys.stdout.write("\rTime elapsed : {}".format(missing_time-start_time))
            row = in_process_data[missing[0]]
            attribute = missing[1]

            if attribute == "label":
                in_process_labels[missing(0)] = unprocessed_data["majority_label"]
            else:
                if attribute in attribute_majorities:
                    majority = attribute_majorities[attribute]
                else:
                    # Get sorted list of attribute values to identify the majority value
                    attribute_count = unprocessed_data["attribute_counts"][attribute]
                    majority = max(attribute_count, key=attribute_count.get)
                    attribute_majorities[attribute] = majority
                row[attribute] = majority

        vectorizer = DictVectorizer()
        processed_data = []
        for row_data in in_process_data.values():
            vectorize_time = datetime.datetime.now()
            sys.stdout.write("\rTime elapsed : {}".format(vectorize_time - start_time))
            # Test data includes the ID, so remove that before finishing the processing
            if len(row_data) == 15:
                row_data.pop('ID')
            processed_data.append(vectorizer.fit_transform(row_data).toarray())

        # Reshape so it can be handled by the sklearn methods
        processed_data = np.reshape(processed_data, (len(processed_data), 14))

        # Training data
        if len(in_process_labels) != 0:
            self.training_X = processed_data
            self.training_y = in_process_labels
        else:
            self.testing_X = processed_data

    def run_midterm_progress(self):

        self.load_data()

        # print("\nFitting the data using default AdaBoost")
        # adaBoost = AdaBoostClassifier()
        # adaBoost.fit(self.training_X, self.training_y)
        # predictions = adaBoost.predict(self.testing_X)

        print("\nFitting the data using default Random Forests")
        randforest = RandomForestClassifier()
        randforest.fit(self.training_X, self.training_y)
        predictions = randforest.predict(self.testing_X)

        write_results_to_file(predictions)


project = KaggleProject()
project.run_midterm_progress()
