#!/usr/bin/env python
# kaggle_income_project.py

"""
Description:
Author: Cameron Davis
Date Created: 24 October 2023
Modified : 13 December 2023
"""

import sys
import datetime
import numpy as np
import sklearn.preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


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
            print("\nWriting predictions to file:", filename)
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

    def run_final_progress(self):
        self.load_data()

        # print("\nFitting the data using Perceptron with the default settings")
        # defaultPerceptron = Perceptron()
        # defaultPerceptron.fit(self.training_X, self.training_y)
        # predictions = defaultPerceptron.predict(self.testing_X)

        # print("\nFitting the data using SVM with the default settings")
        # defaultSVM = svm.SVC()
        # defaultSVM.fit(self.training_X, self.training_y)
        # predictions = defaultSVM.predict(self.testing_X)

        # print("\nFitting the data using Neural Net with relu")
        # reluNN = MLPClassifier(random_state=1, activation='relu', solver='adam')
        # reluNN.fit(self.training_X, self.training_y)
        # predictions = reluNN.predict(self.testing_X)

        # print("\nFitting the data using Gaussian Naive Bayes with the default settings")
        # defaultGNB = GaussianNB()
        # defaultGNB.fit(self.training_X, self.training_y)
        # predictions = defaultGNB.predict(self.testing_X)

        # print("\nFitting the data using Logistic Regression with the default settings")
        # defaultLogReg = LogisticRegression()
        # defaultLogReg.fit(self.training_X, self.training_y)
        # predictions = defaultLogReg.predict(self.testing_X)

        # print("\nFitting the data using Gaussian Naive Bayes with tuned settings")
        #
        # smoothing_options = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        # for option in smoothing_options:
        #     tunedGNB = GaussianNB(var_smoothing=option)
        #     tunedGNB.fit(self.training_X, self.training_y)
        #
        #     training_predictions = tunedGNB.predict(self.training_X)
        #     trainingAccuracy = accuracy_score(self.training_y, training_predictions)
        #     print(option, ": ", trainingAccuracy)

        # maxed out at 0.80556 accuracy using 1e-5

        # predictions = tunedGNB.predict(self.testing_X)

        # print("\nFitting the data using a tuned Random Forest")

        # numTrees = [5, 10, 25, 50, 100]
        # maxTreeDepth = [None, 5, 10, 15, 20, 25, 30]
        # minSamplesToSplit = [2, 4, 6, 8, 10]
        # usingBootstrap = [True, False]

        # print("Trees | Depth | Samples to Split | Is Bootstrap | Training Accuracy")
        # for trees in numTrees:
        #     for depth in maxTreeDepth:
        #         for samples in minSamplesToSplit:
        #             for isBootstrap in usingBootstrap:
        #                 tunedRandforest = RandomForestClassifier(n_estimators=trees,
        #                                                          max_depth=depth,
        #                                                          min_samples_split=samples,
        #                                                          bootstrap=isBootstrap)
        #                 tunedRandforest.fit(self.training_X, self.training_y)
        #                 trainingPredictions = tunedRandforest.predict(self.training_X)
        #                 trainingAccuracy = accuracy_score(self.training_y, trainingPredictions)
        #
        #                 print(trees, "|", depth, "|", samples, "|", isBootstrap, "|", trainingAccuracy)

        # 10, 30, 2, False -- 0.69673 (overfit)
        # 10, 25, 6, False -- 0.70371
        # 5, None, 10, True -- 0.702
        # finalRandForest = RandomForestClassifier(n_estimators=5,
        #                                          max_depth=None,
        #                                          min_samples_split=10,
        #                                          bootstrap=True)

        # best_num_trees = 0
        # best_tree_accuracy = 0
        # for tree in numTrees:
        #     treeTuning = RandomForestClassifier(n_estimators=tree)
        #     treeTuning.fit(self.training_X, self.training_y)
        #     treePredictions = treeTuning.predict(self.training_X)
        #     treeAccuracy = accuracy_score(self.training_y, treePredictions)
        #
        #     if treeAccuracy > best_tree_accuracy:
        #         best_tree_accuracy = treeAccuracy
        #         best_num_trees = tree
        #
        # print(best_num_trees, best_tree_accuracy)
        #
        # best_max_depth = None
        # best_depth_accuracy = 0
        # for depth in maxTreeDepth:
        #     depthTuning = RandomForestClassifier(n_estimators=best_num_trees,
        #                                          max_depth=depth)
        #     depthTuning.fit(self.training_X, self.training_y)
        #     depthPredictions = depthTuning.predict(self.training_X)
        #     depthAccuracy = accuracy_score(self.training_y, depthPredictions)
        #
        #     if depthAccuracy > best_depth_accuracy:
        #         best_depth_accuracy = depthAccuracy
        #         best_max_depth = depth
        #
        # print(best_max_depth, best_depth_accuracy)
        #
        # best_split = None
        # best_split_accuracy = 0
        # for split in minSamplesToSplit:
        #     splitTuning = RandomForestClassifier(n_estimators=best_num_trees,
        #                                          max_depth=best_max_depth,
        #                                          min_samples_split=split)
        #     splitTuning.fit(self.training_X, self.training_y)
        #     splitPredictions = splitTuning.predict(self.training_X)
        #     splitAccuracy = accuracy_score(self.training_y, splitPredictions)
        #
        #     if splitAccuracy > best_split_accuracy:
        #         best_split_accuracy = splitAccuracy
        #         best_split = split
        #
        # print(best_split, best_split_accuracy)
        #
        # best_split = None
        # best_split_accuracy = 0
        # for split in minSamplesToSplit:
        #     splitTuning = RandomForestClassifier(n_estimators=best_num_trees,
        #                                          max_depth=best_max_depth,
        #                                          min_samples_split=split)
        #     splitTuning.fit(self.training_X, self.training_y)
        #     splitPredictions = splitTuning.predict(self.training_X)
        #     splitAccuracy = accuracy_score(self.training_y, splitPredictions)
        #
        #     if splitAccuracy > best_split_accuracy:
        #         best_split_accuracy = splitAccuracy
        #         best_split = split
        #
        # bootstrappedForest = RandomForestClassifier(n_estimators=best_num_trees,
        #                                             max_depth=best_max_depth,
        #                                             min_samples_split=best_split,
        #                                             bootstrap=True)
        # bootstrappedForest.fit(self.training_X, self.training_y)
        # bootstrappedPredictions = bootstrappedForest.predict(self.training_X)
        # boostrappedAccuracy = accuracy_score(self.training_y, bootstrappedPredictions)
        #
        # nonBootstrappedForest = RandomForestClassifier(n_estimators=best_num_trees,
        #                                                max_depth=best_max_depth,
        #                                                min_samples_split=best_split,
        #                                                bootstrap=False)
        # nonBootstrappedForest.fit(self.training_X, self.training_y)
        # nonBootstrappedPredictions = nonBootstrappedForest.predict(self.training_X)
        # nonBootstrappedAccuracy = accuracy_score(self.training_y, nonBootstrappedPredictions)
        #
        # print("Bootstrapped:", boostrappedAccuracy)
        # print("Non Bootstrapped:", nonBootstrappedAccuracy)
        #
        # if boostrappedAccuracy > nonBootstrappedAccuracy:
        #     predictions = bootstrappedForest.predict(self.testing_X)
        # else:
        #     predictions = bootstrappedForest.predict(self.testing_X)

        # regularization term no affect with otherwise default settings
        # lbfgs 0.7994, newton-cg 0.81868, newton-cholesky 0.8186
        # solverOptions = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
        # for option in solverOptions:
        #     tunedLR = LogisticRegression(solver=option, max_iter=500)
        #     tunedLR.fit(self.training_X, self.training_y)
        #     lrPredictions = tunedLR.predict(self.training_X)
        #     lrAccuracy = accuracy_score(self.training_y, lrPredictions)
        #     print(option, lrAccuracy)

        # tunedLR = LogisticRegression(solver='newton-cg', max_iter=500)
        # tunedLR.fit(self.training_X, self.training_y)
        # predictions = tunedLR.predict(self.testing_X)

        # print('\nFitting the data using SVM on normalized training data')
        #
        # scaler = MinMaxScaler()
        # scaled_training_X = scaler.fit_transform(self.training_X)
        # scaled_testing_X = scaler.transform(self.testing_X)

        # cOptions = [0.001, 0.01, 0.1, 1, 10, 100]
        # cOptions = [1000, 10000, 100000] --- higher than 100 takes way too long

        # for c in cOptions:
        #     SVC = svm.SVC(C=c)
        #
        #     SVC.fit(scaled_training_X, self.training_y)
        #     predictions = SVC.predict(scaled_training_X)
        #     svcAccuracy = accuracy_score(self.training_y, predictions)
        #     print(c, svcAccuracy)

        # SVC = svm.SVC(C=100)
        # SVC.fit(scaled_training_X, self.training_y)
        # predictions = SVC.predict(scaled_testing_X)

        bestForest = RandomForestClassifier(n_estimators=10,
                                            max_depth=30,
                                            min_samples_split=6,
                                            bootstrap=False)
        bestForest.fit(self.training_X, self.training_y)
        predictions = bestForest.predict(self.training_X)

        write_results_to_file(predictions)


def main():
    project = KaggleProject()
    # project.run_midterm_progress()
    project.run_final_progress()


if __name__ == "__main__":
    main()
