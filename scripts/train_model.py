#################################################################################################
#################################################################################################
########                                                                           ##############
########                                                                           ##############
########                                                                           ##############
########                                                                           ##############
########                                                                           ##############
#################################################################################################
#################################################################################################

import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


random_seed = 14038


def read_train_and_test_data(train_data_file, test_data_file):
    train_data = pd.read_csv(train_data_file, sep='\t', low_memory=False)
    test_data = pd.read_csv(test_data_file, sep='\t', low_memory=False)

    return train_data, test_data


def prepare_data(data_to_prepare, feature_list):
    # make sure that the following three parameters are named correctly if they are present in the feature_list
    for feature in feature_list:
        if feature == "SegDupMax":
            data_to_prepare[feature].fillna(0, inplace=True)
        elif feature == "MaxAF":
            data_to_prepare[feature].fillna(0, inplace=True)
        elif feature == "ABB_SCORE":
            data_to_prepare[feature].fillna(0, inplace=True)
        else:
            print(feature)
            print(data_to_prepare[feature])
            data_to_prepare[feature].fillna(data_to_prepare[feature].median(), inplace=True)
            print(data_to_prepare[feature])

    prepared_labels = np.asarray(data_to_prepare["RANK"])
    prepared_features = np.asarray(data_to_prepare[feature_list])

    return prepared_features, prepared_labels


def train_model_with_gridsearch(training_features, training_labels):
    train_features, test_features, train_labels, test_labels = train_test_split(training_features, training_labels, test_size = 0.2, random_state = random_seed)

    # grid with the default parameters
    parameter_grid = {"n_estimators": [1000],
                      "criterion": ["gini"],
                      "max_depth": [None],
                      "max_features": ["auto"],
                      "min_samples_split": [2],
                      "min_samples_leaf": [1],
                      "bootstrap": [True],
                      "class_weight": [None],
                      "oob_score": [False]
                      }

    rf_clf = RandomForestClassifier(random_state = random_seed)
    grid_search = GridSearchCV(estimator = rf_clf, param_grid = parameter_grid, cv = 5, n_jobs = -1, verbose = 2)
    grid_search.fit(training_features, training_labels)

    best_grid = grid_search.best_estimator_

    return best_grid


def compute_model_statistics(trained_model, test_features, test_labels):
    print("Parameters of the trained model: \n")
    pprint(trained_model.get_params())
    print("Mean accuracy: ", trained_model.score(test_features, test_labels))    #grid_accuracy =evaluate(best_grid, test_features, test_labels)

    #print("Improvement of {:0.2f}%.".format(100 * (grid_accuracy - base_accuracy) / base_accuracy))

    #print(clf.feature_importances_)


def export_trained_model(export_file, model_to_export):
    if not export_file.endswith('.pkl'):
        export_file = export_file + '.pkl'

    pickle.dump(model_to_export, open(export_file, 'wb'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, dest='train_data', metavar='train.csv', required=True, help='CSV file containing the training data, used to train the random forest model\n')
    parser.add_argument('--test_data', type=str, dest='test_data', metavar='tets.csv', required=True, help='CSV file containing the test data, used to compute the model statistics\n')
    parser.add_argument('--model', type=str, dest='model', metavar='model.pkl', required=True, help='Specifies the name of the trained model to export\n')
    parser.add_argument('--model', type=str, dest='feature_list', required=True, help='List containing the features that should be used in the training\n')
    args = parser.parse_args()

    train_data, test_data = read_train_and_test_data(args.train_data, args.test_data)

    feature_list = args.feature_list

    train_features, train_labels = prepare_data(train_data, feature_list)
    test_features, test_labels = prepare_data(test_data, feature_list)

    trained_rf_model = train_model_with_gridsearch(train_features, train_labels)

    compute_model_statistics(trained_rf_model, test_features, test_labels)

    export_trained_model(args.model, trained_rf_model)
