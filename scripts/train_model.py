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


def extract_features_from_input_data(data_to_extract, feature_list):
    prepared_labels = np.asarray(data_to_prepare["RANK"])
    prepared_features = np.asarray(data_to_prepare[feature_list])

    return prepared_features, prepared_labels


def train_model_with_gridsearch(training_data):
    extract_features_from_training_data

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
    print("Mean accuracy: ", trained_model.score(test_features, test_labels))


def export_trained_model(export_file, model_to_export):
    if not export_file.endswith('.pkl'):
        export_file = export_file + '.pkl'

    pickle.dump(model_to_export, open(export_file, 'wb'))


def perform_model_training_and_evaluation(feature_list, training_data, test_data=None, model_name):
    training_features, training_labels = extract_features_from_input_data(training_data, feature_list)

    trained_rf_model = train_model_with_gridsearch(training_features, training_labels)
    export_trained_model(model_name, trained_rf_model)

    # compute basic evalution scores if test data is present
    test_features, test_labels = extract_features_from_input_data(test_data, feature_list)
    compute_model_statistics(trained_rf_model, test_features, test_labels)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, dest='train_data', metavar='train.csv', required=True, help='CSV file containing the training data, used to train the random forest model\n')
    parser.add_argument('--test_data', type=str, dest='test_data', metavar='tets.csv', required=True, help='CSV file containing the test data, used to compute the model statistics\n')
    parser.add_argument('--model_name', type=str, dest='model_name', metavar='model.pkl', required=True, help='Specifies the name of the trained model to export\n')
    parser.add_argument('--feature_list', type=str, dest='feature_list', required=True, help='List containing the features that should be used in the training\n')
    args = parser.parse_args()

    perform_model_training_and_evaluation(args.feature_list, args.train_data, args.test_data, args.model_name)
