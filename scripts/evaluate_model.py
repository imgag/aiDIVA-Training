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
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef


random_seed = 14038


def extract_features_from_input_data(data_to_extract, feature_list):
    extracted_labels = np.asarray(data_to_extract["RANK"])
    extracted_features = np.asarray(data_to_extract[feature_list])

    return extracted_features, extracted_labels

def compute_model_statistics(trained_model, test_features, test_labels):
    prediction = trained_model.predict(test_features)

    print("Parameters of the trained model: \n")
    pprint(trained_model.get_params())
    print("Mean accuracy: ", trained_model.score(test_features, test_labels))
    print("Accuracy: ", accuracy_score(test_labels, prediction))
    print("Balanced-Accuracy: ", balanced_accuracy_score(test_labels, prediction))
    print("Precision: ", precision_score(test_labels, prediction))
    print("Recall: ", recall_score(test_labels, prediction))
    print("Average-Precision: ", average_precision_score(test_labels, prediction))
    print("Matthews Correlation Coefficient: ", matthews_corrcoef(test_labels, prediction))


def import_model(model_file):
    model_to_import = open(model_file, "rb")
    model = pickle.load(model_to_import)
    model_to_import.close()

    return model


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, dest='test_data', metavar='tets.csv', required=True, help='CSV file containing the test data, used to compute the model statistics\n')
    parser.add_argument('--model', type=str, dest='model', metavar='model.pkl', required=True, help='Specifies the name of the trained model to export\n')
    parser.add_argument('--feature_list', type=str, dest='feature_list', metavar='feature1,feature2,feature3', required=True, help='List containing the features that were used to train the model\n')
    args = parser.parse_args()

    test_data = pd.read_csv(args.test_data, sep="\t", low_memory=False)
    model = import_model(args.model)
    feature_list = args.feature_list.split(",")
    test_features, test_labels = extract_features_from_input_data(test_data, feature_list)
    
    compute_model_statistics(model, test_features, test_labels)
