import argparse
import pandas as pd
import numpy as np
import pickle
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef


RANDOM_SEED = 14038


DEFAULT_PARAM_INFRAME_RF = {"bootstrap": [True],
                            "class_weight": [None],
                            "criterion": ["gini"],
                            "max_depth": [None],
                            "max_features": [1/3],
                            "min_samples_leaf": [1],
                            "min_samples_split": [2],
                            "n_estimators": [1000],
                            "oob_score": [False]}

PARAM_GRID_INFRAME_RF = {"bootstrap": [True],
                     "class_weight": ["balanced", None],
                     "criterion": ["entropy", "gini", "log_loss"],
                     "max_depth": [5, 7, 9, None],
                     "max_features": ["sqrt", "log2", 1/3],
                     #"min_samples_leaf": [1],
                     #"min_samples_split": [2],
                     "n_estimators": [1000], # not needed to tune (no risk of overfitting if number of trees increases)
                     "oob_score": [False]}


DEFAULT_PARAM_SNP_RF = {"bootstrap": [True],
                        "class_weight": [None],
                        "criterion": ["gini"],
                        "max_depth": [None],
                        "max_features": ["sqrt"],
                        "min_samples_leaf": [1],
                        "min_samples_split": [2],
                        "n_estimators": [1000],
                        "oob_score": [False]}

PARAM_GRID_SNP_RF = {"bootstrap": [True],
                     #"class_weight": [None],
                     "criterion": ["entropy", "gini", "log_loss"],
                     "max_depth": [5, 7, 9, None],
                     "max_features": ["sqrt", "log2", 1/3],
                     #"min_samples_leaf": [1],
                     #"min_samples_split": [2],
                     "n_estimators": [1000], # not needed to tune (no risk of overfitting if number of trees increases)
                     "oob_score": [False]}


DEFAULT_PARAM_INFRAME_AB = {"n_estimators": [1000],
                            "estimator__criterion": ["entropy"],
                            "estimator__max_depth": [9],
                            "estimator__max_features": ["sqrt"],
                            "estimator__min_samples_split": [2],
                            "estimator__min_samples_leaf": [2],
                            "learning_rate": [0.5]}

DEFAULT_PARAM_SNP_AB = {"n_estimators": [100],
                        "estimator__criterion": ["gini"],
                        "estimator__max_depth": [None],
                        "estimator__max_features": ["sqrt"],
                        "estimator__min_samples_split": [2],
                        "estimator__min_samples_leaf": [2],
                        "learning_rate": [0.5]}


DEFAULT_PARAM_INFRAME_HGB = {"loss": ["log_loss"],
                             "learning_rate": [0.1],
                             "max_iter": [300],
                             "max_leaf_nodes": [31],
                             "max_depth": [5],
                             "min_samples_leaf": [10],
                             "l2_regularization": [0.3],
                             "max_bins": [150]}

DEFAULT_PARAM_SNP_HGB = {"loss": ["log_loss"],
            "learning_rate": [0.05],
            "max_iter": [300],
            "max_leaf_nodes": [20],
            "max_depth": [7],
            "min_samples_leaf": [10],
            "l2_regularization": [0.1],
            "max_bins": [150]}


DEFAULT_PARAM_INFRAME_SVM = {"C": [10.0],
                             "kernel": ["poly"],
                             "degree": [3],
                             "gamma": ["auto"],
                             "coef0": [0.0],
                             "class_weight": [None]}

DEFAULT_PARAM_SNP_SVM = {"C": [50.0],
                         "kernel": ["poly"],
                         "degree": [3],
                         "gamma": ["auto"],
                         "coef0": [0.0],
                         "class_weight": [None]}


def read_train_and_test_data(train_data_file, test_data_file):
    train_data = pd.read_csv(train_data_file, sep='\t', low_memory=False)
    test_data = pd.read_csv(test_data_file, sep='\t', low_memory=False)

    return train_data, test_data


def extract_features_from_input_data(data_to_extract, feature_list):
    extracted_labels = np.asarray(data_to_extract["CLASS_LABEL"])
    extracted_features = np.asarray(data_to_extract[feature_list])

    return extracted_features, extracted_labels


def train_model_with_gridsearch(training_features, training_labels, feature_list, is_indel=False, hyperparameter_tuning=False, model_type="RF"):
    # grid with the default parameters
    parameter_grid_rf = {"n_estimators": [1000],
                      "criterion": ["gini"],
                      "max_depth": [None],
                      "max_features": ["sqrt"],
                      "min_samples_split": [2],
                      "min_samples_leaf": [1],
                      "bootstrap": [True],
                      "class_weight": [None],
                      "oob_score": [False]
                      }

    parameter_grid_ab = {"n_estimators": [50,100,500,1000],
                      "estimator__criterion": ["gini"],
                      "estimator__max_depth": [2,7,None],
                      "estimator__max_features": ["sqrt"],
                      "estimator__min_samples_split": [2],
                      "estimator__min_samples_leaf": [1],
                      "learning_rate": [0.5,1,1.5]
                      }
    
    if not hyperparameter_tuning:
        if model_type == "RF":
            if is_indel:
                parameter_grid_to_use = DEFAULT_PARAM_INFRAME_RF
                classifier = RandomForestClassifier(random_state = RANDOM_SEED)

            else:
                parameter_grid_to_use = DEFAULT_PARAM_SNP_RF
                classifier = RandomForestClassifier(random_state = RANDOM_SEED)

        elif model_type == "AB":
            if is_indel:
                parameter_grid_to_use = DEFAULT_PARAM_INFRAME_AB
                classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_SEED), random_state = RANDOM_SEED)

            else:
                parameter_grid_to_use = DEFAULT_PARAM_SNP_AB
                classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_SEED), random_state = RANDOM_SEED)

        elif model_type == "HGB":
            if is_indel:
                parameter_grid_to_use = DEFAULT_PARAM_INFRAME_HGB
                classifier = HistGradientBoostingClassifier(random_state = RANDOM_SEED)

            else:
                parameter_grid_to_use = DEFAULT_PARAM_SNP_HGB
                classifier = HistGradientBoostingClassifier(random_state = RANDOM_SEED)

        elif model_type == "SVM":
            if is_indel:
                parameter_grid_to_use = DEFAULT_PARAM_INFRAME_SVM
                classifier = SVC(random_state = RANDOM_SEED)

            else:
                parameter_grid_to_use = DEFAULT_PARAM_SNP_SVM
                classifier = SVC(random_state = RANDOM_SEED)

        else:
            pass

    else:
        if model_type == "RF":
            if is_indel:
                parameter_grid_to_use = PARAM_GRID_INFRAME_RF
                classifier = RandomForestClassifier(random_state = RANDOM_SEED)

            else:
                parameter_grid_to_use = PARAM_GRID_SNP_RF
                classifier = RandomForestClassifier(random_state = RANDOM_SEED)

        elif model_type == "AB":
            if is_indel:
                parameter_grid_to_use = PARAM_GRID_INFRAME_AB
                classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_SEED), random_state = RANDOM_SEED)

            else:
                parameter_grid_to_use = PARAM_GRID_SNP_AB
                classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_SEED), random_state = RANDOM_SEED)

        elif model_type == "HGB":
            if is_indel:
                parameter_grid_to_use = PARAM_GRID_INFRAME_HGB
                classifier = HistGradientBoostingClassifier(random_state = RANDOM_SEED)

            else:
                parameter_grid_to_use = PARAM_GRID_SNP_HGB
                classifier = HistGradientBoostingClassifier(random_state = RANDOM_SEED)

        elif model_type == "SVM":
            if is_indel:
                parameter_grid_to_use = PARAM_GRID_INFRAME_SVM
                classifier = SVC(random_state = RANDOM_SEED)

            else:
                parameter_grid_to_use = PARAM_GRID_SNP_SVM
                classifier = SVC(random_state = RANDOM_SEED)

        else:
            pass

    #classifier = RandomForestClassifier(random_state = random_seed)
    #ab_clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(),random_state = random_seed)
    grid_search = GridSearchCV(estimator = classifier, param_grid = parameter_grid_to_use, cv = 10, n_jobs = 10, verbose = 2)
    grid_search.fit(training_features, training_labels)

    best_grid = grid_search.best_estimator_
    
    if model_type != "SVM" and model_type != "HGB":
        importances = best_grid.feature_importances_
        std = np.std([best_grid.feature_importances_ for tree in best_grid.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        print("Feature ranking:")

        for f in range(training_features.shape[1]):
            #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
            print("%d. feature %s (%f)" % (f + 1, feature_list[indices[f]], importances[indices[f]]))

    return best_grid


def compute_model_statistics(trained_model, test_features, test_labels):
    prediction = trained_model.predict(test_features)

    print("Parameters of the trained model: \n")
    pprint(trained_model.get_params())
    print("Mean accuracy: ", trained_model.score(test_features, test_labels))
    print("Accuracy: ", accuracy_score(test_labels, prediction))
    #print("Balanced-Accuracy: ", balanced_accuracy_score(test_labels, prediction))
    print("Precision: ", precision_score(test_labels, prediction))
    print("Recall: ", recall_score(test_labels, prediction))
    print("Average-Precision: ", average_precision_score(test_labels, prediction))
    print("Matthews Correlation Coefficient: ", matthews_corrcoef(test_labels, prediction))
    
    

def export_trained_model(export_file, model_to_export):
    if not export_file.endswith('.pkl'):
        export_file = export_file + '.pkl'

    pickle.dump(model_to_export, open(export_file, 'wb'))


def perform_model_training_and_evaluation(feature_list, training_data, test_data, model_name, model_type="RF", is_indel=False, hyperparameter_tuning=False):
    training_features, training_labels = extract_features_from_input_data(training_data, feature_list)

    trained_rf_model = train_model_with_gridsearch(training_features, training_labels, feature_list, is_indel, hyperparameter_tuning, model_type)
    export_trained_model(model_name, trained_rf_model)

    # compute basic evalution scores if test data is present
    test_features, test_labels = extract_features_from_input_data(test_data, feature_list)
    compute_model_statistics(trained_rf_model, test_features, test_labels)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, dest='train_data', metavar='train.csv', required=True, help='CSV file containing the training data, used to train the random forest model\n')
    parser.add_argument('--test_data', type=str, dest='test_data', metavar='tets.csv', required=True, help='CSV file containing the test data, used to compute the model statistics\n')
    parser.add_argument('--model_name', type=str, dest='model_name', metavar='model.pkl', required=True, help='Specifies the name of the trained model to export\n')
    parser.add_argument('--feature_list', type=str, dest='feature_list', metavar='feature1,feature2,feature3', required=True, help='List containing the features that should be used in the training\n')
    parser.add_argument('--is_indel', dest='is_indel', action='store_true', required=False, help='Indicate that the used data consists of InDel data\n')
    parser.add_argument('--model_type', type=str, dest='model_type', metavar='RF', required=True, help='Specify the model type used for training [RF, AB, HGB, SVM]\n')
    parser.add_argument('--hyperparameter_tuning', dest='hyperparameter_tuning', action='store_true', required=False, help='Perform hyperparameter tuning\n')
    args = parser.parse_args()

    train_data = pd.read_csv(args.train_data, sep="\t", low_memory=False)
    test_data = pd.read_csv(args.test_data, sep="\t", low_memory=False)
    feature_list = args.feature_list.split(",")
    
    #print(train_data.columns)
    #print(test_data.columns)
    #print(feature_list)
    
    print("InDel model:", args.is_indel)
    print("Hyperparamter tuning:", args.hyperparameter_tuning)

    perform_model_training_and_evaluation(feature_list, train_data, test_data, args.model_name, args.model_type, args.is_indel, args.hyperparameter_tuning)
