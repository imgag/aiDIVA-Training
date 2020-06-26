import argparse
import pandas as pd
import numpy as np


mean_dict = {"phastCons46mammal": 0.09691308336428194,
             "phastCons46primate": 0.12353343703613741,
             "phastCons46vertebrate": 0.1366339183101041,
             "phyloP46mammal": -0.0063575303590607925,
             "phyloP46primate": -0.012076641890840553,
             "phyloP46vertebrate": 0.06761867323083483,
             "phastCons100": 0.11273633387190414,
             "phyloP100": 0.052907788505469275,
             "custom_MutationAssessor": 1.7961304794577417,
             "fannsdb_CONDEL": 0.49699016949707825,
             "custom_EIGEN_PHRED": 4.342947928406315,
             "CADD_PHRED": 4.471745325,
             "custom_FATHMM_XF": 0.35846023623584666,
             "SIFT": 0.35216996259535444,
             "REVEL": 0.28019263637740743,
             "PolyPhen": 0.5169017014355943}


median_dict = {"custom_MutationAssessor": 1.87,
               "fannsdb_CONDEL": 0.4805749233199981,
               "custom_EIGEN_PHRED": 3.010301,
               "CADD_PHRED": 3.99,
               "custom_FATHMM_XF": 0.209614,
               "SIFT": 0.153,
               "REVEL": 0.193,
               "PolyPhen": 0.547}


random_seed = 14038


## customize this method according to the features used in the training
def prepare_data_and_fill_missing_values(data_to_prepare, allele_frequency_list, feature_list):
    # make sure all population allele frequencies are present in the input file
    for allele_frequency in allele_frequency_list:
        input_data[allele_frequency] = input_data[allele_frequency].fillna(0)
        input_data[allele_frequency] = input_data[allele_frequency].apply(lambda row: pd.Series(max([float(frequency) for frequency in str(row).split("&")], default=np.nan)))

    # compute maximum Minor Allele Frequency (MAF)
    if not "MaxAF" in data_to_prepare.columns:
    input_data["MaxAF"] = input_data.apply(lambda row: pd.Series(max([float(frequency) for frequency in row[allele_frequency_list].tolist()])), axis=1)

    # make sure that the following three parameters are named correctly if they are present in the feature_list
    for feature in feature_list:
        if feature == "MaxAF" or feature == "MAX_AF":
            data_to_prepare[feature] = data_to_prepare[feature].fillna(0)
        elif feature == "segmentDuplication":
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            data_to_prepare[feature] = data_to_prepare[feature].fillna(0)
        elif feature == "ABB_SCORE":
            data_to_prepare[feature] = data_to_prepare[feature].fillna(0)
        elif "SIFT" in feature:
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            data_to_prepare[feature] = data_to_prepare[feature].fillna(median_dict["SIFT"])
        else:
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            if ("phastCons" in feature) | ("phyloP" in feature):
                data_to_prepare[feature] = data_to_prepare[feature].fillna(mean_dict[feature])
            else:
                data_to_prepare[feature] = data_to_prepare[feature].fillna(median_dict[feature])

    return data_to_prepare


def perform_preparation_and_save(in_data, out_data, feature_list):
    data_to_prepare = pd.read_csv(in_data, sep='\t', low_memory=False)

    prepared_data = prepare_data_and_fill_missing_values(data_to_prepare, feature_list)
    prepared_data.to_csv(out_data, sep="\t", index=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_data', type=str, dest='in_data', metavar='input.csv', required=True, help='CSV file containing the data that should be prepared\n')
    parser.add_argument('--out_data', type=str, dest='out_data', metavar='outpu.csv', required=True, help='CSV file where the prepared data is stored\n')
    parser.add_argument('--feature_list', type=str, dest='feature_list', metavar='feature_name1,feaute_name2,feature_name3', required=True, help='List containing the used feature names separated with a comma\n')
    args = parser.parse_args()

    feature_list = args.feature_list.split(",")

    perform_preparation_and_save(args.in_data, args.out_data, feature_list)
