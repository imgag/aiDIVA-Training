import argparse
import pandas as pd
import numpy as np


MEAN_DICT = {"phastCons46mammal": 0.09691308336428194,
             "phastCons46primate": 0.12353343703613741,
             "phastCons46vertebrate": 0.1366339183101041,
             "phyloP46mammal": -0.0063575303590607925,
             "phyloP46primate": -0.012076641890840553,
             "phyloP46vertebrate": 0.06761867323083483,
             "MutationAssessor": 1.7961304794577417,
             "custom_MutationAssessor": 1.7961304794577417,
             "CONDEL": 0.49699016949707825,
             "custom_CONDEL": 0.49699016949707825,
             "EIGEN_PHRED": 4.342947928406315,
             "custom_EIGEN_PHRED": 4.342947928406315,
             "CADD_PHRED": 4.471745325,
             "FATHMM_XF": 0.35846023623584666,
             "custom_FATHMM_XF": 0.35846023623584666,
             "SIFT": 0.35216996259535444,
             "REVEL": 0.28019263637740743,
             "PolyPhen": 0.5169017014355943,
             "oe_lof": 0.53667483333333332,
             "phyloP_vertebrate": 3.656086,
             "phyloP_mammal": 0.738286,
             "phyloP_primate": 0.447020,
             "phastCons_vertebrate": 0.730734,
             "phastCons_mammal": 0.720765,
             "phastCons_primate": 0.698389,
             "ALPHA_MISSENSE_SCORE": 0.4074365673385914,
             "EVE_SCORE": 0.4866676824933756,
             "CAPICE": 0.04487945928377704,
             "CADD_INDEL": 2.2435275606140186,
             "CAPICE_INDEL": 0.007112348665338191}


MEDIAN_DICT = {"MutationAssessor": 1.87,
               "custom_MutationAssessor": 1.87,
               "CONDEL": 0.4805749233199981,
               "custom_CONDEL": 0.4805749233199981,
               "EIGEN_PHRED": 3.010301,
               "custom_EIGEN_PHRED": 3.010301,
               "CADD_PHRED": 3.99,
               "FATHMM_XF": 0.209614,
               "custom_FATHMM_XF": 0.209614,
               "SIFT": 0.153,
               "REVEL": 0.193,
               "PolyPhen": 0.547,
               "oe_lof": 0.48225,
               "phyloP_vertebrate": 3.12,
               "phyloP_mammal": 1.026,
               "phyloP_primate": 0.618,
               "phastCons_vertebrate": 1.0,
               "phastCons_mammal": 0.986,
               "phastCons_primate": 0.953,
               "ALPHA_MISSENSE_SCORE": 0.2509,
               "EVE_SCORE": 0.4946792317600748,
               "CAPICE": 0.0006,
               "CADD_INDEL": 1.067,
               "CAPICE_INDEL": 0.00012335256906226277}

SYNONYMOUS_VARIANTS = ["synonymous_variant",
                       "start_retained_variant",
                       "stop_retained_variant"]

SPLICE_VARIANTS = ["splice_acceptor_variant",
                   "splice_donor_variant",
                   "splice_donor_5th_base_variant",
                   "splice_region_variant",
                   "splice_donor_region_variant",
                   "splice_polypyrimidine_tract_variant"]


RANDOM_SEED = 14038


## customize this method according to the features used in the training
def prepare_data_and_fill_missing_values(data_to_prepare, allele_frequency_list, feature_list):
    # make sure all population allele frequencies are present in the input file
    for allele_frequency in allele_frequency_list:
        data_to_prepare[allele_frequency] = data_to_prepare[allele_frequency].fillna(0)
        data_to_prepare[allele_frequency] = data_to_prepare[allele_frequency].apply(lambda row: pd.Series(max([float(frequency) for frequency in str(row).split("&")], default=np.nan)))

    # compute maximum Minor Allele Frequency (MAF)
    if not (("MaxAF" in data_to_prepare.columns) | ("MAX_AF" in data_to_prepare.columns)):
        data_to_prepare["MaxAF"] = data_to_prepare.apply(lambda row: pd.Series(max([float(frequency) for frequency in row[allele_frequency_list].tolist()])), axis=1)

    # make sure that the following three parameters are named correctly if they are present in the feature_list
    for feature in feature_list:
        if (feature == "MaxAF") | (feature == "MAX_AF"):
            data_to_prepare[feature] = data_to_prepare[feature].fillna(0)

        elif (feature == "homAF"):
            data_to_prepare[feature] = data_to_prepare[feature].fillna(0)

        elif (feature == "segmentDuplication"):
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            data_to_prepare[feature] = data_to_prepare[feature].fillna(0)

        elif (feature == "SIFT"):
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            data_to_prepare[feature] = data_to_prepare[feature].fillna(MEDIAN_DICT["SIFT"])

        elif feature == "oe_lof":
            data_to_prepare[feature] = data_to_prepare.apply(lambda row: min([float(value) for value in str(row[feature]).split("&") if ((value != ".") & (value != "nan") & (value != "") & (not ":" in value) & (not "-" in value))], default=np.nan), axis=1)
            data_to_prepare[feature] = data_to_prepare[feature].fillna(MEDIAN_DICT["oe_lof"])

        elif (feature == "HIGH_IMPACT"):
            data_to_prepare[feature] = data_to_prepare[feature].fillna(0)

        elif feature == "REVEL" or feature == "ALPHA_MISSENSE_SCORE" or feature == "EVE_SCORE":
            data_to_prepare[feature] = data_to_prepare.apply(lambda row: max([float(value) for value in str(row[feature]).split("&") if ((value != ".") & (value != "nan") & (value != "") & (not ":" in value) & (not "-" in value))], default=np.nan), axis=1)
            data_to_prepare.loc[~(data_to_prepare["MOST_SEVERE_CONSEQUENCE"].str.contains("|".join(SPLICE_VARIANTS))) & (data_to_prepare["IMPACT"] == "HIGH") & (data_to_prepare[feature].isna()), feature] = 1.0
            data_to_prepare[feature] = data_to_prepare[feature].fillna(MEDIAN_DICT[feature])

        elif feature == "IS_INDEL":
            continue
        
        else:
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            if ("phastCons" in feature) | ("phyloP" in feature):
                data_to_prepare[feature] = data_to_prepare[feature].fillna(MEAN_DICT[feature])

            else:
                data_to_prepare[feature] = data_to_prepare[feature].fillna(MEDIAN_DICT[feature])

    return data_to_prepare


def perform_preparation_and_save_separate(in_data_benign, in_data_path, out_train, out_test, allele_frequency_list, feature_list):
    data_to_prepare_benign = pd.read_csv(in_data_benign, sep='\t', low_memory=False)
    data_to_prepare_path = pd.read_csv(in_data_path, sep='\t', low_memory=False)

    #print(data_to_prepare.columns)
    #exit()

    prepared_data_benign = prepare_data_and_fill_missing_values(data_to_prepare_benign, allele_frequency_list, feature_list)
    prepared_data_pathogenic = prepare_data_and_fill_missing_values(data_to_prepare_path, allele_frequency_list, feature_list)

    train_data_benign = prepared_data_benign.sample(frac=0.9, random_state=RANDOM_SEED)
    train_data_pathogenic = prepared_data_pathogenic.sample(frac=0.9, random_state=RANDOM_SEED)
    test_data_benign = prepared_data_benign.drop(train_data_benign.index)
    test_data_pathogenic = prepared_data_pathogenic.drop(train_data_pathogenic.index)

    final_train_data = pd.concat([train_data_benign, train_data_pathogenic])
    final_train_data = final_train_data.sort_values(["#CHROM", "POS"], ascending=[True, True])
    final_test_data = pd.concat([test_data_benign, test_data_pathogenic])
    final_test_data = final_test_data.sort_values(["#CHROM", "POS"], ascending=[True, True])

    final_train_data.to_csv(out_train, sep="\t", index=False)
    final_test_data.to_csv(out_test, sep="\t", index=False)


def perform_preparation_and_save(in_data_benign, in_data_path, out_data, allele_frequency_list, feature_list):
    data_to_prepare_benign = pd.read_csv(in_data_benign, sep='\t', low_memory=False)
    data_to_prepare_path = pd.read_csv(in_data_path, sep='\t', low_memory=False)

    #print(data_to_prepare.columns)
    #exit()

    prepared_data_benign = prepare_data_and_fill_missing_values(data_to_prepare_benign, allele_frequency_list, feature_list)
    prepared_data_pathogenic = prepare_data_and_fill_missing_values(data_to_prepare_path, allele_frequency_list, feature_list)

    prepared_data = pd.concat([prepared_data_benign, prepared_data_pathogenic])
    prepared_data = prepared_data.sort_values(["#CHROM", "POS"], ascending=[True, True])
    prepared_data.to_csv(out_data, sep="\t", index=False)


def perform_preparation_and_save_single(in_data, out_data, allele_frequency_list, feature_list):
    data_to_prepare = pd.read_csv(in_data, sep='\t', low_memory=False)

    #print(data_to_prepare.columns)
    #exit()

    prepared_data = prepare_data_and_fill_missing_values(data_to_prepare, allele_frequency_list, feature_list)
    prepared_data = prepared_data.sort_values(["#CHROM", "POS"], ascending=[True, True])
    prepared_data.to_csv(out_data, sep="\t", index=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_data', type=str, dest='in_data', metavar='input.csv', required=False, help='CSV file containing the benign data that should be prepared\n')
    parser.add_argument('--in_benign', type=str, dest='in_benign', metavar='input.csv', required=False, help='CSV file containing the benign data that should be prepared\n')
    parser.add_argument('--in_path', type=str, dest='in_path', metavar='input.csv', required=False, help='CSV file containing the pathogenic data that should be prepared\n')
    parser.add_argument('--out_data', type=str, dest='out_data', metavar='output.csv', required=False, help='CSV file where the prepared data is stored\n')
    parser.add_argument('--out_train', type=str, dest='out_train', metavar='output.csv', required=False, help='CSV file where the prepared data is stored\n')
    parser.add_argument('--out_test', type=str, dest='out_test', metavar='output.csv', required=False, help='CSV file where the prepared data is stored\n')
    parser.add_argument('--feature_list', type=str, dest='feature_list', metavar='feature_name1,feaute_name2,feature_name3', required=True, help='List containing the used feature names separated with a comma\n')
    parser.add_argument('--allele_frequency_list', type=str, dest='allele_frequency_list', metavar='feature_name1,feaute_name2,feature_name3', required=True, help='List containing the used feature names separated with a comma\n')
    args = parser.parse_args()

    feature_list = args.feature_list.split(",")
    allele_frequency_list = args.allele_frequency_list.split(",")

    if (args.in_benign is not None) and (args.in_path is not None) and (args.out_test is not None) and (args.out_train is not None):
        perform_preparation_and_save_separate(args.in_benign, args.in_path, args.out_train, args.out_test, allele_frequency_list, feature_list)
    
    elif (args.in_benign is not None)  and (args.in_path is not None) and (args.out_data is not None):
        perform_preparation_and_save(args.in_benign, args.in_path, args.out_data, allele_frequency_list, feature_list)
    
    elif (args.in_data is not None) and (args.out_data is not None):
        perform_preparation_and_save_single(args.in_data, args.out_data, allele_frequency_list, feature_list)
    
    else:
        print("ERROR!")
