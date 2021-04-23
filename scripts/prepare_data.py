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
             "Polyphen": 0.5169017014355943,
             "oe_lof": 0.53667483333333332,
             "SIFT_score": 0.125889,
             "SIFT4G_score": 0.165176,
             "Polyphen2_HDIV_score": 0.625297,
             "Polyphen2_HVAR_score": 0.525333,
             "LRT_score": 0.083387,
             "MutationTaster_score": 0.976718,
             "MutationAssessor_score": 1.804558,
             "FATHMM_score": 0.406737,
             "PROVEAN_score": -2.756074,
             "VEST4_score": 0.487812,
             "MetaSVM_score": -0.576924,
             "MetaLR_score": 0.264987,
             "M-CAP_score": 0.095247,
             "REVEL_score": 0.293227,
             "MutPred_score": 0.461979,
             "MVP_score": 0.503517,
             "MPC_score": 0.723236,
             "PrimateAI_score": 0.576749,
             "DEOGEN2_score": 0.250167,
             "BayesDel_addAF_score": 0.026130,
             "BayesDel_noAF_score": -0.191720,
             "ClinPred_score": 0.664226,
             "LIST-S2_score": 0.756013,
             "DANN_score": 0.921397,
             "fathmm-MKL_coding_score": 0.685495,
             "fathmm-XF_coding_score": 0.473009,
             "Eigen-raw_coding": 0.063430,
             "Eigen-phred_coding": 4.492025,
             "Eigen-PC-raw_coding": 0.046218,
             "Eigen-PC-phred_coding": 4.501418,
             "GenoCanyon_score": 0.788855,
             "GERP++_RS": 3.161467,
             "phyloP100way_vertebrate": 3.656086,
             "phyloP30way_mammalian": 0.738286,
             "phyloP17way_primate": 0.447020,
             "phastCons100way_vertebrate": 0.730734,
             "phastCons30way_mammalian": 0.720765,
             "phastCons17way_primate": 0.698389,
             "SiPhy_29way_pi": 0.824618}


median_dict = {"MutationAssessor": 1.87,
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
               "Polyphen": 0.547,
               "oe_lof": 0.48225,
               "SIFT_score": 0.019000,
               "SIFT4G_score": 0.043,
               "Polyphen2_HDIV_score": 0.906,
               "Polyphen2_HVAR_score": 0.574,
               "LRT_score": 0.00011,
               "MutationTaster_score": 1.0,
               "MutationAssessor_score": 1.88,
               "FATHMM_score": 0.76,
               "PROVEAN_score": -2.23,
               "VEST4_score": 0.465,
               "MetaSVM_score": -0.877,
               "MetaLR_score": 0.1633,
               "M-CAP_score": 0.027157,
               "REVEL_score": 0.207,
               "MutPred_score": 0.443,
               "MVP_score": 0.503622,
               "MPC_score": 0.522819,
               "PrimateAI_score": 0.579387,
               "DEOGEN2_score": 0.157944,
               "BayesDel_addAF_score": -0.011897,
               "BayesDel_noAF_score": -0.248141,
               "ClinPred_score": 0.848067,
               "LIST-S2_score": 0.832417,
               "DANN_score": 0.989939,
               "fathmm-MKL_coding_score": 0.89497,
               "fathmm-XF_coding_score": 0.443175,
               "Eigen-raw_coding": 0.174976,
               "Eigen-phred_coding": 3.19434,
               "Eigen-PC-raw_coding": 0.202754,
               "Eigen-PC-phred_coding": 3.196693,
               "GenoCanyon_score": 0.999975,
               "GERP++_RS": 4.27,
               "phyloP100way_vertebrate": 3.12,
               "phyloP30way_mammalian": 1.026,
               "phyloP17way_primate": 0.618,
               "phastCons100way_vertebrate": 1.0,
               "phastCons30way_mammalian": 0.986,
               "phastCons17way_primate": 0.953,
               "SiPhy_29way_logOdds": 11.5556}


random_seed = 14038


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
        elif (feature == "ABB_SCORE"):
            data_to_prepare[feature] = data_to_prepare[feature].fillna(0)
        elif (feature == "SIFT"):
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            data_to_prepare[feature] = data_to_prepare[feature].fillna(median_dict["SIFT"])
        elif (feature == "SIFT_score"):
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            data_to_prepare[feature] = data_to_prepare[feature].fillna(median_dict["SIFT_score"])
        elif (feature == "SIFT4G_score"):
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            data_to_prepare[feature] = data_to_prepare[feature].fillna(median_dict["SIFT4G_score"])
        elif (feature == "FATHMM_score"):
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            data_to_prepare[feature] = data_to_prepare[feature].fillna(median_dict["FATHMM_score"])
        elif (feature == "PROVEAN_score"):
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            data_to_prepare[feature] = data_to_prepare[feature].fillna(median_dict["PROVEAN_score"])
        elif feature == "oe_lof":
            data_to_prepare[feature] = data_to_prepare.apply(lambda row: min([float(value) for value in str(row[feature]).split("&") if ((value != ".") & (value != "nan") & (value != "") & (not ":" in value) & (not "-" in value))], default=np.nan), axis=1)
            data_to_prepare[feature] = data_to_prepare[feature].fillna(median_dict["oe_lof"])
        else:
            data_to_prepare[feature] = data_to_prepare[feature].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan") & (value != ""))], default=np.nan))
            if ("phastCons" in feature) | ("phyloP" in feature):
                data_to_prepare[feature] = data_to_prepare[feature].fillna(mean_dict[feature])
            else:
                data_to_prepare[feature] = data_to_prepare[feature].fillna(median_dict[feature])

    return data_to_prepare


def perform_preparation_and_save(in_data_benign, in_data_path, out_train, out_test, allele_frequency_list, feature_list):
    data_to_prepare_benign = pd.read_csv(in_data_benign, sep='\t', low_memory=False)
    data_to_prepare_path = pd.read_csv(in_data_path, sep='\t', low_memory=False)

    #print(data_to_prepare.columns)
    #exit()

    prepared_data_benign = prepare_data_and_fill_missing_values(data_to_prepare_benign, allele_frequency_list, feature_list)
    prepared_data_pathogenic = prepare_data_and_fill_missing_values(data_to_prepare_path, allele_frequency_list, feature_list)
    
    train_data_benign = prepared_data_benign.sample(frac=0.9, random_state=14038)
    train_data_pathogenic = prepared_data_pathogenic.sample(frac=0.9, random_state=14038)
    test_data_benign = prepared_data_benign.drop(train_data_benign.index)
    test_data_pathogenic = prepared_data_pathogenic.drop(train_data_pathogenic.index)
    
    final_train_data = pd.concat([train_data_benign, train_data_pathogenic])
    final_train_data = final_train_data.sort_values(["CHROM", "POS"], ascending=[True, True])
    final_test_data = pd.concat([test_data_benign, test_data_pathogenic])
    final_test_data = final_test_data.sort_values(["CHROM", "POS"], ascending=[True, True])
    
    final_train_data.to_csv(out_train, sep="\t", index=False)
    final_test_data.to_csv(out_test, sep="\t", index=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_benign', type=str, dest='in_benign', metavar='input.csv', required=True, help='CSV file containing the benign data that should be prepared\n')
    parser.add_argument('--in_path', type=str, dest='in_path', metavar='input.csv', required=True, help='CSV file containing the pathogenic data that should be prepared\n')
    parser.add_argument('--out_train', type=str, dest='out_train', metavar='output.csv', required=True, help='CSV file where the prepared data is stored\n')
    parser.add_argument('--out_test', type=str, dest='out_test', metavar='output.csv', required=True, help='CSV file where the prepared data is stored\n')
    parser.add_argument('--feature_list', type=str, dest='feature_list', metavar='feature_name1,feaute_name2,feature_name3', required=True, help='List containing the used feature names separated with a comma\n')
    args = parser.parse_args()

    feature_list = args.feature_list.split(",")

    perform_preparation_and_save(args.in_benign, args.in_path, args.out_train, args.out_test, list(), feature_list)
