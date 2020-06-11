import argparse
import pandas as pd
import numpy as np


random_seed = 14038


## customize this method according to the features used in the training
def prepare_data_and_fill_missing_values(data_to_prepare, feature_list):
    # make sure all population allele frequencies are present in the input file
    data_to_prepare["gnomAD_AF"] = data_to_prepare["gnomAD_AF"].fillna(0)
    data_to_prepare["gnomAD_AFR_AF"] = data_to_prepare["gnomAD_AFR_AF"].fillna(0)
    data_to_prepare["gnomAD_AMR_AF"] = data_to_prepare["gnomAD_AMR_AF"].fillna(0)
    data_to_prepare["gnomAD_ASJ_AF"] = data_to_prepare["gnomAD_ASJ_AF"].fillna(0)
    data_to_prepare["gnomAD_EAS_AF"] = data_to_prepare["gnomAD_EAS_AF"].fillna(0)
    data_to_prepare["gnomAD_FIN_AF"] = data_to_prepare["gnomAD_FIN_AF"].fillna(0)
    data_to_prepare["gnomAD_NFE_AF"] = data_to_prepare["gnomAD_NFE_AF"].fillna(0)
    data_to_prepare["gnomAD_OTH_AF"] = data_to_prepare["gnomAD_OTH_AF"].fillna(0)
    data_to_prepare["gnomAD_SAS_AF"] = data_to_prepare["gnomAD_SAS_AF"].fillna(0)

    data_to_prepare["AF"] = data_to_prepare["AF"].fillna(0)
    data_to_prepare["AA_AF"] = data_to_prepare["AA_AF"].fillna(0)
    data_to_prepare["EA_AF"] = data_to_prepare["EA_AF"].fillna(0)
    data_to_prepare["AFR_AF"] = data_to_prepare["AFR_AF"].fillna(0)
    data_to_prepare["AMR_AF"] = data_to_prepare["AMR_AF"].fillna(0)
    data_to_prepare["EAS_AF"] = data_to_prepare["EAS_AF"].fillna(0)
    data_to_prepare["EUR_AF"] = data_to_prepare["EUR_AF"].fillna(0)
    data_to_prepare["SAS_AF"] = data_to_prepare["SAS_AF"].fillna(0)

    data_to_prepare["gnomAD_AF"] = data_to_prepare["gnomAD_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["gnomAD_AFR_AF"] = data_to_prepare["gnomAD_AFR_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["gnomAD_AMR_AF"] = data_to_prepare["gnomAD_AMR_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["gnomAD_ASJ_AF"] = data_to_prepare["gnomAD_ASJ_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["gnomAD_EAS_AF"] = data_to_prepare["gnomAD_EAS_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["gnomAD_FIN_AF"] = data_to_prepare["gnomAD_FIN_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["gnomAD_NFE_AF"] = data_to_prepare["gnomAD_NFE_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["gnomAD_OTH_AF"] = data_to_prepare["gnomAD_OTH_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["gnomAD_SAS_AF"] = data_to_prepare["gnomAD_SAS_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))

    data_to_prepare["AF"] = data_to_prepare["AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["AA_AF"] = data_to_prepare["AA_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["EA_AF"] = data_to_prepare["EA_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["AFR_AF"] = data_to_prepare["AFR_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["AMR_AF"] = data_to_prepare["AMR_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["EAS_AF"] = data_to_prepare["EAS_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["EUR_AF"] = data_to_prepare["EUR_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))
    data_to_prepare["SAS_AF"] = data_to_prepare["SAS_AF"].apply(lambda row: max([float(value) for value in str(row).split("&")]))

    if not "MaxAF" in data_to_prepare.columns:
        data_to_prepare[["MaxAF"]] = data_to_prepare.apply(lambda row: pd.Series(max([float(row["AFR_AF"]), float(row["AMR_AF"]), float(row["EAS_AF"]), float(row["EUR_AF"]), float(row["SAS_AF"]), float(row["AA_AF"]), float(row["EA_AF"]), float(row["gnomAD_AFR_AF"]), float(row["gnomAD_AMR_AF"]), float(row["gnomAD_ASJ_AF"]), float(row["gnomAD_EAS_AF"]), float(row["gnomAD_FIN_AF"]), float(row["gnomAD_NFE_AF"]), float(row["gnomAD_OTH_AF"]), float(row["gnomAD_SAS_AF"])])), axis=1)

    data_to_prepare["SIFT"] = data_to_prepare["SIFT"].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["PolyPhen"] = data_to_prepare["PolyPhen"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["Condel"] = data_to_prepare["Condel"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["CADD_PHRED"] = data_to_prepare["CADD_PHRED"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["CADD_RAW"] = data_to_prepare["CADD_RAW"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["REVEL"] = data_to_prepare["REVEL"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["segmentDuplication"] = data_to_prepare["segmentDuplication"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    #data_to_prepare["simpleRepeat"] = data_to_prepare["simpleRepeat"].apply(lambda row: max([float(value) for value in str(row).split("&") if (value != "." & value != "nan")]))
    #data_to_prepare["custom_EIGEN_PHRED"] = data_to_prepare["custom_EIGEN_PHRED"].apply(lambda row: max([float(value) for value in str(row).split("&") if (value != "." & value != "nan")]))
    data_to_prepare["custom_SIFT"] = data_to_prepare["custom_SIFT"].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["fannsdb_CONDEL"] = data_to_prepare["fannsdb_CONDEL"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["fannsdb_FATHMM"] = data_to_prepare["fannsdb_FATHMM"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["fannsdb_MA"] = data_to_prepare["fannsdb_MA"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["fannsdb_POLYPHEN2"] = data_to_prepare["fannsdb_POLYPHEN2"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["fannsdb_SIFT"] = data_to_prepare["fannsdb_SIFT"].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["custom_MutationAssessor"] = data_to_prepare["custom_MutationAssessor"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
    data_to_prepare["custom_FATHMM_XF"] = data_to_prepare["custom_FATHMM_XF"].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))

    # make sure that the following three parameters are named correctly if they are present in the feature_list
    for feature in feature_list:
        if feature == "segmentDuplication":
            data_to_prepare[feature].fillna(0, inplace=True)
        elif feature == "MaxAF":
            data_to_prepare[feature].fillna(0, inplace=True)
        elif feature == "ABB_SCORE":
            data_to_prepare[feature].fillna(0, inplace=True)
        else:
            data_to_prepare[feature].fillna(data_to_prepare[feature].median(), inplace=True)

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
