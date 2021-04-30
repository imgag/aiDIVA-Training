import pandas as pd
import numpy as np
import multiprocessing as mp
import tempfile
import argparse
from operator import itemgetter


variant_consequences = {"transcript_ablation": 1,
                        "splice_acceptor_variant": 2,
                        "splice_donor_variant": 3,
                        "stop_gained": 4,
                        "frameshift_variant": 5,
                        "stop_lost": 6,
                        "start_lost": 7,
                        "transcript_amplification": 8,
                        "inframe_insertion": 9,
                        "inframe_deletion": 10,
                        "missense_variant": 11,
                        "protein_altering_variant": 12,
                        "splice_region_variant": 13,
                        "incomplete_terminal_codon_variant": 14,
                        "start_retained_variant": 15,
                        "stop_retained_variant": 16,
                        "synonymous_variant": 17,
                        "coding_sequence_variant": 18,
                        "mature_miRNA_variant": 19,
                        "5_prime_UTR_variant": 20,
                        "3_prime_UTR_variant": 21,
                        "non_coding_transcript_exon_variant": 22,
                        "intron_variant": 23,
                        "NMD_transcript_variant": 24,
                        "non_coding_transcript_variant": 25,
                        "upstream_gene_variant": 26,
                        "downstream_gene_variant": 27,
                        "TFBS_ablation": 28,
                        "TFBS_amplification": 29,
                        "TF_binding_site_variant": 30,
                        "regulatory_region_ablation": 31,
                        "regulatory_region_amplification": 32,
                        "feature_elongation": 33,
                        "regulatory_region_variant": 34,
                        "feature_truncation": 35,
                        "intergenic_variant": 36}


grouped_expanded_vcf = None
feature_list = None
num_partitions = 10


def reformat_vcf_file_and_read_into_pandas_and_extract_header(filepath):
    vcf_file_to_reformat = open(filepath, "r")

    header_line = ""
    comment_lines = []

    with open(filepath, "r") as temp_vcf:
        for line in temp_vcf:
            if line.strip().startswith("##"):
                comment_lines.append(line.strip())
            if line.strip().startswith("#CHROM"):
                header_line = line.strip()
            else:
                continue

        if header_line == "":
            print("ERROR: The VCF file seems to be corrupted")

    tmp = tempfile.NamedTemporaryFile(mode="w")
    tmp.write(vcf_file_to_reformat.read().replace(r"(\n(?!((((((chr)?[0-9]{1,2}|(chr)?[xXyY]{1}|(chr)?(M|m|MT|mt){1})\t)(.+\t){6,}(.+(\n|\Z))))|(#{1,2}.*(\n|\Z))|(\Z))))", ""))

    vcf_header = header_line.strip().split("\t")

    vcf_as_dataframe = pd.read_csv(tmp.name, names=vcf_header, sep="\t", comment="#", low_memory=False)

    vcf_file_to_reformat.close()
    tmp.close()

    vcf_as_dataframe = vcf_as_dataframe.rename(columns={"#CHROM": "CHROM"})
    vcf_as_dataframe = vcf_as_dataframe.drop(columns=["ID", "QUAL", "FILTER"])

    return comment_lines, vcf_as_dataframe


def extract_annotation_header(header):
    annotation_header = [entry.strip().replace("\">", "").split(": ")[1].split("|") for entry in header if entry.startswith("##INFO=<ID=CSQ")][0]

    return annotation_header


def extract_columns(cell):
    info_fields = str(cell).strip().split(";")
    new_cols = []
    csq = ""
    clnsig = ""
    clnvc = ""
    mc = ""
    rank = ""
    indel_ID = ""
    fathmm_xf = np.nan
    condel = np.nan
    eigen_phred = np.nan
    mutation_assessor = np.nan
    gnomAD_hom = np.nan
    gnomAD_an = np.nan
    gnomAD_homAF = np.nan
    capice = np.nan

    for field in info_fields:
        if field == "nan":
            print("NaN:", field)
            continue
        if field.startswith("CSQ="):
            csq = field.split("=")[1]
        if field.startswith("CLNSIG="):
            clnsig = field.split("=")[1]
        if field.startswith("CLNVC="):
            clnvc = field.split("=")[1]
        if field.startswith("MC="):
            mc = field.split("=")[1]
        if field.startswith("RANK="):
            rank = field.split("=")[1]
        if field.startswith("INDEL_ID="):
            indel_ID = field.split("=")[1]
        if field.startswith("FATHMM_XF="):
            if field.split("=")[1] != "nan":
                fathmm_xf = field.split("=")[1]
        if field.startswith("CONDEL="):
            if field.split("=")[1] != "nan":
                condel = field.split("=")[1]
        if field.startswith("EIGEN_PHRED="):
            if field.split("=")[1] != "nan":
                eigen_phred = field.split("=")[1]
        if field.startswith("MutationAssessor="):
            if field.split("=")[1] != "nan":
                mutation_assessor = field.split("=")[1]
        if field.startswith("gnomAD_Hom"):
            if field.split("=")[1] != "nan":
                gnomAD_hom = float(field.split("=")[1])
        if field.startswith("gnomAD_AN"):
            if field.split("=")[1] != "nan":
                gnomAD_an = float(field.split("=")[1])
        if field.startswith("CAPICE"):
            if field.split("=")[1] != "nan":
                capice = float(field.split("=")[1])

        if (gnomAD_hom != np.nan) & (gnomAD_an != np.nan) & (gnomAD_hom > 0.0) & (gnomAD_an > 0.0):
            gnomAD_homAF = gnomAD_hom / gnomAD_an
        
        #else:
        #    print("SKIP INFORMATION ENTRY")

    return [rank, indel_ID, csq, fathmm_xf, condel, eigen_phred, mutation_assessor, gnomAD_homAF, capice]
    #return [rank, indel_ID, csq]


def extract_vep_annotation(cell, annotation_header):
    annotation_fields = str(cell).strip().split(",")
    new_cols = []
    consequences = []

    # take the most severe annotation variant
    for field in annotation_fields:
        consequences.append(min([variant_consequences.get(x) for x in field.strip().split("|")[annotation_header.index("Consequence")].strip().split("&")]))

    target_index = min(enumerate(consequences), key=itemgetter(1))[0]
    new_cols = annotation_fields[target_index].strip().split("|")

    return new_cols


def add_INFO_fields_to_dataframe(vcf_as_dataframe):
    vcf_as_dataframe[["RANK","INDEL_ID", "CSQ", "FATHMM_XF", "CONDEL", "EIGEN_PHRED", "MutationAssessor", "homAF", "CAPICE"]] = vcf_as_dataframe.INFO.apply(lambda x: pd.Series(extract_columns(x)))
    #vcf_as_dataframe[["RANK","INDEL_ID", "CSQ"]] = vcf_as_dataframe.INFO.apply(lambda x: pd.Series(extract_columns(x)))
    vcf_as_dataframe = vcf_as_dataframe.drop(columns=["INFO"])

    return vcf_as_dataframe


def add_VEP_annotation_to_dataframe(vcf_as_dataframe, annotation_header):
    vcf_as_dataframe[annotation_header] = vcf_as_dataframe.CSQ.apply(lambda x: pd.Series(extract_vep_annotation(x, annotation_header)))
    vcf_as_dataframe = vcf_as_dataframe.drop(columns=["CSQ"])

    return vcf_as_dataframe

def calculate_homAF(row):
    if (row["gnomAD_AN"] != "") and (row["gnomAD_Hom"] != ""):
        if (float(row["gnomAD_Hom"]) > 0.0) & (float(row["gnomAD_AN"]) > 0.0):
            gnomAD_homAF = float(row["gnomAD_Hom"]) / float(row["gnomAD_AN"])
            return gnomAD_homAF
        else:
                return np.nan
    else:
        return np.nan

def add_sample_information_to_dataframe(vcf_as_dataframe):
    for sample in [col for col in vcf_as_dataframe if col.startswith("NA")]:
        vcf_as_dataframe.rename(columns={sample: sample + ".full"}, inplace=True)
        sample_header = [sample, "DP." + sample, "REF." + sample, "ALT." + sample, "AF." + sample, "GQ." + sample]
        vcf_as_dataframe[sample_header] = vcf_as_dataframe.apply(lambda x: pd.Series(extract_sample_information(x, sample)), axis=1)

        vcf_as_dataframe = vcf_as_dataframe.drop(columns=[sample + ".full"])

    vcf_as_dataframe = vcf_as_dataframe.drop(columns=["FORMAT"])

    return vcf_as_dataframe


def annotate_indels_with_combined_snps_information(row, grouped_expanded_vcf, feature):
    if grouped_expanded_vcf[feature].get_group(row["INDEL_ID"]).empty:
        return np.nan
    else:
        return grouped_expanded_vcf[feature].get_group(row["INDEL_ID"]).median()


def add_simple_repeat_annotation(row, grouped_expanded_vcf):
    simple_repeat = grouped_expanded_vcf["simpleRepeat"].get_group(row["INDEL_ID"]).unique().astype(str)

    return "&".join(simple_repeat)


def combine_vcf_dataframes(vcf_as_dataframe):
    global grouped_expanded_vcf

    for feature in feature_list:
        if (feature == "MaxAF") | (feature == "MAX_AF"):
            continue
        elif (feature == "simpleRepeat"):
            continue
        elif (feature == "oe_lof"):
            continue
        elif (feature == "homAF"):
            continue
        else:
            vcf_as_dataframe[feature] = vcf_as_dataframe.apply(lambda row : pd.Series(annotate_indels_with_combined_snps_information(row, grouped_expanded_vcf, feature)), axis=1)

    return vcf_as_dataframe


def parallelized_indel_combination(vcf_as_dataframe, expanded_vcf_as_dataframe, features, n_cores):
    global feature_list
    feature_list = features

    for feature in feature_list:
        if (feature == "MaxAF") | (feature == "MAX_AF"):
            continue
        elif (feature == "simpleRepeat"):
            continue
        elif (feature == "oe_lof"):
            continue
        elif (feature == "homAF"):
            continue
        elif (feature == "SIFT"):
            expanded_vcf_as_dataframe[feature] = expanded_vcf_as_dataframe[feature].apply(lambda row: min([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))
        else:
            expanded_vcf_as_dataframe[feature] = expanded_vcf_as_dataframe[feature].apply(lambda row: max([float(value) for value in str(row).split("&") if ((value != ".") & (value != "nan"))], default=np.nan))


    global grouped_expanded_vcf
    grouped_expanded_vcf = expanded_vcf_as_dataframe.groupby("INDEL_ID")

    if n_cores is None:
        num_cores = 1
    else:
        num_cores = n_cores

    global num_partitions
    num_partitions = num_cores * 2

    if len(vcf_as_dataframe) <= num_partitions:
        dataframe_splitted = np.array_split(vcf_as_dataframe, 1)
    else:
        dataframe_splitted = np.array_split(vcf_as_dataframe, num_partitions)

    pool = mp.Pool(num_cores)
    vcf_as_dataframe = pd.concat(pool.map(combine_vcf_dataframes, dataframe_splitted))
    pool.close()
    pool.join()
    
    return vcf_as_dataframe


def convert_vcf_to_pandas_dataframe(input_file):
    header, vcf_as_dataframe = reformat_vcf_file_and_read_into_pandas_and_extract_header(input_file)
    annotation_header = extract_annotation_header(header)

    vcf_as_dataframe = add_INFO_fields_to_dataframe(vcf_as_dataframe)
    vcf_as_dataframe = add_VEP_annotation_to_dataframe(vcf_as_dataframe, annotation_header)

    # replace empty strings or only spaces with NaN
    vcf_as_dataframe = vcf_as_dataframe.replace(r"^\s*$", np.nan, regex=True)

    return vcf_as_dataframe


def write_vcf_to_csv(vcf_combined_as_dataframe, out_file):
    vcf_combined_as_dataframe.to_csv(out_file, sep="\t", encoding="utf-8", index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data", type=str, dest="in_data", metavar="input.vcf", required=True, help="VCF file to convert\n")
    parser.add_argument("--in_data_expanded", type=str, dest="in_data_expanded", metavar="input_expanded.vcf", required=True, help="Expanded VCF file to convert\n")
    parser.add_argument("--out_data", type=str, dest="out_data", metavar="output.csv", required=True, help="CSV file containing the combined converted VCF files\n")
    parser.add_argument("--feature_list", type=str, dest="feature_list", metavar="feature1,feature2,feature3", required=True, help="Comma separated list with the names of the previously annotated features")
    args = parser.parse_args()

    vcf_as_dataframe = convert_vcf_to_pandas_dataframe(args.in_data)
    #vcf_as_dataframe["homAF"] = vcf_as_dataframe.apply(lambda x: pd.Series(calculate_homAF(x)), axis=1)
    expanded_vcf_as_dataframe = convert_vcf_to_pandas_dataframe(args.in_data_expanded)

    # "SIFT,PolyPhen,REVEL,CADD_PHRED,ABB_SCORE,MAX_AF,segmentDuplication,EIGEN_PHRED,CONDEL,FATHMM_XF,MutationAssessor,phastCons46mammal,phastCons46primate,phastCons46vertebrate,phyloP46mammal,phyloP46primate,phyloP46vertebrate,oe_lof,homAF,CAPICE"
    feature_list = args.feature_list.split(",")
    vcf_combined_as_dataframe = parallelized_indel_combination(vcf_as_dataframe, expanded_vcf_as_dataframe, feature_list, 1)
    write_vcf_to_csv(vcf_combined_as_dataframe, args.out_data)
