import pandas as pd
import numpy as np
import argparse
import random
from Bio import SeqIO


def create_expanded_indel_vcf(indel_vcf, expanded_vcf, ref_sequence_path):
    random.seed(14038)
    
    chroms_to_cover = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"]
    ref_dict = {}
    
    if indel_vcf.endswith(".gz"):
        indel_file = gzip.open(indel_vcf, "rt")
    else:
        indel_file = open(indel_vcf, "r")
    expanded_file = open(expanded_vcf, "w")

    ref_seq_records = SeqIO.index(ref_sequence_path, "fasta")

    for chrom in chroms_to_cover:
        ref_dict[chrom] = str(ref_seq_records[chrom].seq)

    for line in indel_file:
        if line.startswith("#"):
            expanded_file.write(line)
            continue
        
        splitted_line = line.replace("\n", "").split("\t")
        info_fields = splitted_line[7].split(";")
        indel_ID = ""
        
        for field in info_fields:
            if field.startswith("INDEL_ID="):
                indel_ID = field.split("=")[1]
        
        if indel_ID == "":
            print("ERROR: Somethingwent wrong while determining the indel_ID!")
            print("INFO-FIELD:", info_fields)

        if splitted_line[0].startswith("chr"):
            chrom_id = splitted_line[0]
        else:
            chrom_id = "chr" + splitted_line[0]

        if (chrom_id == "chrMT") or (chrom_id == "chrM"):
            print("WARNING: Skip mitochondrial variant")
            continue

        ref_seq = ref_dict[chrom_id]
        window_start = int(splitted_line[1]) - 3
        window_end = int(splitted_line[1]) + len(splitted_line[3]) + 2
        extended_ref_seq = ref_seq[window_start:window_end]

        for i in range(abs(window_end-window_start)):
            alt_variant = ""
            ## TODO: Handle the case if "N" is in the reference
            if (extended_ref_seq[i] == "A") | (extended_ref_seq[i] == "T"):
                alt_variant = random.choice(["G", "C"])
            elif (extended_ref_seq[i] == "G") | (extended_ref_seq[i] == "C"):
                alt_variant = random.choice(["A", "T"])
            elif (extended_ref_seq[i] == "N"):
                print("WARNING: Reference base skipped since it was N!")
                continue
            else:
                print("ERROR: Something went wrong!")
            expanded_file.write(chrom_id + "\t" + str(window_start + i + 1) + "\t" + "." + "\t" + extended_ref_seq[i] + "\t" + alt_variant + "\t" + "." + "\t" + "." + "\t" + splitted_line[7] + "\n")

    indel_file.close()
    expanded_file.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data", type=str, dest="in_data", metavar="input.csv", required=True, help="CSV file to convert to VCF\n")
    parser.add_argument("--out_data", type=str, dest="out_data", metavar="output.vcf", required=True, help="output VCF file\n")
    parser.add_argument("--ref_path", type=str, dest="ref_path", metavar="ref.fasta", required=True, help="Path to the reference file.\n")
    args = parser.parse_args()

    create_expanded_indel_vcf(args.in_data, args.out_data, args.ref_path)

