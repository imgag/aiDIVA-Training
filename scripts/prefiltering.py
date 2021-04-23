import argparse
import gzip
import tempfile


def filter_clinvar_variants(filepath_clinvar, filepath_indel_clinvar, filepath_snp_clinvar):
    if filepath_clinvar.endswith(".gz"):
        vcf_file_to_reformat = gzip.open(filepath_clinvar, "rt")
    else:
        vcf_file_to_reformat = open(filepath_clinvar, "r")
    file_indel_clinvar = open(filepath_indel_clinvar, "w")
    file_snp_clinvar = open(filepath_snp_clinvar, "w")

    # make sure that there are no unwanted linebreaks in the variant entries
    #tmp = tempfile.NamedTemporaryFile(mode="w+")
    #tmp.write(vcf_file_to_reformat.read().replace(r"(\n(?!((((((chr)?[0-9]{1,2}|(chr)?[xXyY]{1}|(chr)?(M|m|MT|mt){1})\t)(.+\t){6,}(.+(\n|\Z))))|(#{1,2}.*(\n|\Z))|(\Z))))", ""))
    #tmp.seek(0)
    
    indel_ID = 0

    # extract header from vcf file
    for line in vcf_file_to_reformat:
        if line.strip().startswith("##"):
            file_indel_clinvar.write(line)
            file_snp_clinvar.write(line)
            continue
        elif line.strip().startswith("#CHROM"):
            file_indel_clinvar.write(line)
            file_snp_clinvar.write(line)
            continue
        else:
            # skip empty lines if the VCF file is not correctly formatted (eg. if there are multiple blank lines in the end of the file)
            if line == "\n":
                continue
            
            splitted_line = line.split("\t")
            info_fields = splitted_line[7].split(";")
            
            clnsig = ""
            clnrevstat = ""
            clnvc = ""
            mc=""
            
            if splitted_line[0] == "MT":
                continue
            
            if not splitted_line[0].startswith("chr"):
                splitted_line[0] = "chr" + splitted_line[0]
            
            for field in info_fields:
                if field.startswith("CLNREVSTAT="):
                    clnrevstat = field.split("=")[1]
                if field.startswith("CLNSIG="):
                    clnsig = field.split("=")[1]
                if field.startswith("CLNVC="):
                    clnvc = field.split("=")[1]
                if field.startswith("MC="):
                    mc = field.split("=")[1]

            if (clnrevstat == "criteria_provided,_multiple_submitters,_no_conflicts") | (clnrevstat == "criteria_provided,_single_submitter"):
                if ((clnsig != "Uncertain_significance") & (("Benign" in clnsig) | ("Likely_benign" in clnsig) | ("Pathogenic" in clnsig) | ("Likely_pathogenic" in clnsig))):
                    if clnvc == "single_nucleotide_variant":
                        if not (("nonsense" in mc) | ("synonymous" in mc) | ("intron" in mc) | ("splice" in mc) | ("UTR" in mc) | ("non-coding" in mc) | ("downstream" in mc) | ("upstream" in mc)):
                            if (("Benign" in clnsig) | ("Likely_benign" in clnsig)):
                                splitted_line[7] = "RANK=" + str(0) + ";" + splitted_line[7]
                            elif (("Pathogenic" in clnsig) | ("Likely_pathogenic" in clnsig)):
                                splitted_line[7] = "RANK=" + str(1) + ";" + splitted_line[7]
                            else:
                                print("ERROR: There seems to be a problem in the significance filtering!")
                                print(clnsig)
                            file_snp_clinvar.write("\t".join(splitted_line))
                    else:
                        if not (("frameshift" in mc) | ("intron" in mc) | ("splice" in mc) | ("UTR" in mc) | ("nonsense" in mc) | ("non-coding" in mc) | ("no_sequence_alteration" in mc) | ("missense" in mc) | ("synonymous" in mc) | ("downstream" in mc) | ("upstream" in mc)):
                            if (("Benign" in clnsig) | ("Likely_benign" in clnsig)):
                                splitted_line[7] = "RANK=" + str(0) + ";" + splitted_line[7]
                            elif (("Pathogenic" in clnsig) | ("Likely_pathogenic" in clnsig)):
                                splitted_line[7] = "RANK=" + str(1) + ";" + splitted_line[7]
                            else:
                                print("ERROR: There seems to be a problem in the significance filtering!")
                                print(clnsig)
                            if (abs(len(splitted_line[3]) - len(splitted_line[4])) % 3 == 0):
                                splitted_line[7] = "indel_ID=" + str(indel_ID) + ";" + splitted_line[7]
                                file_indel_clinvar.write("\t".join(splitted_line))
                                indel_ID += 1

    vcf_file_to_reformat.close()
    file_indel_clinvar.close()
    file_snp_clinvar.close()
    #tmp.close()


def filter_hgmd_variants(filepath_hgmd, filepath_indel_hgmd, filepath_snp_hgmd):
    if filepath_hgmd.endswith(".gz"):
        vcf_file_to_reformat = gzip.open(filepath_hgmd, "rt")
    else:
        vcf_file_to_reformat = open(filepath_hgmd, "r")
    file_indel_hgmd = open(filepath_indel_hgmd, "w")
    file_snp_hgmd = open(filepath_snp_hgmd, "w")

    # make sure that there are no unwanted linebreaks in the variant entries
    #tmp = tempfile.NamedTemporaryFile(mode="w+")
    #tmp.write(vcf_file_to_reformat.read().replace(r"(\n(?!((((((chr)?[0-9]{1,2}|(chr)?[xXyY]{1}|(chr)?(M|m|MT|mt){1})\t)(.+\t){6,}(.+(\n|\Z))))|(#{1,2}.*(\n|\Z))|(\Z))))", ""))
    #tmp.seek(0)
    
    indel_ID = 0

    # extract header from vcf file
    for line in vcf_file_to_reformat:
        if line.strip().startswith("##"):
            file_indel_hgmd.write(line)
            file_snp_hgmd.write(line)
            continue
        elif line.strip().startswith("#CHROM"):
            file_indel_hgmd.write(line)
            file_snp_hgmd.write(line)
            continue
        else:
            # skip empty lines if the VCF file is not correctly formatted (eg. if there are multiple blank lines in the end of the file)
            if line == "\n":
                continue
            
            splitted_line = line.split("\t")
            info_fields = splitted_line[7].split(";")
            hgmd_class = ""
            
            if splitted_line[0] == "chrMT":
                continue
            
            for field in info_fields:
                if field.startswith("CLASS="):
                    hgmd_class = field.split("=")[1]

            if (hgmd_class == "DM") | (hgmd_class == "DM?"):
                splitted_line[7] = "RANK=" + str(1) + ";" + splitted_line[7]
                if ((len(splitted_line[3]) == 1) & (len(splitted_line[4]) == 1)):
                    file_snp_hgmd.write("\t".join(splitted_line))
                elif (abs(len(splitted_line[3]) - len(splitted_line[4])) % 3 == 0):
                    splitted_line[7] = "indel_ID=" + str(indel_ID) + ";" + splitted_line[7]
                    file_indel_hgmd.write("\t".join(splitted_line))
                    indel_ID += 1
                else:
                    print("INFO: Frameshift variant skipped!")

    vcf_file_to_reformat.close()
    file_indel_hgmd.close()
    file_snp_hgmd.close()
    #tmp.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_clinvar", type=str, dest="in_clinvar", metavar="clinvar.vcf", required=True, help="VCF file to filter\n")
    parser.add_argument("--indel_clinvar", type=str, dest="indel_clinvar", metavar="indel_clinvar.vcf", required=True, help="VCF file containing only the filtered InDel variants\n")
    parser.add_argument("--snp_clinvar", type=str, dest="snp_clinvar", metavar="snp_clinvar.vcf", required=True, help="VCF file containing only the filtered SNP variants\n")
    # optional arguments
    parser.add_argument("--in_hgmd", type=str, dest="in_hgmd", metavar="hgmd.vcf", required=False, help="VCF file to filter\n")
    parser.add_argument("--indel_hgmd", type=str, dest="indel_hgmd", metavar="indel_hgmd.vcf", required=False, help="VCF file containing only the filtered InDel variants\n")
    parser.add_argument("--snp_hgmd", type=str, dest="snp_hgmd", metavar="snp_hgmd.vcf", required=False, help="VCF file containing only the filtered SNP variants\n")
    args = parser.parse_args()

    ## TODO: Use temporary directory for files created during runtime

    filter_clinvar_variants(args.in_clinvar, args.indel_clinvar, args.snp_clinvar)
    
    if ("in_hgmd" in args) & ("indel_hgmd" in args) & ("snp_hgmd" in args):
        filter_hgmd_variants(args.in_hgmd, args.indel_hgmd, args.snp_hgmd)
