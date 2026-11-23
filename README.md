# Explainable-single-omic-and-multi-omic-analyses
Single-omic and multi_omic approaches for the identification of genetic markers for Alzheimer's disease.

The repository contains two folders.

The **datasets** folder contains training and test datasets for both single-omic and multi-omic models, organised into the respective folders. 

The **models** folder contains models for single-omic, intermediate fusion (IF) and late fusion (LF) approaches. These are divided in three folders:

1) The **single-omic** folder contains:
- the python scripts for training and test
- the requirements txt file
- the .csv file containing the dictionary that associates each mutation with the corresponding chromosome
  
2) The **multi_omic_IF** folder contains:
- the python scripts for training and test
- the requirements txt file
- the trained MOFA model (.hd5f file)
  
3) The **multi-omic_LF** folder contains:
-  the python scripts for training and test
-  the requirements txt file
-  the scripts needed for the main (micro_meta_utils_train.py, snp_gene_models.py, snp_gene_models_explain.py)
-  the early_best.pth file for loading the early fusion model and outdir_best folder containing the models relating to the      final layers of the single-omics configurations.
