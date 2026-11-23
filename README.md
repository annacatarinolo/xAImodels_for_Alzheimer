# Explainable-single-omic-and-multi-omic-analyses
Single-omic and multi_omic approaches for the identification of genetic markers for Alzheimer's disease.

The repository contains two folders.

The **datasets** folder contains training and test datasets for both single-omic and multi-omic models. 

The **models** folder contains model for single-omic, intermediate fusion (IF) and late fusion (LF) approaches. These are divided in three folders:

1) The **single-omic** folder contains the python scripts for training and test and the requirements txt file.
  
2) The **multi_omic_IF** folder contains the python scripts for training and test, requirements txt file, and the trained MOFA model (.hd5f file)
  
4) The **multi-omic_LF** folder contains the python scripts for training and test and the scripts needed for the main (snp_gene_model_explain.py) and the early_best.pth file for loading the early fusion model.
