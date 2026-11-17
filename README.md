# Explainable-single-omic-and-multi-omic-analyses
Single-omic and multi_omic approaches for the identification of genetic markers.

The repository contains two folders.

The **single_omic** folder contains training and test datasets, python script and the txt requirements file. 

The **multi_omic** folder contains genes and SNPs datasets, common for both intermediate fusion (IF) and late fusion (LF) approaches. This folder contains other two folders:

1) The **multi_omic_IF** folder contains the python script, requirements txt file and the trained MOFA model (.hdf5 file).
  
2) The **multi_omic_LF** folder contains the python script, requirements txt file, the python script needed for the main (snp_gene_model_expalin.py) and the early_best.pth file for loading the early fusion model.
