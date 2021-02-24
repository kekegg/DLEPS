# DLEPS
A Deep Learning based Efficacy Prediction System for Drug Discovery

# Setup

- ## Install package

This package requires the **rdkit**, **tensorflow >=1.15.0** and **Keras >=2.3.0**.

conda install -c rdkit rdkit
apt-get update
apt install libxrender1
apt install libxext6
pip install nltk
pip install tensorflow==1.15.0
pip install keras==2.3.0

- ## On Code ocean

The supporting files and sample input files for the model locates in the data folder. Results were saved in results folder.

# Run the model

- **Script options**

input files
1. The csv file with all the chemical SMILES in the column with string SMILES as the header, other columns will be copied to the output file and an efficacy score column will be appended.
2. The upregulated gene signatures using ENTREZGENE_ACC in a file without header, each gene occupy a row
3. The downregulated gene signatures using the same format

Conversion of gene names can be accomplished at https://biit.cs.ut.ee/gprofiler/convert

A sample command is as followed:
python driv_DLEPS.py --input=../../data/Brief_Targetmol_natural_product_2719 --output=../../results/np2719_Browning.csv --upset=../../data/BROWNING_up --downset=../../data/BROWNING_down --reverse=False

Batch jobs were put into run_script

Other options include:
    '--input', default=INPUTFILE,
                        'Brief format of chemicals: contains SMILES column. '
    '--use_onehot',  default=True,
                        'If use pre-stored one hot array to save time.'
    '--use_l12k',  default=None,
                        'Use pre-calculated L12k'
    '--upset',  default=None,
                        'Up set of genes'
    '--downset',  default=None,
                        'Down set of genes. '
    '--reverse',  default=True,
                        'If the drug Reverse the Up / Down set of genes. '
    '--output',  default='out.csv',
                        'Output file name. '

Jupyter notebook users may run DLEPS_tutorial.ipynb for better iterative computing and analysis.

