# DLEPS
A Deep Learning based Efficacy Prediction System for Drug Discovery

> **If you use DLEPS — the code, the model weights, or any drug-efficacy predictions it
> produces — please cite the paper below.**
> Academic/research use only: commercial use of the code, the weights, or the predictions
> requires prior written permission. See [LICENSE](LICENSE) · [LICENSE.zh-CN.md](LICENSE.zh-CN.md).

## 📖 Citation

> Zhu J#, Wang J#, Wang X#, Gao M, Guo B, Gao M, Liu J, Yu Y, Wang L, Kong W, An Y, Liu Z,
> Sun X, Huang Z, Zhou H, Zhang N, Zheng R, Xie Z\*. Prediction of drug efficacy from
> transcriptional profiles with deep learning. *Nat Biotechnol*. 2021;39(11):1444–1452.
> [doi:10.1038/s41587-021-00946-z](https://doi.org/10.1038/s41587-021-00946-z)

**BibTeX**

```bibtex
@article{zhu2021prediction,
  title   = {Prediction of drug efficacy from transcriptional profiles with deep learning},
  author  = {Zhu, Jie and Wang, Jingxiang and Wang, Xin and Gao, Mingjing and
             Guo, Bingbing and Gao, Miaomiao and Liu, Jiarui and Yu, Yanqiu and
             Wang, Liang and Kong, Weikaixin and An, Yongpan and Liu, Zurui and
             Sun, Xinpei and Huang, Zhuo and Zhou, Hong and Zhang, Ning and
             Zheng, Ruimao and Xie, Zhengwei},
  journal = {Nature Biotechnology},
  year    = {2021},
  volume  = {39},
  number  = {11},
  pages   = {1444--1452},
  doi     = {10.1038/s41587-021-00946-z},
  url     = {https://doi.org/10.1038/s41587-021-00946-z}
}
```

**Machine-readable metadata:** [`citation.json`](citation.json) · [`CITATION.cff`](CITATION.cff) · [`citation.bib`](citation.bib) · [`CITATION.md`](CITATION.md)

**Cited by** — these indexes maintain independent citation graphs, so the numbers legitimately differ. Cite the index you actually consulted, together with its access date (retrieved 2026-10-21).

| Index | Count | Link |
|---|---:|---|
| Google Scholar | 222 | [cites](https://scholar.google.com.hk/scholar?oi=bibs&hl=en&cites=13650249607434417486&as_sdt=5) |
| OpenAlex | 192 | [W3192398644](https://openalex.org/W3192398644) |
| Crossref | 186 | [DOI](https://doi.org/10.1038/s41587-021-00946-z) |
| Semantic Scholar | 160 | [corpus 235470977](https://www.semanticscholar.org/paper/235470977) |

---

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

denseweight.h5
DLEPS_30000_tune_gvae10000.h5
SMILE_train_demo.h5
SMILE_test_demo.h5
RNA_train_demo.h5
RNA_test_demo.h5 can be found: 
https://drive.google.com/drive/folders/1Fye5Bn_ZkT_wjZJIWTcj4RqGXZq-nd4L

benchmark.csv can be found in data folder. 


