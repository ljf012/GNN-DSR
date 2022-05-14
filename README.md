# GNN-DSR
This code is for the paper "Graph Neural Networks with Dynamic and Static Representations for Social Recommendation" which is accepted by DASFAA 2022. 

[Lin, J., Chen, S., Wang, J. (2022). Graph Neural Networks with Dynamic and Static Representations for Social Recommendation. In: , et al. Database Systems for Advanced Applications. DASFAA 2022. Lecture Notes in Computer Science, vol 13246. Springer, Cham. https://doi.org/10.1007/978-3-031-00126-0_18](https://link.springer.com/chapter/10.1007/978-3-031-00126-0_18)

This paper proposes a PyTorch framework called GNN-DSR for social recommendation.

If you use our works and codes in your research, please cite:
```bash
@inproceedings{lin2022gnndsr,
    title="Graph Neural Networks with Dynamic and Static Representations for Social Recommendation",
    author={Lin, Junfa and Chen, Siyuan and Wang, Jiahai},
    booktitle={Database Systems for Advanced Applications},
    year={2022},
    publisher={Springer International Publishing},
    address={Cham},
    pages={264--271},
    isbn={978-3-031-00126-0}
}
```

## Requirements
- Python 3.8
- CUDA 11.3
- PyTorch 1.8.1
- NumPy 1.19.2
- Pandas 1.1.3
- tqdm 4.50.2

## Get Started
1. Install all the requirements.

2. Train and evaluate the GNN-DSR using the Python script [main.py](main.py).  
   To reproduce the results on Ciao in our paper, you can run
   ```bash
   python main.py --test
   ```

   To see the detailed usage of main.py, you can run
   ```bash
   python main.py -h
   ```

3. Preprocess the datasets using the Python script [preprocess.py](preprocess.py).  
   For example, to preprocess the *Ciao* dataset, you can run
   ```bash
   python preprocess.py --dataset Ciao
   ```
   The above command will store the preprocessed data files in folder `datasets/Ciao`.

   Raw Datasets (Ciao and Epinions) can be downloaded at http://www.cse.msu.edu/~tangjili/trust.html 

   To see the detailed usage of [preprocess.py](preprocess.py), you can run
   ```bash
   python preprocess.py -h
   ```

## Preprocessed Data & Weights
If you cannot download the documents of preprocessed data and weights, you can try to download them at [Google Drive](https://drive.google.com/drive/folders/1Rma8Uh3vHjUuMUzHi10GUvYt49cN40Ta?usp=sharing)

