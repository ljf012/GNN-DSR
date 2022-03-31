# GNN-DSR
This code is for the paper "Graph Neural Networks with Dynamic and Static Representations for Social Recommendation" which is accepted by DASFAA 2022. 

This paper proposes a PyTorch framework called GNN-DSR for social recommendation.


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
If you cannot download the documents of preprocessed data and weights, you can try to download them at https://drive.google.com/drive/folders/1Rma8Uh3vHjUuMUzHi10GUvYt49cN40Ta?usp=sharing

