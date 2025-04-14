# UNFRAMED

UNFRAMED is an unsupervised and fragment-based drug design approach for molecular optimization,  It can specifically generate new molecules with improved drug-like properties based on the input molecule using deep learning models. 

- [Overview](#Overview)
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Data Organization](#Data-Organization)
  
    - [Train_set](#Train_set)
    - [Test_set](#Test_set)
- [Training](#Training)
- [Optimizing](#Optimizing)

## Overview
UNFRAMED is a fragment-based molecular optimization framework that integrates multi-head attention-based graph models to capture complex interactions between molecular fragments and bonds. It features two core components: the Graph-based Action Prediction Model (GAPM) and the Graph-based Fragment Prediction Model (GFPM), which collaboratively guide molecular modifications. By preserving scaffold similarity, UNFRAMED maintains biological activity or drug-likeness—especially when starting from advanced drug candidates or approved drugs. Designed as an iterative and efficient optimization process, UNFRAMED consistently outperforms baseline methods across various tasks (e.g., QED and PLogP), even when trained on a single dataset.

## Requirements 
Operating systems: Ubuntu 20.04.4 LTS  
- python==3.10
- pytorch==1.12.1  
- rdkit==2023.9.6  
- dgl-cuda11.3==0.9.1

Environment Installation:
You can create the environment and install all the dependencies by running following commands:
```
conda create -n unframed python=3.10
conda activate unframed
pip install -r requirements.txt
```

## Installation
Users can download the codes by executing the command:
<pre><code>git clone https://github.com/Davezqq/UNFRAMED
</code></pre>
Downloading may take seconds to minutes according to the quality of the network.

## Data Organization

### Train_set
The data used for training was collected from ZINC, consisting of 2 files:  

Molecule_dataset(<code>src/data/train_set/Molecule_dataset.txt</code>): This txt file consists of the SMILES representation of the molecules, and each row has 1 SMILES sequence. 

Vocabulary(<code>src/data/train_set/vocabulary.txt</code>): This txt file consists of the fragments that are viewed as nodes. For each row, the first column is the SMILES of the fragment, and the second column is the time of showing up in the dataset.

### Test_set
The datasets used for testing the performance in different optimization problems were published by [Modof](https://github.com/ziqi92/Modof), there are 4 different tasks:
- Improve the DRD score(<code>src/data/test_set/drd2</code>)
- Improve the Penalized LogP score(<code>src/data/test_set/plogp</code>)
- Improve the QED score(<code>src/data/test_set/qed</code>)
- Improve the QED and DRD scores at the same time(<code>src/data/test_set/drd2_25_qed6</code>)

### Using your own dataset to train the models
If you want to use your own dataset to train GFPM and GAPM, you only need to follow these steps:
1. **Constructing a molecule smiles file of your molecules**, it should be a text file that contains multiple rows, and each row only has 1 SMILES sequence. Please noted that no comma needed at the end of each line.  
   For example:
   ```
   COCCCN1C(=O)c2cccc3c(C(=O)c4ccccc4)ccc(c23)C1=O
   Cc1nccn1-c1ccc(CNC(=O)C[C@@H]2CCc3ccccc3N2)cc1
   CC(C)Cn1c(CCCCC#N)nnc1N1CCC(OC2CCC2)CC1  
   COc1ccccc1NC(=O)C1=C(C)NC(=O)N[C@H]1c1cccc(Cl)c1
   Cc1ncc(-c2ccncc2)c([C@H]2CC[C@H](CNC(=O)C3CC3)CC2)n1
   O=C(Cc1cc(F)c(F)cc1F)Nc1nc(-c2ccc(Cl)cc2)n[nH]1
   Cc1ccc(-c2nnc(S)n2CCC(=O)N2CCC[C@@H]2C(C)(C)C)cc1
   CS(=O)(=O)c1ccc(Sc2nc3ccccc3cc2[N+](=O)[O-])cc1
   Cc1cc(C)c(C#N)c(SCc2nc3ccc(Cl)cc3c(=O)[nH]2)n1  
   ```
2. **Constructing a vocabulary based on your molecules**, you can run the <code>vocabulary.py</code> file to generate the vocabulary file, the command is as follows:
   ```python
   python vocabulary.py -d [data path] -th [threshold]
   ```
   The meaning of the arguments is as follows:
   - <code>d</code>: the path of your smiles file
   - <code>th</code>: threshold, the output vocabulary only reserves the fragments show up more than the threshold.
     
   example:
   ```python
   python vocabulary.py -d data/new_set/Molecule_dataset.txt -th 1000
   ```
  There will be two expected output files at the same directory:
  - <code>all_structures.txt</code>: This file contains all the smiles of the fragments extracted from given dataset.
  - <code>vocabulary.txt</code>: This file contains the screened fragments vocabulary based on the threshold.
  
## Training 
We designed two different models: GAPM and GFPM, both are used in the process of molecule optimization.  

### Training of GFPM
The script to train GFPM is located in <code>src/train.py</code>.
You can run the command to train GFPM as follows:  
```python
python3 train.py -ep 200 -bs 128 -nw 5 -ne 5 -feat 64 -nh 4 -nmul 4 -lr 1e-2 -data [the path of molecule data]
```
The meaning of the arguments is as follows:  
- <code>ep</code>: number of epochs  
- <code>bs</code>: the batch size of training  
- <code>nw</code>: number of workers used when loading the data  
- <code>ne</code>: number of the types of edges  
- <code>feat</code>: the dimensions of the feature vectors  
- <code>nh</code>: number of heads in RelMAL layers  
- <code>nmul</code>: number of RelMAL layers  
- <code>lr</code>: learning rate  
- <code>data</code>: the path of molecule data file

Expected output:

The trained model will be saved at <code>src/save_model</code>

### Training of GAPM
The script to train GAPM is located in <code>src/PositionModel/train.py</code>  

You can run the command to train GAPM as follows:  
```python
python3 train.py -ep 200 -bs 128 -nw 5 -ne 5 -feat 64 -nh 4 -nmul 4 -lr 1e-2 -data [the path of molecule data]
```

## Optimizing
The script for molecule optimization is located in  <code>src/batch_evaluate.py</code>   
You can run the command to optimize the molecules:  
```python
python3 batch_evaluate.py -dir [path of result files] -smi [path of test files] -ng 5 -oracle qed -gen 5 -pop 5 -sim 0.6 -clo_path [the path of pretrained GFPM] -pos_path [the path of pretrained GAPM]
```
The meaning of the arguments is as follows:
- dir: The path of result files 
- smi: The path of test files
- ng: number of parallel groups
- oracle: the property of the molecules for optimization
- gen: the number of iteration
- pop: the number of the molecules kept after each iteration
- sim: the constraint of similarity 
- clo_path: the path of pretrained GFPM model  
- pos_path: the path of pretrained GAPM model
