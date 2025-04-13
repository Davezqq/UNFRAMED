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
- python==3.10.6  
- pytorch==1.12.1  
- rdkit==2022.9.1  
- dgl-cuda11.3==0.9.1  
- pandas==1.5.3

## Installation
Users can download the codes by executing the command:
<pre><code>git clone https://github.com/Davezqq/UNFRAMED
</code></pre>
Downloading may take seconds to minutes according to the quality of the network.

## Data Organization
### Train_set
The data used for training was randomly collected from ZINC, consisting of 2 files:  

Molecule_dataset(<code>src/data/train_set/Molecule_dataset.txt</code>): This txt file consists of the SMILES representation of the molecules, and each row has 1 SMILES sequence. 

Vocabulary(<code>src/data/train_set/vocabulary.txt</code>): This txt file consists of the fragments that are viewed as nodes. For each row, the first content is the SMILES of the fragment, and the second content is the time of showing up in the dataset.

### Test_set
The datasets used for testing the performance in different optimization problems were published by [Modof](https://github.com/ziqi92/Modof), there are 4 different tasks:
- Improve the DRD score(<code>src/data/test_set/drd2</code>)
- Improve the Penalized LogP score(<code>src/data/test_set/plogp</code>)
- Improve the QED score(<code>src/data/test_set/qed</code>)
- Improve the QED and DRD scores at the same time(<code>src/data/test_set/drd2_25_qed6</code>)
  
## Training
We designed two different models: GAPM and GFPM, both are used in the process of molecule optimization.   
The script to train GFPM is located in <code>src/train.py</code>  
You can run the command to train GFPM as follows:  
```python
python3 train.py -ep 200 -bs 128 -nw 5 -ne 5 -feat 64 -nh 4 -nmul 4 -lr 1e-2 -data [the path of molecule data]
```
The meaning of the arguments is as follows:  
- <code>ep</code>: number of epochs  
- bs: the batch size of training  
- nw: number of workers used when loading the data  
- ne: number of the types of edges  
- feat: the dimensions of the feature vectors  
- nh: number of heads in RelMAL layers  
- nmul: number of RelMAL layers  
- lr: learning rate  
- data: the path of molecule data file

The script to train GAPM is located in <code>src/PositionModel/train.py</code>  
You can run the command to train GFPM as follows:  
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
