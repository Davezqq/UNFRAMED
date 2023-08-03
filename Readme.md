# FOMO:A Fragment-based Objective Molecule Optimization Framework

This is the official code for FOMO

We are cleaning the code and will release it soon.
## Requirements 
Operating systems: Ubuntu 20.04.4 LTS  
- python==3.10.6  
- pytorch==1.12.1  
- rdkit==2022.9.1  
- dgl-cuda11.3==0.9.1  
- pandas==1.5.3

## Installation
Users can download the codes by executing the command:
<pre><code>git clone https://github.com/Davezqq/FOMO
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

