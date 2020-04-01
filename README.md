# an Implement of LENSR
This repository is an implement of the paper "Embedding Symbolic Knowledge into Deep Networks" (NeurIPS 2019).  
It referenced and quote a part of author's [repository].(https://github.com/ZiweiXU/LENSR)  

## Environment
Use `conda env create -f environment.yml` to creat required environment in Anaconda.  

## Run
Under `scripts` directory, using `bash train_synthetic.sh`,`bash train_vrd.sh`,`bash train_compared.sh` to train the model of LENSR in synthetic dataset, LENSR in VRD dataset, and other compared models in VRD dataset respectively.

## Dataset
Synthetic dataset is provided, but VRD dataset and Glove dataset are downloaded through `wget` at the scripts. A lot of preprocess of dataset will be executed before training.
