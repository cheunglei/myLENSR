#!/usr/bin/env bash
source activate LENSR

###### Download & Preprocess glove ######
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
python glove.py

mv glove.6B.50d.idx.pkl ../dataset/glove/
mv glove.6B.50d.words.pkl ../dataset/glove/
mv glove.6B.50d.dat ../dataset/glove/glove.6B.50d.dat

rm glove.6B.zip
rm *.txt

###### Download VRD dataset ######
wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
unzip sg_dataset.zip
mv sg_dataset/sg_train_images ../dataset/VRD/sg_dataset/sg_train_images
mv sg_dataset/sg_test_images ../dataset/VRD/sg_dataset/sg_test_images
cd ../tools
python preprocess_image.py
echo finish preprocess_image.py
python remove_empty_sample.py
echo finish remove_empty_sample.py
cd -
rm -rf sg_dataset
rm sg_dataset.zip

###### Create neccessary data ######
cd ../tools
python find_rels.py
python tokenize_vocabs.py
python rel2cnf.py
python cnf2ddnnf.py --ds_name vrd --save_path ../dataset/VRD
python relcnf2data.py
python relddnnf2data.py
cd -

###### Train embedder ######
cd ../model/gcn

atom_options=("" "_ddnnf")
dataset_options=("vrd")
seed="0"
ind_options="--indep_weight"
reg_options="--w_reg 0.1"
non_reg_options="--w_reg 0.0"

for dataset in ${dataset_options[@]}; do
    for atom in "${atom_options[@]}"; do
        if [[ "${atom}" == "" ]]; then
            python train.py --ds_path ../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${non_reg_options} ${ind_options} --seed ${seed}
            python train.py --ds_path ../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${non_reg_options} --seed ${seed}
        fi
        if [[ ${atom} == '_ddnnf' ]]; then
            python train.py --ds_path ../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${reg_options} ${ind_options} --seed ${seed}
            python train.py --ds_path ../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${reg_options} --seed ${seed}
            python train.py --ds_path ../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${non_reg_options} ${ind_options} --seed ${seed}
            python train.py --ds_path ../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${non_reg_options} --seed ${seed}
        fi
    done
done
cd -

###### Train relation predictor ######
cd ../model/relation_prediction_LENSR

atom_options=("" "_ddnnf")
dataset_options=("vrd")

ind_options=".ind"
reg_options=".reg0.1"
non_reg_options=".reg0.0"
cls_reg_options=".cls0.1"
seed_options=".seed0"

for dataset in ${dataset_options[@]}; do
    for atom in "${atom_options[@]}"; do
        if [[ "${atom}" == "" ]]; then
            python train.py --epochs 10 --filename "${dataset}${atom}${non_reg_options}${cls_reg_options}${seed_options}.model" --ds_name vrd
            python train.py --epochs 10 --filename "${dataset}${atom}${non_reg_options}${ind_options}${cls_reg_options}${seed_options}.model" --ds_name vrd
        fi
        if [[ ${atom} == '_ddnnf' ]]; then
            python train.py --epochs 10 --filename "${dataset}${atom}${non_reg_options}${cls_reg_options}${seed_options}.model" --ds_name vrd_ddnnf
            python train.py --epochs 10 --filename "${dataset}${atom}${reg_options}${cls_reg_options}${seed_options}.model" --ds_name vrd_ddnnf
			python train.py --epochs 10 --filename "${dataset}${atom}${non_reg_options}${ind_options}${cls_reg_options}${seed_options}.model" --ds_name vrd_ddnnf
            python train.py --epochs 10 --filename "${dataset}${atom}${reg_options}${ind_options}${cls_reg_options}${seed_options}.model" --ds_name vrd_ddnnf
        fi
    done
done
cd -

