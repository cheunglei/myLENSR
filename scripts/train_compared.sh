#!/usr/bin/env bash
source activate LENSR



###### no_logic ######
cd ../model/relation_prediction_no_logic
#python train.py --epochs 10 --seed 0
cd -

###### semantic_loss ######
semantic_loss_weight="0.0005"
cd ../model/relation_prediction_semantic_loss
#python train.py --epochs 10 --seed 0 --semantic_loss_weight ${semantic_loss_weight}
cd -

###### tree-lstm ######
###### Train embedder ######
cd ../model/tree_lstm

#atom_options=("" "_ddnnf")
atom_options=( "_ddnnf")
dataset_options=("vrd")
seed="0"

#for dataset in ${dataset_options[@]}; do
#    for atom in "${atom_options[@]}"; do
#        if [[ "${atom}" == "" ]]; then
#            python train.py --ds_path ../../dataset/VRD --dataset ${dataset}${atom} --epochs 1 --dataloader_worker 0  --seed ${seed}
#        fi
#        if [[ ${atom} == '_ddnnf' ]]; then
#            python train.py --ds_path ../../dataset/VRD --dataset ${dataset}${atom} --epochs 1 --dataloader_worker 0  --seed ${seed}
#        fi
#    done
#done
cd -

###### Train relation predictor ######
cd ../model/relation_prediction_tree_lstm

atom_options=("" "_ddnnf")
dataset_options=("vrd")
cls_reg_options=".cls0.1"
seed_options=".seed0"
directed_options=".dir"
embedder_path="../tree_lstm/model_save/"

for dataset in ${dataset_options[@]}; do
    for atom in "${atom_options[@]}"; do
        if [[ "${atom}" == "" ]]; then
            python train.py --epochs 10 --filename "${dataset}${atom}${directed_options}${cls_reg_options}${seed_options}.model" --ds_name vrd --embedder_path ${embedder_path}
        fi
        if [[ ${atom} == '_ddnnf' ]]; then
            python train.py --epochs 10 --filename "${dataset}${atom}${directed_options}${cls_reg_options}${seed_options}.model" --ds_name vrd_ddnnf --embedder_path ${embedder_path}
        fi
    done
done
cd -

