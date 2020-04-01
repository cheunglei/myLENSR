#!/usr/bin/env bash
set -e
source activate LENSR

ds_names=("0303" "0306" "0606")

###### General form to CNF, d-DNNF #####
cd ../tools
for ds_name in ${ds_names[@]}; do
    python raw2data.py --save_path ../dataset/Synthetic --ds_name "${ds_name}"
done
cd -

###### CNF, d-DNNF to GCN compatible #####
cd ../tools
for ds_name in ${ds_names[@]}; do
    python cnf2data.py --save_path ../dataset/Synthetic --ds_name "${ds_name}"
    python ddnnf2data.py --save_path ../dataset/Synthetic --ds_name "${ds_name}"
done
cd -

###### Train Synthetic dataset #####
cd ../model/gcn

atom_options=("_0606" "_0306" "_0303" )
dataset_options=( "ddnnf"  )
seeds=(  "80" "81" "82" "83" "84" )
atom_options=("_0606" "_0306" "_0303"  )
dataset_options=( "general"   )
seeds=( "8"  )



ind_options="--indep_weight"
reg_options="--w_reg 0.1"
non_reg_options="--w_reg 0.0"
#cls_options="--cls_reg 0.1"

for seed in ${seeds[@]}; do
    for dataset in ${dataset_options[@]}; do
        for atom in "${atom_options[@]}"; do
            if [[ ${dataset} == 'general' ]]; then
                python train.py --ds_path ../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${non_reg_options} ${ind_options} --seed ${seed}
		        python train.py --ds_path ../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${non_reg_options}  --seed ${seed}
            fi
            if [[ ${dataset} == 'cnf' ]]; then
                python train.py --ds_path ../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${non_reg_options} ${ind_options} --seed ${seed}
                python train.py --ds_path ../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${non_reg_options} --seed ${seed}
            fi
            if [[ ${dataset} == 'ddnnf' ]]; then
                python train.py --ds_path ../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${reg_options} ${ind_options} --seed ${seed}
                python train.py --ds_path ../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${reg_options} --seed ${seed}
                python train.py --ds_path ../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${non_reg_options} ${ind_options} --seed ${seed}
                python train.py --ds_path ../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 0 ${non_reg_options} --seed ${seed}
            fi
        done
    done
done
cd -
