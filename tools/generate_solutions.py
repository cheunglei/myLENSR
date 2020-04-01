import re, sys, os, argparse
sys.path.append('../')
import argparse

import numpy as np
import pickle as pk

from tqdm import tqdm
from sympy import sympify, Symbol
from sympy.abc import A, B, C, D, E, F, G, H, I, J, K, L
from sympy.logic import And, Or, Not
from sympy.logic import to_cnf, to_nnf
from pysat import formula

from model.Misc.Formula import find, cnf_to_dimacs, dimacs_to_nnf,dimacs_to_cnf


def write_data(input_file, output_file, features):

    cnf, _ = dimacs_to_cnf(input_file)

    variables = open(output_file[0], 'w')
    relations = open(output_file[1], 'w')

    relations_str = ''
    variables_str = ''

    and_children_list = []
    or_children_list = []

    num_feature = 50

    feature_OR = features['Or']
    feature_AND = features['And']
    feature_G = features['Global']
    # feature_leaf: [1,2,3]
    feature_leaf = {'1': features['a'], '2': features['b'],
                    '3': features['c'], '4': features['d'],
                    '5': features['e'], '6': features['f'],
                    '7': features['g'], '8': features['h'],
                    '9': features['i'], '10': features['j'],
                    '11': features['k'], '12': features['l']
                    }
    feature_leaf['-1'] = -feature_leaf['1']
    feature_leaf['-2'] = -feature_leaf['2']
    feature_leaf['-3'] = -feature_leaf['3']
    feature_leaf['-4'] = -feature_leaf['4']
    feature_leaf['-5'] = -feature_leaf['5']
    feature_leaf['-6'] = -feature_leaf['6']
    feature_leaf['-7'] = -feature_leaf['7']
    feature_leaf['-8'] = -feature_leaf['8']
    feature_leaf['-9'] = -feature_leaf['9']
    feature_leaf['-10'] = -feature_leaf['10']
    feature_leaf['-11'] = -feature_leaf['11']
    feature_leaf['-12'] = -feature_leaf['12']


    # add global var
    feature = feature_G
    label = 0
    variables_str += str(0) + '\t'
    for j in range(len(feature)):
        variables_str += str(feature[j]) + '\t'
    variables_str += str(label) + '\n'

    var_id = 1
    pseudo_clause = [i[0] for i in cnf]
    for l in pseudo_clause:
        variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, feature_leaf[str(l)]))) + '\t1\n')
        var_id += 1
    variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, feature_AND))) + '\t3\n')
    var_id += 1

    for i in range(1, var_id):
        relations_str += (str(i) + '\t' + '0\n')
    for i in range(1, var_id-1):
        relations_str += (str(i) + '\t' + str(var_id-1) + '\n')

    relations.write(relations_str)
    variables.write(variables_str)
    relations.close()
    variables.close()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='../dataset/Synthetic')
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--atoms', type=int, default=None)
    args = parser.parse_args()
    if args.atoms is None:
        atoms = int(args.ds_name[0:2])
    else:
        atoms = args.atoms
    num_vars = 2**atoms
	
    
    d = pk.load(open('../model/pygcn/pygcn/features.pk', 'rb'))
    _, features, type_map = d['digit_to_sym'], d['features'], d['type_map']
    SAVE_PATH = args.save_path
    DIGIT_TO_SYM = [None, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']## DIGIT_TO_SYM == d['digit_to_sym']
    ds_name = args.ds_name

    if not os.path.exists(f'{SAVE_PATH}/sol_cnf_{ds_name}_raw/'):
        os.mkdir(f'{SAVE_PATH}/sol_cnf_{ds_name}_raw/')
    if not os.path.exists(f'{SAVE_PATH}/sol_ddnnf_{ds_name}_raw/'):
        os.mkdir(f'{SAVE_PATH}/sol_ddnnf_{ds_name}_raw/')

    symbol = []
    for i in range(atoms):
        symbol.append(0)
    for n in range(num_vars):
        sloution = []
        a = n
        ##确定符号，即将n二进制化
        for i in range(atoms):
            symbol[atoms-i-1] = a%2
            a = int(a/2)
        for i in range(atoms):
            if symbol[i] == 0:
                sloution.append(-(i+1))
            else:
                sloution.append(i+1)
        cnf_to_dimacs(f'{SAVE_PATH}/sol_cnf_{ds_name}_raw/{n}.cnf', [[i] for i in sloution], atoms)
        dimacs_to_nnf(f'{SAVE_PATH}/sol_cnf_{ds_name}_raw/{n}.cnf',
                      f'{SAVE_PATH}/sol_ddnnf_{ds_name}_raw/{n}.nnf',
                      '../c2d_linux')
    ##cnf2data
    save_path = SAVE_PATH

    directory_in_str = f'{save_path}/sol_cnf_{ds_name}_raw/'
    directory_in_str_out = f'{save_path}/sol_cnf_{ds_name}/'
    if not os.path.exists(directory_in_str_out):
        os.mkdir(directory_in_str_out)

    for file in tqdm(os.listdir(directory_in_str)):
        if file.endswith(".cnf"):
            input_dire = os.path.join(directory_in_str, file)
            output_dire = [os.path.join(directory_in_str_out, file[:-4] + '.var'),
                           os.path.join(directory_in_str_out, file[:-4] + '.rel'),
                           os.path.join(directory_in_str_out, file[:-4] + '.and'),
                           os.path.join(directory_in_str_out, file[:-4] + '.or')]

            write_data(input_dire, output_dire, features)

