import os, sys, argparse
sys.path.append('../')
import numpy as np
import pickle as pk

from model.Misc.Formula import dimacs_to_cnf
from tqdm import tqdm

def write_data(input_file, output_file, features):

    cnf, _ = dimacs_to_cnf(input_file)

    variables = open(output_file[0], 'w')
    relations = open(output_file[1], 'w')

    relations_str = ''
    variables_str = ''

    feature_leaf = {}

    for i in range(
            len(DIGIT_TO_SYM)):# DIGIT_TO_SYM = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
        feature_leaf[str(i)] = features[DIGIT_TO_SYM[i]]
        feature_leaf[str(-i)] = -features[DIGIT_TO_SYM[i]]


    # add global var
    feature = features['Global']
    label = 0
    variables_str += str(0) + '\t'
    variables_str += '\t'.join(list(map(str, feature))) + '\t'
    variables_str += str(label) + '\n'

    # record known vars
    if '.s' not in input_file: #the formula
        known_var = {}
        var_id = 1
        and_vars = []
        or_vars = []
        or_children = {}
        for c in cnf:
            or_vars.append(var_id)
            or_children[var_id] = []
            variables_str += (
                        str(var_id) + '\t' + '\t'.join(list(map(str, features['Or']))) + '\t' + str(type_map['Or']) + '\n')
            current_or = var_id
            var_id += 1
            for l in c:
                if l not in known_var.keys():
                    known_var[l] = var_id
                    variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, feature_leaf[str(l)]))) + '\t' + str(
                        type_map['Symbol']) + '\n')
                    this_var_id = var_id
                    var_id += 1
                else:
                    this_var_id = known_var[l]
                or_children[current_or].append(this_var_id)
        and_vars.append(var_id)
        variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, features['And']))) + '\t' + str(
                        type_map['And']) + '\n')
        var_id += 1

        for i in range(1, var_id):
            relations_str += (str(i) + '\t' + '0\n')
        for or_var in or_vars:
            for or_child in or_children[or_var]:
                relations_str += (str(or_child) + '\t' + str(or_var) + '\n')
        for or_var in or_vars:
            relations_str += (str(or_var) + '\t' + str(and_vars[0]) + '\n')
    else: #the assignment
        var_id = 1
        pseudo_clause = [i[0] for i in cnf]
        for l in pseudo_clause:
            variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, feature_leaf[str(l)]))) + '\t' + str(
                type_map['Symbol']) + '\n')
            var_id += 1
        variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, features['And']))) + '\t' + str(
                        type_map['And']) + '\n')
        var_id += 1

        for i in range(1, var_id):
            relations_str += (str(i) + '\t'+ str(0) + '\n')
        for i in range(1, var_id-1):
            relations_str += (str(i) + '\t' + str(var_id-1) + '\n')

    relations.write(relations_str)
    variables.write(variables_str)
    relations.close()
    variables.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str,default='../dataset/Synthetic')
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--source_path', type=str, default='../model/gcn/features.pk')
    args = parser.parse_args()

    source_path = args.source_path
    ds_name = args.ds_name
    save_path = args.save_path

    d = pk.load(open(source_path, 'rb'))
    DIGIT_TO_SYM, features, type_map = d['digit_to_sym'], d['features'], d['type_map']
    DIGIT_TO_SYM = DIGIT_TO_SYM[1:]

    directory_in_str = f'{save_path}/cnf_{ds_name}_raw/'
    directory_in_str_out = f'{save_path}/cnf_{ds_name}/'
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
