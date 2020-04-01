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

from model.Misc.Formula import find, cnf_to_dimacs, dimacs_to_nnf

"""
This script generates three things: 
(1) the pygcn compatible data file for GENERAL form 
(2) CNF raw file
(3) DDNNF raw file
"""

def get_clauses(cnf_form):
    def _get_clauses(expr) -> tuple:
        if not isinstance(expr, And):
            return expr
        return expr.args

    atoms = [None] + sorted(
        list(map(str, list(cnf_form.atoms()))),key=lambda x: DIGIT_TO_SYM.index(x))
    clauses_tuple = _get_clauses(cnf_form)
    clauses = []
    for c in clauses_tuple:
        if type(c) == Not:
            clauses.append([-atoms.index(str(c)[1])])
        elif type(c) == Symbol:
            clauses.append([atoms.index(str(c))])
        elif type(c) == Or:
            clauses.append([atoms.index(str(i)) if str(i)[0] != '~' else -atoms.index(str(i)[1]) for i in c.args])
        else:
            raise ValueError(f'Unable to handle {str(c)}')

    return clauses, atoms


class node:
    def __init__(self):
        self.data = None
        self.children = []
        self.id = None

    def __repr__(self):
            return '(Node_id: ' + str(self.id) + '; Data: ' + str(self.data) + '; Children: ' + str(self.children) + ')'

def make_graph(fml, features, type_map):
    relations = []
    variables = []
    # global node
    variables.append([0] + list(features['Global']) + [type_map['Global']])
    identified_var = [None]
    root = node()

    def build_var(root, node):
        node.data = root
        type = type(root)

        if type == Symbol:
            if str(root) not in identified_var:
                node.id = len(identified_var)
                identified_var.append(str(root))
                variables.append([node.id] + list(features[str(root)]) + [type_map['Symbol']])
            else:
                node.id = identified_var.index(str(root))

            return
        else:  ## type == Or/And/Not
            node.id = len(identified_var)
            identified_var.append(str(root))
            variables.append([node.id] + list(features[str(type(root))]) + [type_map[str(type(root))]])
            node.children = [node() for i in range(len(root.args))]

            for i in range(len(node.children)):
                build_var(root.args[i], node.children[i])


    def build_rel(node):
        for i in range(len(node.children)):
            relations.append()
            relations.append([node.children[i].id, node.id])
        for i in range(len(node.children)):
            build_rel(node.children[i])

    build_var(fml,root)
    build_rel(root)
    # global node relation
    for i in range(1, len(variables)):
        relations.append([i, 0])

    return variables, relations, identified_var, root

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str,default='../dataset/Synthetic')
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--source_path',type=str,default='../model/gcn/features.pk')
    args = parser.parse_args()

    source_path = args.source_path
    save_path = args.save_path
    ds_name = args.ds_name


    d = pk.load(open(source_path, 'rb'))
    DIGIT_TO_SYM, features, type_map = d['digit_to_sym'], d['features'], d['type_map']

    if not os.path.exists(f'{save_path}/cnf_{ds_name}_raw/'):
        os.mkdir(f'{save_path}/cnf_{ds_name}_raw/')
    if not os.path.exists(f'{save_path}/ddnnf_{ds_name}_raw/'):
        os.mkdir(f'{save_path}/ddnnf_{ds_name}_raw/')
    if not os.path.exists(f'{save_path}/general_{ds_name}/'):
        os.mkdir(f'{save_path}/general_{ds_name}/')

    fml_strings = pk.load(open(f'{save_path}/formula_strings_{ds_name}.pk', 'rb'))
    for fml_idx in tqdm(range(len(fml_strings))):
        ##Converts an arbitrary expression to a type that can be used inside SymPy.
        this_fml = sympify(fml_strings[fml_idx], evaluate=False)
        clauses, atom_mapping = get_clauses(to_cnf(this_fml))##clauses 是以数字形式表示的，e.g.[[1], [3], [1, 2], [1, 3]]
        # construct CNF formula
        f = formula.CNF()
        for c in clauses: f.append(c)
        # find truth and false assignments of the CNF formula
        st, sf = find(f, 5, assumptions=[])
        if len(st) < 1:
            continue

        variables, relations, identified_var, root = make_graph(this_fml, features=features, type_map=type_map)



        with open(f'{save_path}/general_{ds_name}/{fml_idx}' + '.var', 'w') as f:
            for var in variables:
                f.write('\t'.join(list(map(str, var))))
                f.write('\n')
        with open(f'{save_path}/general_{ds_name}/{fml_idx}' + '.rel', 'w') as f:
            for rel in relations:
                f.write('\t'.join(list(map(str, rel))))
                f.write('\n')

        # write cnf for fml
        cnf_to_dimacs(f'{save_path}/cnf_{ds_name}_raw/{fml_idx}.cnf', clauses, len(atom_mapping) - 1)
        dimacs_to_nnf(f'{save_path}/cnf_{ds_name}_raw/{fml_idx}.cnf',
                      f'{save_path}/ddnnf_{ds_name}_raw/{fml_idx}.nnf',
                      '../c2d_linux')
        # write truth assignments
        for ii, tt in enumerate(st):
            tt_sym = ['~' + atom_mapping[abs(i)] if i < 0 else atom_mapping[abs(i)] for i in tt]
            tt_variables = []
            tt_relations = []
            tt_variables.append([0] + list(features['Global']) + [type_map['Global']])
            tt_variables.append([1] + list(features['And']) + [type_map['And']])
            idx = 2
            for tt_sym_i in tt_sym:
                if tt_sym_i[0] == '~':
                    tt_variables.append([idx] + list((-1) * features[tt_sym_i[1]]) + [type_map['Symbol']])
                else:
                    tt_variables.append([idx] + list(features[tt_sym_i]) + [type_map['Symbol']])
                tt_relations.append([idx, 1])
                idx += 1
            for i in range(1, idx):
                tt_relations.append([i, 0])
            with open(f'{save_path}/general_{ds_name}/{fml_idx}.st{ii}.var', 'w') as f:
                for var in tt_variables:
                    f.write('\t'.join(list(map(str, var))))
                    f.write('\n')
            with open(f'{save_path}/general_{ds_name}/{fml_idx}.st{ii}.rel', 'w') as f:
                for rel in tt_relations:
                    f.write('\t'.join(list(map(str, rel))))
                    f.write('\n')
            # create raw cnf file
            cnf_to_dimacs(f'{save_path}/cnf_{ds_name}_raw/{fml_idx}.st{ii}.cnf', [[i] for i in tt], len(atom_mapping) - 1)
            dimacs_to_nnf(f'{save_path}/cnf_{ds_name}_raw/{fml_idx}.st{ii}.cnf',
                          f'{save_path}/ddnnf_{ds_name}_raw/{fml_idx}.st{ii}.nnf',
                          '../c2d_linux')
        # write false assignments
        for ii, tt in enumerate(sf):
            tt_sym = ['~' + atom_mapping[abs(i)] if i < 0 else atom_mapping[abs(i)] for i in tt]
            tt_variables = []
            tt_relations = []
            tt_variables.append([0] + list(features['Global']) + [type_map['Global']])
            tt_variables.append([1] + list(features['And']) + [type_map['And']])
            idx = 2
            for tt_sym_i in tt_sym:
                if tt_sym_i[0] == '~':
                    tt_variables.append([idx] + list((-1) * features[tt_sym_i[1]]) + [type_map['Symbol']])
                else:
                    tt_variables.append([idx] + list(features[tt_sym_i]) + [type_map['Symbol']])
                tt_relations.append([idx, 1])
                idx += 1
            for i in range(1, idx):
                tt_relations.append([i, 0])
            with open(f'{save_path}/general_{ds_name}/{fml_idx}.sf{ii}.var', 'w') as f:
                for var in tt_variables:
                    f.write('\t'.join(list(map(str, var))))
                    f.write('\n')
            with open(f'{save_path}/general_{ds_name}/{fml_idx}.sf{ii}.rel', 'w') as f:
                for rel in tt_relations:
                    f.write('\t'.join(list(map(str, rel))))
                    f.write('\n')
            # create raw cnf file
            cnf_to_dimacs(f'{save_path}/cnf_{ds_name}_raw/{fml_idx}.sf{ii}.cnf', [[i] for i in tt], len(atom_mapping) - 1)
            dimacs_to_nnf(f'{save_path}/cnf_{ds_name}_raw/{fml_idx}.sf{ii}.cnf',
                          f'{save_path}/ddnnf_{ds_name}_raw/{fml_idx}.sf{ii}.nnf',
                          '../c2d_linux')
