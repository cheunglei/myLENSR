import os, argparse
import json
import pickle as pk

from tqdm import tqdm

def write_data(input_file, output_file, features):
    ddnnf = open(input_file, 'r')
    variables = open(output_file[0], 'w')
    relations = open(output_file[1], 'w')

    and_children = open(output_file[2], 'w')
    or_children = open(output_file[3], 'w')

    relations_str = ''
    variables_str = ''

    and_children_list = []
    or_children_list = []

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

    line_num = 1
    for line in ddnnf.readlines():
        # filter the first line
        if line_num == 1:
            continue

        line = line.split()
        type = line[0]
        children = line[1:]
        if type == 'L': # leaf node
            feature = feature_leaf[children[0]]
            label = type_map['Symbol']
        elif type == 'O': # OR node
            feature = features['Or']
            label = type_map['Or']
            or_children_list.append([])
            # for child in children[1:]:
            for child in children[2:]:
                child = int(child)
                or_children_list[-1].append(child + 1)
                relations_str += str(child + 1) + '\t' + str(line_num) + '\n'
        elif type == 'A': # AND node
            feature = features['And']
            label = type_map['And']
            and_children_list.append([])
            for child in children[1:]:
                child = int(child)
                and_children_list[-1].append(child + 1)
                relations_str += str(child + 1) + '\t' + str(line_num) + '\n'

        variables_str += str(line_num) + '\t'
        variables_str += '\t'.join(list(map(str, feature))) + '\t'
        variables_str += str(label) + '\n'

        line_num += 1

    # add edge for global variable
    for j in range(line_num):
        relations_str += str(j + 1) + '\t' + str(0) + '\n'

    relations.write(relations_str)
    variables.write(variables_str)
    relations.close()
    variables.close()

    json.dump(and_children_list, and_children)
    json.dump(or_children_list, or_children)


if __name__ == '__main__':
    features = pk.load(open('../model/pygcn/pygcn/features.pk', 'rb'))['features']
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

    directory_in_str = f'{save_path}/ddnnf_{ds_name}_raw/'
    directory_in_str_out = f'{save_path}/ddnnf_{ds_name}/'
    if not os.path.exists(directory_in_str_out):
        os.mkdir(directory_in_str_out)

    for file in tqdm(os.listdir(directory_in_str)):
        if file.endswith(".nnf"):
            input_dire = os.path.join(directory_in_str, file)
            output_dire = [os.path.join(directory_in_str_out, file[:-4] + '.var'),
                           os.path.join(directory_in_str_out, file[:-4] + '.rel'),
                           os.path.join(directory_in_str_out, file[:-4] + '.and'),
                           os.path.join(directory_in_str_out, file[:-4] + '.or')]

            write_data(input_dire, output_dire, features)
