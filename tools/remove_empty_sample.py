import numpy as np
import pickle as pk
import sys

sys.path.append('.')
sys.path.append('../')
import os
import json

preprocessed_dire = '../dataset/VRD/'
preprocessed_annotation_train_raw = {}
preprocessed_annotation_test_raw = {}
clearn_train = 0
clearn_test = 0

# with open(preprocessed_dire + 'preprocessed_annotation_train.pk', 'rb') as f:
#     while True:
#         try:
#             img_name = pk.load(f)
#             subSet = pk.load(f)
#             #preprocessed_annotation_train_raw[img_name] = subSet
#             if subSet != {} :
#                 clearn_train += 1
#                 with open('../dataset/VRD/preprocessed_annotation_train_clean.pk', 'ab') as f2:
#                     pk.dump(img_name, f2)
#                     pk.dump(subSet, f2)
#         except EOFError:
#             break

with open(preprocessed_dire + 'preprocessed_annotation_test.pk', 'rb') as f:
    while True:
        try:
            img_name = pk.load(f)
            subSet = pk.load(f)
            #preprocessed_annotation_train_raw[img_name] = subSet
            if subSet != {} :
                clearn_test += 1
                with open('../dataset/VRD/preprocessed_annotation_test_clean.pk', 'ab') as f2:
                    pk.dump(img_name, f2)
                    pk.dump(subSet, f2)
        except EOFError:
            break

print(clearn_train,clearn_test)

#  clean annotations_train/test.json
## train
# annotation = json.load(open('../dataset/VRD/annotations_train.json'))
# directory_in_str = '../dataset/VRD/sg_dataset/sg_train_images'
# newAnnotation = annotation.copy()
# for key in annotation.keys():
#     if key not in os.listdir(directory_in_str):
#         del newAnnotation[key]
# print(len(newAnnotation))
# json.dump(newAnnotation,open('../dataset/VRD/annotations_train_clean.json','w'))
# os.system('rm ' + preprocessed_dire + 'preprocessed_annotation_train.pk')
# os.system('mv ' + '../dataset/VRD/preprocessed_annotation_train_clean.pk'  + ' '
#           + preprocessed_dire + 'preprocessed_annotation_train.pk')

## test
annotation = json.load(open('../dataset/VRD/annotations_test.json'))
directory_in_str = '../dataset/VRD/sg_dataset/sg_test_images'
newAnnotation = annotation.copy()
for key in annotation.keys():
    if key not in os.listdir(directory_in_str):
        del newAnnotation[key]
print(len(newAnnotation))
json.dump(newAnnotation,open('../dataset/VRD/annotations_test_clean.json','w'))

os.system('rm ' + preprocessed_dire + 'preprocessed_annotation_test.pk')
os.system('mv ' + '../dataset/VRD/preprocessed_annotation_test_clean.pk' + ' '
          + preprocessed_dire + 'preprocessed_annotation_test.pk')