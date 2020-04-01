import json
import os, sys
import argparse
import pickle as pk
from itertools import combinations_with_replacement
import gc
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from PIL import Image

sys.path.append('../../')
sys.path.append('../gcn/')
from tqdm import tqdm
from pysat import formula


from torch.utils.data import Dataset, DataLoader
from mydataloader import *

from helper import augment_bbox

preprocessed_dire = '../../dataset/VRD/'
save_dire = './saved_model/'


parser = argparse.ArgumentParser()
# parser.add_argument('--ds_name', type=str)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--filename', type=str)
parser.add_argument('--epochs', type=int, default=10)
# parser.add_argument('--logic-weight', type=float, default=0.1)
args = parser.parse_args()

preprocessed_annotation_train = {}
preprocessed_image_features_train = {}
preprocessed_annotation_test = {}
preprocessed_image_features_test = {}

info_train = {}
info_test = {}

with open(preprocessed_dire + 'preprocessed_annotation_train.pk', 'rb') as f:
    while True:
        try:
            img_name = pk.load( f)
            subSet = pk.load( f)
            preprocessed_annotation_train[img_name] = subSet
        except EOFError:
            break
with open(preprocessed_dire + 'preprocessed_image_features_train.pk', 'rb') as f:
    while True:
        try:
            img_name = pk.load( f)
            subSet = pk.load( f)
            preprocessed_image_features_train[img_name] = subSet
        except EOFError:
            break
with open(preprocessed_dire + 'preprocessed_annotation_test.pk', 'rb') as f:
    while True:
        try:
            img_name = pk.load( f)
            subSet = pk.load( f)
            preprocessed_annotation_test[img_name] = subSet
        except EOFError:
            break
with open(preprocessed_dire + 'preprocessed_image_features_test.pk', 'rb') as f:
    while True:
        try:
            img_name = pk.load( f)
            subSet = pk.load( f)
            preprocessed_image_features_test[img_name] = subSet
        except EOFError:
            break

with open(preprocessed_dire + 'info_train.pk', 'rb') as f:
    while True:
        try:
            img_name = pk.load( f)
            subSet = pk.load( f)
            info_train[img_name] = subSet
        except EOFError:
            break
with open(preprocessed_dire + 'info_test.pk', 'rb') as f:
    while True:
        try:
            img_name = pk.load( f)
            subSet = pk.load( f)
            info_test[img_name] = subSet
        except EOFError:
            break


seed_name = '.seed'+str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def remove_tensor(info):
    info_clearn = []
    for pair in range(len(info)):
        info_clearn.append([])
        for iii in range(len(info[pair])):
            if iii == 0:
                info_clearn[-1].append(info[pair][iii][0])
            else:
                info_clearn[-1].append([])
                for jjj in range(len(info[pair][iii])):
                    if jjj == 0:
                        info_clearn[-1][-1].append(int(info[pair][iii][jjj][0]))
                    else:
                        info_clearn[-1][-1].append([])
                        for kkk in range(len(info[pair][iii][jjj])):
                            info_clearn[-1][-1][-1].append(int(info[pair][iii][jjj][kkk][0]))
    return info_clearn


class Relation_Pred(nn.Module):
    def __init__(self, MLP_hidden=30, num_relations=71):
        super(Relation_Pred, self).__init__()
        self.num_relations = num_relations

        self.num_features = 512
        self.num_labelvec = 300
        self.num_latent = 512


        self.MLP = nn.Sequential(nn.Linear(self.num_features + 2 * self.num_labelvec + 8, self.num_latent),
                                 nn.ReLU(),
                                 nn.Linear(self.num_latent, self.num_relations))

    def forward(self, inputs):

        prediction = self.MLP(inputs)
        return prediction


# Training
num_epoches = args.epochs
learning_rate = 0.001
batch_size = 1
# logic_loss_weight = args.logic_weight

model = Relation_Pred().cuda()
criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_set_keys = list(preprocessed_annotation_train.keys())
test_set_keys = list(preprocessed_annotation_test.keys())

train_set = VRD_dataset(train_set_keys, preprocessed_image_features_train, preprocessed_annotation_train, info_train)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = VRD_dataset_test(test_set_keys, preprocessed_image_features_test, preprocessed_annotation_test, info_test)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


# test dataset
def run_test(model_test, k=5):
    model_test.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        avg_loss = None
        for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x, y, info = batch
            x = x.squeeze(0).cuda()
            y = y.squeeze(0).cuda()
            info = remove_tensor(info)

            x = augment_bbox(x, info)

            prediction = model_test(x)
            loss = F.nll_loss(F.log_softmax(prediction,dim=1), y)
            if avg_loss is None:
                avg_loss = loss
            else:
                avg_loss += loss

            # _, predicted = torch.max(prediction.data, 1)
            _, predicted = torch.topk(prediction.data, k)
            pred = predicted.t()
            correct_num = pred.eq(y.view(1, -1).expand_as(pred))
            correct_k = correct_num[:k].view(-1).float().sum(0, keepdim=True)

            total += y.size(0)
            correct += correct_k[0]
        avg_loss = avg_loss / i
        acc = correct / total
        return avg_loss, acc


loss_save = {}
loss_save['train_avgloss_all'] = []
loss_save['train_avgloss_ce'] = []
loss_save['test_avgloss'] = []
loss_save['train_acc'] = []
loss_save['test_acc'] = []

loss_by_iter = []
celoss_by_iter = []

model.train()
best_acc = 0
for iter in range(num_epoches):
    print('\n Iteration: ', iter)
    # print(args.filename)
    correct = 0
    total = 0
    avg_loss_all = None
    avg_loss_ce = None
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        gc.collect()
        x, y, info = batch
        x = x.squeeze(0).cuda()
        y = y.squeeze(0).cuda()
        info = remove_tensor(info)

        x = augment_bbox(x, info)

        prediction = model(x)
        #print(prediction.dim())
        #exit(0)
        loss_entropy = F.nll_loss(F.log_softmax(prediction,dim=1), y)
        loss = loss_entropy


        loss_by_iter.append(float(loss))
        celoss_by_iter.append(float(loss_entropy))

        # calcuate the avg loss and accuracy
        if avg_loss_all is None:
            avg_loss_all = loss
            avg_loss_ce = loss_entropy
        else:
            avg_loss_all += loss
            avg_loss_ce += loss_entropy

        _, predicted = torch.max(prediction.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss_all = avg_loss_all / i
    avg_loss_ce = avg_loss_ce / i
    acc = correct / total
    loss_save['train_avgloss_all'].append(float(avg_loss_all))
    loss_save['train_avgloss_ce'].append(float(avg_loss_ce))
    loss_save['train_acc'].append(float(acc))
    print('Train Acc: {:0.4f} '.format(acc))
    print('Train AvgLoss_ALL: {:0.4f}'.format(avg_loss_all))
    print('Train AvgLoss_CE: {:0.4f}'.format(avg_loss_ce))

    if not os.path.exists(save_dire):
        os.mkdir(save_dire)

    # test model for this epoch
    test_avg_loss, test_acc = run_test(model)
    if test_acc > best_acc:
        best_acc = test_acc
        # save model for this epoch
        torch.save(model, save_dire + "MLP{}.best".format(seed_name))
    torch.save(model, save_dire + "MLP{}.latest".format(seed_name))
    
    loss_save['test_avgloss'].append(float(test_avg_loss))
    loss_save['test_acc'].append(float(test_acc))
    print('Test Acc: {:0.4f} '.format(test_acc) )
    print('Test AvgLoss_CE: {:0.4f}'.format(test_avg_loss))
    test_avg_loss, test_acc = run_test(model,k=1)
    print('Test Acc: {:0.4f} k=1'.format(test_acc) )
    print('Test AvgLoss_CE: {:0.4f} k=1'.format(test_avg_loss))

    # re-write the file
    # json.dump(loss_save, open(save_dire + 'loss_save', 'w'), ensure_ascii=False)
    if not os.path.exists('./acc_loss/'):
        os.mkdir('./acc_loss/')
    json.dump(loss_save, open('./acc_loss/' + "MLP{}.best".format(seed_name), 'w'),
              ensure_ascii=False)
    json.dump(loss_by_iter, open('./acc_loss/' + "MLP{}.best".format(seed_name) + '.loss_by_iter', 'w'),
              ensure_ascii=False)
    json.dump(celoss_by_iter,
              open('./acc_loss/' + "MLP{}.best".format(seed_name)+ '.celoss_by_iter','w'),
              ensure_ascii=False)
print(f"Best test acc: {max(loss_save['test_acc'])}, at epoch: {np.argmax(loss_save['test_acc'])}")
# load model to do the testing
model_test = torch.load(save_dire + "MLP{}.best".format(seed_name))
model_test = model_test.cuda()
test_avg_loss, test_acc = run_test(model_test)
print('The final test avgloss: {:0.4f}; final test acc is: {:0.4f} k=5'.format(test_avg_loss, test_acc))
test_avg_loss, test_acc = run_test(model_test,k=1)
print('The final test avgloss: {:0.4f}; final test acc is: {:0.4f} k=1'.format(test_avg_loss, test_acc))
