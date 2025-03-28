# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:25:11 2020

@author: user
"""
import copy

import torch

from torch.nn.functional import softmax

import numpy as np


from sklearn.metrics import accuracy_score, precision_score, recall_score

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from utils import Arguments

"""
def MIA_attack(target_model, shadow_model, shadow_client_loaders, shadow_test_loader, FL_params, client_loaders):
    '''


    Parameters
    ----------
    shadow_model : DNN model
        shadow model.
    shadow_client_loaders : list of Dataloader class for shadow models
        The training set of the shadow model
    shadow_test_loader : Dataloader class for shadow models
        Test sets for shadow models
    FL_params : The training parameters of federated learning
        Mainly used to read the forgotten user IDX
    client_loaders : list of Datalodaer class for standard FL models
        The training data set loader for the normal federated learning model

    Returns
    -------
    None.

    '''
    n_class_dict = dict()
    n_class_dict['adult'] = 2
    n_class_dict['purchase'] = 2
    n_class_dict['mnist'] = 10
    n_class_dict['cifar10'] = 10

    N_class = n_class_dict[FL_params.data_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shadow_model.to(device)

    shadow_model.eval()
    ####
    pred_4_mem = torch.zeros([1,N_class])
    pred_4_mem = pred_4_mem.to(device)
    with torch.no_grad():
        for ii in range(len(shadow_client_loaders)):
            if((ii == FL_params.forget_client_idx) and FL_params.mia_oldGM):
                continue
            data_loader = shadow_client_loaders[ii]

            for batch_idx, (data, target) in enumerate(data_loader):
                    data = data.to(device)
                    out = shadow_model(data)
                    pred_4_mem = torch.cat([pred_4_mem, out])
    pred_4_mem = pred_4_mem[1:,:]
    pred_4_mem = softmax(pred_4_mem,dim = 1)
    pred_4_mem = pred_4_mem.cpu()
    pred_4_mem = pred_4_mem.detach().numpy()

    ####
    pred_4_nonmem = torch.zeros([1,N_class])
    pred_4_nonmem = pred_4_nonmem.to(device)
    with torch.no_grad():
        for batch, (data, target) in enumerate(shadow_test_loader):
            data = data.to(device)
            out = shadow_model(data)
            pred_4_nonmem = torch.cat([pred_4_nonmem, out])
    pred_4_nonmem = pred_4_nonmem[1:,:]
    pred_4_nonmem = softmax(pred_4_nonmem,dim = 1)
    pred_4_nonmem = pred_4_nonmem.cpu()
    pred_4_nonmem = pred_4_nonmem.detach().numpy()

    #The predicted output of data from forgotten users on a given model
    target_model.to(device)

    target_model.eval()

    unlearn_X = torch.zeros([1,N_class])
    unlearn_X = unlearn_X.to(device)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(client_loaders[FL_params.forget_client_idx]):
                    data = data.to(device)
                    out = target_model(data)
                    unlearn_X = torch.cat([unlearn_X, out])
    unlearn_X = unlearn_X[1:,:]
    unlearn_X = softmax(unlearn_X,dim = 1)
    unlearn_X = unlearn_X.cpu().detach().numpy()

    if(FL_params.mia_oldGM):
        unlearn_y = np.ones(unlearn_X.shape[0])
        unlearn_y = unlearn_y.astype(np.int16)
    else:
        unlearn_y = np.ones(unlearn_X.shape[0])
        unlearn_y = unlearn_y.astype(np.int16)

    #Build the MIA attack model
    att_y = np.hstack((np.ones(pred_4_mem.shape[0]), np.zeros(pred_4_nonmem.shape[0])))
    att_y = att_y.astype(np.int16)

    att_X = np.vstack((pred_4_mem, pred_4_nonmem))
    X_train,X_test, y_train, y_test = train_test_split(att_X, att_y, test_size = 0.2)

    attacker = XGBClassifier(n_estimators = 100,
                              n_jobs = -1,
                              # max_depth = 10,
                              objective = 'binary:logistic',
                              booster="gblinear",
                              # learning_rate=None,
                              # tree_method = 'gpu_hist',
                              scale_pos_weight = pred_4_nonmem.shape[0]/pred_4_mem.shape[0]
                              )


    attacker.fit(X_train, y_train)

    # attacker = LogisticRegression(n_jobs = -1,class_weight='balanced')
    # attacker.fit(X_train, y_train)

    print('\n')
    print("MIA Attacker training accuracy")
    print(accuracy_score(y_train, attacker.predict(X_train)))
    print("MIA Attacker testing accuracy")
    print(accuracy_score(y_test, attacker.predict(X_test)))
    # 使用mia攻击模型攻击 forget_client_idx 被遗忘用户
    pred_y = attacker.predict(unlearn_X)
    print("MIA Attacker unlearning accuracy")
    if(FL_params.mia_oldGM == True):
        # rst = precision_score(unlearn_y, pred_y, pos_label=1)
        rst = accuracy_score(unlearn_y, pred_y)
        print(rst)
        return rst
    else:
        # rst = precision_score(unlearn_y, pred_y, pos_label=0)
        rst = accuracy_score(unlearn_y, pred_y)
        print(rst)
        return rst
"""

args = Arguments.Arg()
# 成员推理攻击
def pre_attack(client_data, client_number,test_data):
    selected_shadow_clients = copy.deepcopy((client_data))
    shadow_loaders = list()
    for s in range(len(selected_shadow_clients)):
        train_dataset = torch.utils.data.TensorDataset(client_data[s][0], client_data[s][1].long())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        shadow_loaders.append(train_loader)
    test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    return shadow_loaders, test_loader
# shadow_loaders, test_loader= pre_attack(client_data, client_number,test_data)
def attack(target_model, attack_model, malicious_client_data, test_loader,  unlearn_id):
    malicious_dataset = torch.utils.data.TensorDataset(malicious_client_data[unlearn_id][0], malicious_client_data[unlearn_id][1].long())
    malicious_loader = torch.utils.data.DataLoader(malicious_dataset, batch_size=args.batch_size, shuffle=True)

    n_class_dict = dict()
    n_class_dict['adult'] = 2
    n_class_dict['purchase'] = 2
    n_class_dict['mnist'] = 10
    n_class_dict['cifar10'] = 10

    N_class = n_class_dict['mnist']
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # target_model.to(device)

    target_model.eval()

    # The predictive output of forgotten user data after passing through the target model.
    unlearn_X = torch.zeros([1, N_class])
    # unlearn_X = unlearn_X.to(device)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(malicious_loader):
            # data = data.to(device)
            out = target_model(data)
            unlearn_X = torch.cat([unlearn_X, out])

    unlearn_X = unlearn_X[1:, :]
    unlearn_X = softmax(unlearn_X, dim=1)
    unlearn_X = unlearn_X.cpu().detach().numpy()

    unlearn_X.sort(axis=1)
    unlearn_y = np.ones(unlearn_X.shape[0])
    unlearn_y = unlearn_y.astype(np.int16)

    N_unlearn_sample = len(unlearn_y)

    # Test data, predictive output obtained after passing the target model
    test_X = torch.zeros([1, N_class])
    # test_X = test_X.to(device)
    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            # data = data.to(device)
            out = target_model(data)
            test_X = torch.cat([test_X, out])

            if (test_X.shape[0] > N_unlearn_sample):
                break
    test_X = test_X[1:N_unlearn_sample + 1, :]
    test_X = softmax(test_X, dim=1)
    test_X = test_X.cpu().detach().numpy()

    test_X.sort(axis=1)
    test_y = np.zeros(test_X.shape[0])
    test_y = test_y.astype(np.int16)

    # The data of the forgotten user passed through the output of the target model, and the data of the test set passed through the output of the target model were spliced together
    # The balanced data set that forms the 50% train 50% test.
    XX = np.vstack((unlearn_X, test_X))
    YY = np.hstack((unlearn_y, test_y))

    pred_YY = attack_model.predict(XX)
    # acc = accuracy_score( YY, pred_YY)
    pre = precision_score(YY, pred_YY, pos_label=1)
    rec = recall_score(YY, pred_YY, pos_label=1)
    f1_score = 2 * pre * rec / (pre + rec)
    # print("MIA Attacker accuracy = {:.4f}".format(acc))
    print("MIA Attacker precision = {:.4f}".format(pre))
    print("MIA Attacker recall = {:.4f}".format(rec))
    print("MIA Attacker F1 score = {:.4f}".format(f1_score))

    return (pre, rec, f1_score)

# attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)


def train_attack_model(shadow_old_GM, shadow_client_loaders, shadow_test_loader):
    shadow_model = shadow_old_GM
    n_class_dict = dict()
    n_class_dict['adult'] = 2
    n_class_dict['purchase'] = 2
    n_class_dict['mnist'] = 10
    n_class_dict['cifar10'] = 10

    N_class = n_class_dict['mnist']

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # shadow_model.to(device)

    shadow_model.eval()
    ####
    pred_4_mem = torch.zeros([1, N_class])
    # pred_4_mem = pred_4_mem.to(device)
    with torch.no_grad():
        for ii in range(len(shadow_client_loaders)):
            # if(ii != FL_params.forget_client_idx):
            #     continue
            data_loader = shadow_client_loaders[ii]

            for batch_idx, (data, target) in enumerate(data_loader):
                # data = data.to(device)
                out = shadow_model(data)
                pred_4_mem = torch.cat([pred_4_mem, out])
    pred_4_mem = pred_4_mem[1:, :]
    pred_4_mem = softmax(pred_4_mem, dim=1)
    pred_4_mem = pred_4_mem.cpu()
    pred_4_mem = pred_4_mem.detach().numpy()

    ####
    pred_4_nonmem = torch.zeros([1, N_class])
    # pred_4_nonmem = pred_4_nonmem.to(device)
    with torch.no_grad():
        for batch, (data, target) in enumerate(shadow_test_loader):
            # data = data.to(device)
            out = shadow_model(data)
            pred_4_nonmem = torch.cat([pred_4_nonmem, out])
    pred_4_nonmem = pred_4_nonmem[1:, :]
    pred_4_nonmem = softmax(pred_4_nonmem, dim=1)
    pred_4_nonmem = pred_4_nonmem.cpu()
    pred_4_nonmem = pred_4_nonmem.detach().numpy()

    # 构建MIA 攻击模型
    att_y = np.hstack((np.ones(pred_4_mem.shape[0]), np.zeros(pred_4_nonmem.shape[0])))
    att_y = att_y.astype(np.int16)

    att_X = np.vstack((pred_4_mem, pred_4_nonmem))
    att_X.sort(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(att_X, att_y, test_size=0.1)

    attacker = XGBClassifier(n_estimators=300,
                             n_jobs=-1,
                             max_depth=30,
                             objective='binary:logistic',
                             booster="gbtree",
                             # learning_rate=None,
                             # tree_method = 'gpu_hist',
                             scale_pos_weight=pred_4_nonmem.shape[0] / pred_4_mem.shape[0]
                             )

    attacker.fit(X_train, y_train)
    # print('\n')
    # print("MIA Attacker training accuracy")
    # print(accuracy_score(y_train, attacker.predict(X_train)))
    # print("MIA Attacker testing accuracy")
    # print(accuracy_score(y_test, attacker.predict(X_test)))

    return attacker











