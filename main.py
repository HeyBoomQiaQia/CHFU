import copy
import random

from MIA import pre_attack, attack, train_attack_model

from data import datapro
from learning import federated_learning, test, backdoor_attack_train, backdoor_attack_test,  collaborate_hier_federated_learning, \
    backdoor_attack_test_copy, mi_ch_federated_learning
from Model import m_LeNet, CNNCifar, m_MLP
from utils import Arguments
from datadistri import client_create, select_malicious_clients
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import time
from unlearning import retrain_attack, server_operations
import syft as sy


args = Arguments.Arg()
hook = sy.TorchHook(torch)
client_number = args.client_number
model = m_LeNet.Net()

# model = CNNCifar.Net()  # svhn
retrain_model = copy.deepcopy(model)
# model = m_MLP.Net4()  # fmnist


epoch = args.epochs
poi_epoch = args.poi_epochs
rec_epochs = args.rec_epochs
target_number = 0
ratio = args.target_client_ratio
print(f"Number of clients：{client_number}")
loss_values = []
clean_accuracy_values = []
poison_accuracy_values = []
rec_poison_accuracy_values = []
retrain_clean_accuracy_values = []
retrain_poison_accuracy_values = []

time_values = []
if __name__ == '__main__':
    print("----------1.Data Processing Stage----------")
    train_dataset, test_dataset = datapro.downData("mnist")
    print("----------2.Initialization phase----------")

    client_sample = int(len(train_dataset[0]) / client_number)
    client_list, client_data, client_neighbors_list, cfl_client_neighbors_list, hfl_client_neighbors_list\
        = client_create(client_number, train_dataset)

    retrain_client_neighbors_list = copy.deepcopy(client_neighbors_list)
    test_copy = copy.deepcopy(test_dataset)
    poisoned_test = backdoor_attack_test_copy(test_copy, 9)

    '''--------------------------Normal training -----------------------------------'''
    print("----------3.CH-FL training----------")
    history_con = []
    aggregated_con = [""] * client_number
    rows, cols = client_number, client_number
    similar = np.zeros((rows, cols))  # Cosine distance
    trust = np.zeros(client_number)  # Trust Value
    predict_mal_clients = []  # The client to be removed
    deleted_clients = []  # Clients that have been deleted
    re_deleted_clients = []
    punish = [0] * client_number  # Penalty value
    # start_time = time.time()
    for i in range(epoch):
        print("---------------------------epoch={}---------------------------".format(i + 1))
        # non_malicious_clients = exclude_malicious_clients(client_number, predict_mal_clients)

        (model, aggregated_con, client_model, contribution, test_loss, clean_accuracy,  local_loss,
         similar, trust, predict_mal_clients, punish) = (
            collaborate_hier_federated_learning(client_number, client_list, client_data, model, test_dataset, poisoned_test,
                                                client_neighbors_list, aggregated_con, similar, predict_mal_clients, trust, punish))
        # end_time = time.time()
        # times = end_time - start_time
        history_con = contribution  # Retain the latest round of client updates.

        clean_accuracy_values.append(clean_accuracy)

    print(f'clean_accuracy_values: {clean_accuracy_values}')


    # Copy models are used for MIA
    # shadow_ch_model = copy.deepcopy(model)


    '''--------------------------Introduce malicious attacks-----------------------------------'''
    print("----------4.Introduce malicious attacks and unlearning actions----------")
    # print(similar)
    for i in range(args.malicious_round):
        print(f'The round {i+1} of malicious attacks')
        selected_malicious_clients, malicious_client_data = select_malicious_clients(client_number, client_data, ratio)
        # print(selected_malicious_clients)
        retrain_client_neighbors_list, re_deleted_clients = retrain_attack(selected_malicious_clients, similar, retrain_client_neighbors_list,re_deleted_clients)
        print(re_deleted_clients)
        # print("----------unlearning----------")
        # start_time = time.time()
        (model, aggregated_con, client_model, contribution, test_loss, clean_accuracy,  local_loss,
         similar, trust, predict_mal_clients, punish) = (
                mi_ch_federated_learning(client_number, client_list, malicious_client_data, model, test_dataset,
                                                  poisoned_test, client_neighbors_list, aggregated_con, similar, trust,
                                                  predict_mal_clients, punish))
        # end_time = time.time()
        # print("training time：", end_time - start_time)
        print(selected_malicious_clients)

        print(trust, predict_mal_clients)
        print(f"Accuracy after unlearning:{clean_accuracy}")
        # stop_acc = clean_accuracy
        clean_accuracy_values.append(clean_accuracy)


    # Adjust the network architecture after removing the malicious client
    client_neighbors_list, deleted_clients = server_operations(predict_mal_clients, similar, client_neighbors_list,deleted_clients)

    # MIA
    # for s in selected_malicious_clients:
    #     shadow_loaders, test_loader = pre_attack(client_data, client_number, test_dataset)
    #     target_model = copy.deepcopy(model)
    #     attack_model = train_attack_model(shadow_ch_model, shadow_loaders, test_loader)
    #     (PRE_unlearn, REC_unlearn, F1_unlearn) = attack(target_model, attack_model, malicious_client_data, test_loader, s)



