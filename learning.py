import copy
import math
import time

import syft as sy
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim.lr_scheduler import StepLR

from utils import Arguments
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from unlearning import  euclidean_distance,  zero_model, degradation_model, cd

hook = sy.TorchHook(torch)
args = Arguments.Arg()

criteria = nn.CrossEntropyLoss()


def test_f1(model, test_data, num_classes=10):
    model.eval()
    test_loss = 0
    correct = 0
    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)

    with torch.no_grad():
        test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            # print(pred, target)
            correct += pred.eq(target.view_as(pred)).sum().item()
            for true, pred in zip(target, pred):
                if pred == true:
                    true_positives[pred] += 1
                else:
                    false_positives[pred] += 1
                    false_negatives[true] += 1

    recalls = true_positives / (true_positives + false_negatives)
    precisions = true_positives / (true_positives + false_positives)
    f1_scores = 2 * (recalls * precisions) / (recalls + precisions)
    average_recall = torch.mean(recalls)
    average_f1_score = torch.mean(f1_scores)

    test_loss /= len(test_data[0])
    accuracy = 100. * correct / len(test_data[0])

    print(
        'Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%), Average Recall: {:.6f}, Average F1 Score: {:.6f}'.format(
            test_loss, correct, len(test_data[0]), accuracy, average_recall.item(), average_f1_score.item()))

    return test_loss, accuracy, average_recall, average_f1_score


def test(model, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            # print(data.shape)
            output = model(data)
            # print(output.shape,target.shape)
            # test_loss += F.cross_entropy(output, target, reduction='sum').item()
            test_loss += criteria(output, target).item()

            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # pred = torch.argmax(output, axis=1)
            # correct += accuracy_score(pred, target)

        test_loss /= len(test_data[0])
        accuracy = 100. * correct / len(test_data[0])
        # correct = correct / np.ceil(len(test_loader.dataset)/test_loader.batch_size)
        print('Test set : Average loss : {:.6f}, Accuracy: {}/{} ( {:.2f}%)'.format(
            test_loss, correct, len(test_data[0]),
            accuracy))
        # print('Test set: Average loss: {:.8f}'.format(test_loss))
        # print('Test set: Average acc:  {:.4f}'.format(correct))
        return test_loss, accuracy


def distribute_model(client_number, model):
    client_model = []
    client_optim = []
    client_scheduler = []
    for i in range(client_number):
        client_model.append(model.copy())
        client_optim.append(torch.optim.SGD(client_model[i].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4))
        scheduler = StepLR(client_optim[i], step_size=10, gamma=0.001)

        client_scheduler.append(scheduler)
    return client_model, client_optim, client_scheduler

def fedavg_update_weight(model, client_models, client_sample_counts):
    if len(client_models) != len(client_sample_counts):
        raise ValueError("Number of client models must be equal to the number of sample counts.")

    global_params = model.state_dict()

    for client_model, sample_count in zip(client_models, client_sample_counts):
        client_params = client_model.state_dict()
        for param_name in global_params:
            global_params[param_name] += sample_count * client_params[param_name]
    print(f'sum(client_sample_counts): {sum(client_sample_counts)}')
    for param_name in global_params:
        global_params[param_name] /= sum(client_sample_counts)
    model.load_state_dict(global_params)
    return model



def col_fedavg_update_weight(model, client_models, client_sample_counts, client_sample):
    # Initialize the global model parameters
    global_params = model.state_dict()
    for param in global_params:
        global_params[param] *= client_sample

    for client_model, sample_count in zip(client_models, client_sample_counts):
        client_params = client_model.state_dict()
        for param_name in global_params:
            global_params[param_name] += sample_count * client_params[param_name]
    # print(f'sum(client_sample_counts): {sum(client_sample_counts)}')
    for param_name in global_params:
        global_params[param_name] /= (sum(client_sample_counts) + client_sample)
    model.load_state_dict(global_params)
    return model


# loss_values = []
# accuracy_values = []


def backdoor_attack_train(backdoor_data, percent_poison, backdoor_label: int):
    all_indices = np.arange(len(backdoor_data[0]))
    # print(f"all_indices: {all_indices}")
    remove_indices = (torch.nonzero(backdoor_data[1] == backdoor_label).squeeze()).numpy()
    # print(f'remove_indices:{remove_indices},{len(remove_indices)}')
    target_indices = list(set(all_indices) - set(remove_indices))
    print(f'The length of the target index: {len(target_indices)}')
    num_poison = int(percent_poison * len(target_indices))
    print(f'num poison: {num_poison}')
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)
    print(selected_indices, len(selected_indices))
    backdoor_data[1][selected_indices] = backdoor_label
    return backdoor_data

def backdoor_attack_train_copy(test_data, percent_poison, backdoor_label: int):
    for i in range(len(test_data[0])*percent_poison):
        test_data[0][i][0][26][26] = 255
        test_data[0][i][0][25][25] = 255
        test_data[0][i][0][24][26] = 255
        test_data[0][i][0][26][24] = 255
        test_data[1][i] = backdoor_label
    return test_data

def backdoor_attack_test(test_data, backdoor_label: int):
    all_indices = np.arange(len(test_data[0]))
    print(f"all_indices: {all_indices}")
    remove_indices = (torch.nonzero(test_data[1] == backdoor_label).squeeze()).numpy()
    print(f'remove_indices:{remove_indices},{len(remove_indices)}')
    target_indices = list(set(all_indices) - set(remove_indices))
    print(f'The length of the target index: {len(target_indices)}')
    test_data[1][target_indices] = backdoor_label
    return test_data

def backdoor_attack_test_copy(test_data, backdoor_label: int):

    for i in range(len(test_data[0])):
        test_data[0][i][0][26][26] = 255
        test_data[0][i][0][25][25] = 255
        test_data[0][i][0][24][26] = 255
        test_data[0][i][0][26][24] = 255
        test_data[1][i] = backdoor_label
    return test_data
def federated_learning(client_number, client_list, client_data, model, test_data, backdoor_test_data, client_sample):
    # 分配模型
    client_model, client_optim, client_scheduler = distribute_model(client_number, model)
    contribution = []

    print("----------start training----------")
    for i in range(client_number):
        client_model[i].train()
        # client_model[i].send(client_list[i])
    for i in range(client_number):
        train_dataset = torch.utils.data.TensorDataset(client_data[i][0], client_data[i][1].long())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        for local_epoch in range(args.local_epochs):
            for epoch_ind, (data, target) in enumerate(train_loader):
                # data = data.send(client_list[i])
                # target = target.send(client_list[i])

                client_optim[i].zero_grad()
                pred = client_model[i](data)
                # loss = F.nll_loss(pred, target)
                loss = criteria(pred, target)
                loss.backward()
                client_optim[i].step()
                # if (epoch_ind + 1) % 50 == 0:
                #     print("There is epoch:{}, loss:{:.6f}".format(epoch_ind + 1, loss.get().data.item()))
                # client_scheduler[i].step()
            test(client_model[i], test_data)
    with torch.no_grad():
        cn = []
        n = 0
        local_loss_list = []
        for i in range(client_number):
            cn.append(int(len(client_data[i][0])))
            n += int(len(client_data[i][0]))
            print("client {}: ".format(i + 1))

            # acc_list = []
            # client_model[i].get()
            # add recall、f1_score
            # local_loss, acc, recall, f1_score = test_f1(client_model[i], test_data, 10)
            local_loss, acc = test(client_model[i], test_data)
            contribution.append(client_model[i])
            local_loss_list.append(round(local_loss, 4))

            # acc_list.append(acc)
        print(f"loss:{local_loss_list}")
        print(f'cn:{cn}')
        model = fedavg_update_weight(model, client_model, client_sample_counts=cn)
        # model = avg_update_weight(model, client_model)
        contribution.append(model)
        # print("Global: ")
        # test_loss, clean_accuracy, recall, f1_score = test_f1(model, test_data, 10)
        test_loss, clean_accuracy = test(model, test_data)
        poison_loss, poison_accuracy = test(model, backdoor_test_data)

    return model, client_model, contribution, test_loss, clean_accuracy, poison_accuracy, local_loss_list


# normal CH-FL
def collaborate_hier_federated_learning(client_number, client_list, client_data, model, test_data, backdoor_test_data,
                                   neighbors_list, aggregated_con, similar, predict_mal_clients, trust, punish):
    # 分配模型
    client_model, client_optim, client_scheduler = distribute_model(client_number, model)
    contribution = [None] * len(client_model)
    print("-----Start training---------")
    for i in range(client_number):
        if i not in predict_mal_clients:
            client_model[i].train()
            # client_model[i].send(client_list[i])
    for i in range(client_number):
        if i not in predict_mal_clients:  # Only train benign clients.
            train_dataset = torch.utils.data.TensorDataset(client_data[i][0], client_data[i][1].long())
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            for local_epoch in range(args.local_epochs):
                for epoch_ind, (data, target) in enumerate(train_loader):
                    # data = data.send(client_list[i])
                    # target = target.send(client_list[i])
                    client_optim[i].zero_grad()
                    pred = client_model[i](data)
                    # loss = F.nll_loss(pred, target)
                    loss = criteria(pred, target)
                    loss.backward()
                    client_optim[i].step()
                    # if (epoch_ind + 1) % 50 == 0:
                    #     print("There is epoch:{}, loss:{:.6f}".format(epoch_ind + 1, loss.get().data.item()))
                # client_scheduler[i].step()
                # test(client_model[i], test_data)
    with torch.no_grad():
        cn = [""] * client_number
        n = 0
        local_loss_list = [""] * client_number
        # aggregated_con = []
        for i in range(client_number):
            if i not in predict_mal_clients:
                cn[i] = (int(len(client_data[i][0])))
                n += int(len(client_data[i][0]))
                print("client {}: ".format(i + 1))

                # acc_list = []
                # client_model[i].get()
                local_loss, acc = test(client_model[i], test_data)
                contribution[i] = copy.deepcopy(client_model[i])
                local_loss_list[i] = round(local_loss, 4)

                # acc_list.append(acc)
        print(f"最终loss:{local_loss_list}")
        print(f'cn:{cn}')

        aggregated_con, similar, trust, punish = neighbors_select(client_model, neighbors_list, cn, aggregated_con, similar, trust, punish)

        test_loss, clean_accuracy = test(aggregated_con[0], test_data)
        # poison_loss, poison_accuracy = test(aggregated_con[0], backdoor_test_data)
    print(trust, predict_mal_clients)
    return (aggregated_con[0], aggregated_con, client_model, contribution, test_loss, clean_accuracy,
            local_loss_list, similar, trust, predict_mal_clients, punish)

# CH-FL under malicious attack
def mi_ch_federated_learning(client_number, client_list, client_data, model, test_data, backdoor_test_data,
                                      neighbors_list, his_aggregated_con, similar, trust, predict_mal_clients, punish):
    # Assignment model
    client_model, client_optim, client_scheduler = distribute_model(client_number, model)
    contribution = [None] * len(client_model)  # Record the most recent model for each client
    print("-----Start training---------")
    for i in range(client_number):
        if i not in predict_mal_clients:
            client_model[i].train()
            # client_model[i].send(client_list[i])
    for i in range(client_number):
        if i not in predict_mal_clients:  # Only train benign clients

            train_dataset = torch.utils.data.TensorDataset(client_data[i][0], client_data[i][1].long())
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            for epoch_ind, (data, target) in enumerate(train_loader):
                # data = data.send(client_list[i])
                # target = target.send(client_list[i])
                client_optim[i].zero_grad()
                pred = client_model[i](data)
                # loss = F.nll_loss(pred, target)
                loss = criteria(pred, target)
                loss.backward()
                client_optim[i].step()
                # if (epoch_ind + 1) % 50 == 0:
                #     print("There is epoch:{}, loss:{:.6f}".format(epoch_ind + 1, loss.get().data.item()))
            # client_scheduler[i].step()
    with torch.no_grad():

        cn = [""] * client_number
        n = 0
        local_loss_list = [""] * client_number
        # aggregated_con = []
        for i in range(client_number):
            if i not in predict_mal_clients:
                cn[i] = (int(len(client_data[i][0])))
                n += int(len(client_data[i][0]))
                print("client {}: ".format(i + 1))

                # acc_list = []
                # client_model[i].get()
                local_loss, acc = test(client_model[i], test_data)
                contribution[i] = copy.deepcopy(client_model[i])
                local_loss_list[i] = round(local_loss, 4)

                # acc_list.append(acc)
        # print(f"loss:{local_loss_list}")
        # print(f'cn:{cn}')
        start_time = time.time()
        aggregated_con, similar, trust, predict_mal_clients, punish = mali_neighbors_unlearning(client_model, neighbors_list,
                                                                                        cn, his_aggregated_con, similar,
                                                                                        trust, predict_mal_clients, punish)
        # test_loss, clean_accuracy = test(de_model, test_data)
        test_loss, clean_accuracy = test(aggregated_con[0], test_data)
        # poison_loss, poison_accuracy = test(aggregated_con[0], backdoor_test_data)
    print(trust, predict_mal_clients)
    end_time = time.time()
    print(f"unlearning time：{end_time - start_time}")
    return (aggregated_con[0], aggregated_con, client_model, contribution, test_loss, clean_accuracy,
            local_loss_list, similar, trust, predict_mal_clients, punish)


def neighbors_select(client_model, neighbor_list, cn, his_aggrecon, similar, trust, punish):
    begin_client = []
    aggregated_con = [None] * len(client_model)
    punish_cn = cn  # [0] * len(client_model)
    punish_cn[0] = cn[0]
    for i in neighbor_list.keys():
        if neighbor_list[i] == []:
            begin_client.append(i)
            aggregated_con[i] = client_model[i]
    while (len(begin_client) < len(neighbor_list)):
        for i in begin_client:
            for j in neighbor_list.keys():
                neighbor_model = []
                neighbor_amount = []
                if i in neighbor_list[j] and j not in begin_client:
                    n = 0
                    for k in neighbor_list[j]:
                        if aggregated_con[k]:
                            n += 1
                    if n == len(neighbor_list[j]):
                        print("true")
                        for k in neighbor_list[j]:
                            if his_aggrecon[j] != "":

                                dis = cd(his_aggrecon[j], aggregated_con[k])

                                print(f"The distance between client {j} and his neighbor model {k} is:", dis)
                                similar[j][k] = dis
                                trust[k] = trust[k] - 1 if trust[k] != 0 else trust[k]  # 良性值-1
                            neighbor_model.append(aggregated_con[k])
                            punish_cn[k] = math.floor(cn[k] * 1 / ((punish[k]/args.w) * math.exp(punish[k] / args.w) + 1))
                            neighbor_amount.append(punish_cn[k])
                            # neighbor_amount.append(cn[k])
                    # print("neighbor_model:", neighbor_model)
                    # print("client_model[j]:",client_model[j])
                        print(punish_cn[j])
                        aggregated_con[j] = col_fedavg_update_weight(client_model[j], neighbor_model,
                                                                     neighbor_amount, punish_cn[j])
                        # aggregated_con[j] = avg_update_weight(client_model[j], neighbor_model)
                        begin_client.append(j)
                        print("client: ", begin_client)
            # begin_client.remove(i)
    # print("aggregated_con: ", aggregated_con)
    print(punish_cn)
    return aggregated_con, similar, trust, punish


def mali_neighbors_unlearning(client_model, neighbor_list, cn, his_aggrecon, similar, trust, predict_mal_clients, punish):

    begin_client = []
    change = []  # Prevent repetition in a round
    punish_cn = cn  # The weight value after punishment.
    # punish_cn[0] = cn[0]
    aggregated_con = [None] * len(client_model)  # The aggregated model.

    for i in neighbor_list.keys():
        if not neighbor_list[i]:
            begin_client.append(i)
            aggregated_con[i] = client_model[i]  # Leaf nodes do not aggregate other models
    # print(f"begin_client:{begin_client}")
    while len(begin_client) < len(neighbor_list):
        for i in begin_client:
            for j in neighbor_list.keys():
                neighbor_model = []
                neighbor_amount = []
                if i in neighbor_list[j] and j not in begin_client:
                    n = 0
                    for k in neighbor_list[j]:
                        if aggregated_con[k]:
                            n += 1
                    if n == len(neighbor_list[j]):  # All child nodes are updated before they are aggregated
                        for k in neighbor_list[j]:
                            # Identify malicious attacks
                            if his_aggrecon[j] != "":
                                dis = cd(his_aggrecon[j], aggregated_con[k])
                                # dis = cosine_distance(his_aggrecon[j], aggregated_con[k])
                                print(f"Distance between the {j} and {k} is:", dis)
                                if similar[j][k] != 0:
                                    # print(f"dis:{dis}")
                                    print(f"similar:{similar[j][k]}")
                                    if dis <= similar[j][k]:
                                        similar[j][k] = dis
                                        neighbor_model.append(aggregated_con[k])
                                        # punish_cn[k] = cn[k]
                                        # neighbor_amount.append(punish_cn[k])
                                        trust[k] = trust[k] - 1 if trust[k] != 0 else trust[k]  # Benign value -1
                                    else:
                                        similar[j][k] = (dis + similar[j][k]) / 2
                                        # Go back and unlearning
                                        de_model = degradation_model(his_aggrecon[j], aggregated_con[k], args.de_radio)

                                        neighbor_model.append(de_model)
                                        # neighbor_model.append(his_aggrecon[j])
                                        print(f"The neighbor model {k} may be attacked")
                                        # Prevent repeated additions in a round
                                        if k not in change:
                                            change.append(k)
                                            trust[k] = trust[k] + 1
                                            punish[k] += 1

                                            if trust[k] == 2:
                                                predict_mal_clients.append(k)
                                    # Weight Update
                                    punish_cn[k] = math.floor(cn[k] * 1 / ((punish[k]/args.w) * math.exp(punish[k] / args.w) + 1))
                                    # punish_cn[k] = cn[k]
                                    neighbor_amount.append(punish_cn[k])
                                else:
                                    similar[j][k] = dis
                                    # print(f"dis:{dis}")
                                    # print(f"similar:{similar[j][k]}")
                            else:
                                neighbor_model.append(aggregated_con[k])
                                neighbor_amount.append(cn[k])
                        # print("neighbor_model:", neighbor_model)
                        # print("client_model[j]:",client_model[j])
                        aggregated_con[j] = col_fedavg_update_weight(client_model[j], neighbor_model,
                                                                     neighbor_amount, punish_cn[j])
                        begin_client.append(j)
                        print("client: ", begin_client)
    print(f"punish:{punish}")
    print(f"punish_cn:{punish_cn}")

    # print("aggregated_con: ", aggregated_con)
    # print(f"similar:{similar}")
    return aggregated_con, similar, trust, predict_mal_clients, punish
