import math
import syft as sy
from learning import backdoor_attack_train
from utils import Arguments
import random
import copy
import torch
hook = sy.TorchHook(torch)
args = Arguments.Arg()

def distribute_data(normal_number, train_dataset):
    train_data, train_targets = train_dataset
    normal_len = len(train_data)
    client_data = []
    for i in range(normal_number):
        indx, indy = int(i * normal_len / normal_number), int((i + 1) * normal_len / normal_number)
        client_data.append((train_data[indx:indy], train_targets[indx:indy]))
    print(f"len(client_data):{len(client_data)}")
    return client_data
def our_assign_neighbors(client_number):
    neighbors_list = {}
    list0 = [1, 4, 6]
    list1 = [2]
    list2 = [3]
    list3 = []
    list4 = [5]
    list5 = [7, 9]
    list6 = [5, 7]
    list7 = [8]
    list8 = []
    list9 = []
    list10 = []
    # n=15
    # list0 = [1, 4, 6,12]
    # list1 = [2]
    # list2 = [3,11]
    # list3 = []
    # list4 = [5]
    # list5 = [7, 9]
    # list6 = [5, 7]
    # list7 = [8,10]
    # list8 = []
    # list9 = []
    # list10 = []
    # list11 = []
    # list12 = [13]
    # list13 = [10,14]
    # list14 = []
    for i in range(client_number):

        neighbors_list[i] = locals()[f'list{i}']
    return neighbors_list



def hfl_assign_neighbors(client_number):
    hfl_neighbors_list = {}
    # list0 = [1, 2, 3]
    # list1 = [4, 5]
    # list2 = [6, 7]
    # list3 = [8, 9]
    # list4 = []
    # list5 = []
    # list6 = []
    # list7 = []
    # list8 = []
    # list9 = []
    # list10 = []
    #n=15
    list0 = [1, 2, 3,11,14]
    list1 = [4, 5]
    list2 = [6, 7]
    list3 = [8, 9, 10]
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    list10 = []
    list11 = [12,13]
    list12 = []
    list13 = []
    list14 = []


    for i in range(client_number):
        hfl_neighbors_list[i] = locals()[f'list{i}']
    return hfl_neighbors_list



def cfl_assign_neighbors(client_number):
    cfl_neighbors_list = {}
    # list0 = [1, 2, 3, 4, 5, 6, 7, 8]
    # list1 = []
    # list2 = [9]
    # list3 = []
    # list4 = []
    # list5 = []
    # list6 = []
    # list7 = []
    # list8 = []
    # list9 = []
    # list10 = []
    #n=15
    list0 = [1, 2, 3, 4, 5, 6, 7, 8]
    list1 = [10]
    list2 = [9]
    list3 = [11]
    list4 = [12]
    list5 = [13]
    list6 = [14]
    list7 = []
    list8 = []
    list9 = []
    list10 = []
    list11 = []
    list12 = []
    list13 = []
    list14 = []

    for i in range(client_number):
        cfl_neighbors_list[i] = locals()[f'list{i}']
    return cfl_neighbors_list


def client_create(client_number, train_dataset):
    print("----------① Create a client with a quantity of {}----------".format(client_number))
    client_list = []
    for i in range(client_number):
        # client_list.append(sy.VirtualWorker(hook, id=str(i)))
        client_list.append(i)
    print("----------Create Successful----------")
    torch.manual_seed(args.seed)
    print("----------② Distribution data----------")

    print(len(train_dataset))
    client_data = distribute_data(client_number, train_dataset)
    our_client_neighbors_list = our_assign_neighbors(client_number)
    cfl_client_neighbors_list = cfl_assign_neighbors(client_number)
    hfl_client_neighbors_list = hfl_assign_neighbors(client_number)

    print("----------Distribution successful----------")
    return (client_list, client_data,  our_client_neighbors_list, cfl_client_neighbors_list,
            hfl_client_neighbors_list)


# Percentage of malicious attacks
def select_malicious_clients(client_number, client_data, ratio):
    num_selected_clients = math.ceil((client_number - 1) * ratio)
    all_clients = list(range(1, client_number))
    malicious_client_data = copy.deepcopy(client_data)
    selected_malicious_clients = random.sample(all_clients, num_selected_clients)
    for i in selected_malicious_clients:
        malicious_client_data[i] = backdoor_attack_train(malicious_client_data[i], 0.8, 9)
    return selected_malicious_clients, malicious_client_data
