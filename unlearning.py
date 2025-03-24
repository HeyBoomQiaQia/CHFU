import copy
from utils import Arguments
from Model import m_LeNet
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist
args = Arguments.Arg()
Net = m_LeNet.Net()
r = 1

def cd(model1, model2):

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    net1 = np.concatenate([value.numpy().astype(np.float32).ravel() for value in state_dict1.values()])
    net2 = np.concatenate([value.numpy().astype(np.float32).ravel() for value in state_dict2.values()])

    if net1.ndim != 1 or net2.ndim != 1:
        raise ValueError("The parameters of both models must be one-dimensional arrays.")

    cosine_dist = pdist([net1, net2], 'cosine')

    # print("cosine_dist：", cosine_dist)
    return cosine_dist[0]


def noise_model(model, sigma: float):
    """Aggregates the clients models to create the new model"""
    if sigma > 0:
        for param in model.parameters():
            noise = torch.normal(0.0, sigma, size=param.shape)
            param.data.add_(noise)
    return model


def degradation_model(model1, model2, de_radio):
    de_model = copy.deepcopy(model1)
    de_model_params = de_model.state_dict()
    model1_params = model1.state_dict()
    model2_params = model2.state_dict()

    for param in model1_params:
        de_model_params[param] = model1_params[param] * (1-de_radio) + model2_params[param] * de_radio
    de_model.load_state_dict(de_model_params)
    return de_model

def zero_model(model):
    state_dict = model.state_dict()
    for param_name in state_dict:
        state_dict[param_name] = 0
    model.load_state_dict(state_dict)
    return model


def euclidean_distance(model1, model2, mu, gamma, omega):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    net1 = np.concatenate([value.numpy().ravel() for value in state_dict1.values()])
    net2 = np.concatenate([value.numpy().ravel() for value in state_dict2.values()])
    euclidean_mu = np.linalg.norm(net1 - net2) ** mu
    floor_euclidean_mu = np.floor(np.linalg.norm(net1 - net2)) ** mu
    similarity = 1 / (1 + ((euclidean_mu - omega * floor_euclidean_mu) / euclidean_mu) ** gamma)
    return similarity

def test(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='mean').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(loader.dataset)
        print('Test set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.2f}%)'.format(
            test_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))
        acc = correct / len(loader.dataset)
        return acc, test_loss


# Remove the malicious client and update the network
def server_operations(predict_mal_clients, similar, client_neighbors_list, deleted_clients):

    if predict_mal_clients:  # The list of clients to be deleted is not empty
        # List of pioneer clients
        print(f'predict_mal_clients: {predict_mal_clients}')
        for p in predict_mal_clients:
            if p not in deleted_clients:
                similar_list = []
                precursor_clients = []
                for q in client_neighbors_list.keys():
                    if p in client_neighbors_list[q]:
                        client_neighbors_list[q].remove(p)
                        precursor_clients.append(q)
                        print(f"The pioneer client of {p}: {q}")
                if precursor_clients:
                    for pr in precursor_clients:
                        similar_list.append((pr, similar[pr][p]))

                    agent_client = min(similar_list, key=lambda x: x[1])[0]
                    print(f'For predict_mal_client {p}, the most similar precursor client is {agent_client}')
                    #
                    client_neighbors_list[agent_client] = list(
                        set(client_neighbors_list[agent_client]).union(client_neighbors_list[p]))

                    # client_neighbors_list[agent_client].extend(client_neighbors_list[p])

                    del client_neighbors_list[p]
                    deleted_clients.append(p)
    print(f'client_neighbors_list: {client_neighbors_list}')
    return client_neighbors_list, deleted_clients


def retrain_attack(predict_mal_clients, similar, client_neighbors_list, deleted_clients):
    if predict_mal_clients:

        print(f'predict_mal_clients: {predict_mal_clients}')
        for p in predict_mal_clients:
            if p not in deleted_clients:
                similar_list = []
                precursor_clients = []
                for q in client_neighbors_list.keys():
                    if p in client_neighbors_list[q]:
                        client_neighbors_list[q].remove(p)
                        precursor_clients.append(q)
                        print(f"The pioneer client of {p}: {q}")
                if precursor_clients:
                    for pr in precursor_clients:
                        similar_list.append((pr, similar[pr][p]))

                    agent_client = min(similar_list, key=lambda x: x[1])[0]
                    # print(f'For predict_mal_client {p}, the most similar precursor client is {agent_client}')

                    client_neighbors_list[agent_client] = list(
                        set(client_neighbors_list[agent_client]).union(client_neighbors_list[p]))

                    # client_neighbors_list[agent_client].extend(client_neighbors_list[p])

                    del client_neighbors_list[p]
                    deleted_clients.append(p)
    print(f'client_neighbors_list: {client_neighbors_list}')
    return client_neighbors_list, deleted_clients




