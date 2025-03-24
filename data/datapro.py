import torch
import torchvision

# 下载数据集
def downData(name):
    if name == "mnist":
        print("download mnist")
        train_dataset = torchvision.datasets.MNIST(root='\\', train=True,
                                                   transform=torchvision.transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root='\\', train=False,
                                                  transform=torchvision.transforms.ToTensor(), download=True)
        # print(train_dataset.data.type)
        # print(train_dataset.targets.type)
        train_dataset.data = train_dataset.data.type(torch.float32).view(len(train_dataset.data),1,28, 28)
        train_dataset.targets = train_dataset.targets.type(torch.long).view(len(train_dataset.targets))
        test_dataset.data = test_dataset.data.type(torch.float32).view(len(test_dataset.data), 1, 28, 28)
        test_dataset.targets = test_dataset.targets.type(torch.long).view(len(test_dataset.targets))
        train_dataset = (train_dataset.data, train_dataset.targets)
        test_dataset = (test_dataset.data, test_dataset.targets)
        # print(train_dataset[0][0].size(), train_dataset[1][0])
        return train_dataset, test_dataset

    elif name == "FMNIST":
        print("download FMNIST")
        train_dataset = torchvision.datasets.FashionMNIST(root='\\', train=True,
                                                          transform=torchvision.transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root='\\', train=False,
                                                         transform=torchvision.transforms.ToTensor(), download=True)

        train_dataset.data = train_dataset.data.type(torch.float32).view(len(train_dataset.data), 1, 28, 28)
        train_dataset.targets = train_dataset.targets.type(torch.long).view(len(train_dataset.targets))
        test_dataset.data = test_dataset.data.type(torch.float32).view(len(test_dataset.data), 1, 28, 28)
        test_dataset.targets = test_dataset.targets.type(torch.long).view(len(test_dataset.targets))
        train_dataset = (train_dataset.data, train_dataset.targets)
        test_dataset = (test_dataset.data, test_dataset.targets)
        return train_dataset, test_dataset


    elif name == "SVHN":
        print("download svhn")
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train',
                                                    transform=torchvision.transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test',
                                                   transform=torchvision.transforms.ToTensor(), download=True)
        train_dataset.data = torch.from_numpy(train_dataset.data)  # numpy转化为torch
        test_dataset.data = torch.from_numpy(test_dataset.data)
        train_dataset.labels = torch.tensor(train_dataset.labels)
        test_dataset.labels = torch.tensor(test_dataset.labels)
        train_dataset.data = train_dataset.data.type(torch.float32).view(len(train_dataset.data), 3, 32, 32)
        train_dataset.labels = train_dataset.labels.type(torch.long).view(len(train_dataset.labels))
        test_dataset.data = test_dataset.data.type(torch.float32).view(len(test_dataset.data), 3, 32, 32)
        test_dataset.labels = test_dataset.labels.type(torch.long).view(len(test_dataset.labels))
        train_dataset = (train_dataset.data, train_dataset.labels)
        test_dataset = (test_dataset.data, test_dataset.labels)
        return train_dataset, test_dataset








