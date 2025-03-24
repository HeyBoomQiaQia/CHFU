import torch as pt

class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.fc1 = pt.nn.Linear(784, 512)
        # self.fc2 = pt.nn.Linear(512, 128)
        # self.fc3 = pt.nn.Linear(128, 10)
        self.fc1 = pt.nn.Linear(28*28*1, 512)
        self.fc2 = pt.nn.Linear(512, 10)

    def forward(self, x):
        # din = din.view(-1, 28 * 28)
        # print('x_shape:', x.shape)

        x = x.view(-1,28*28*1)
        x = pt.nn.functional.relu(self.fc1(x))
        x = pt.nn.functional.relu(self.fc2(x))
        return pt.nn.functional.log_softmax(x, dim=1)

def Net4():
    return MLP()

# model = MLP().cuda()
# print(model)