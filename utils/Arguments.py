
class Arguments:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 64
        self.unl_batch_size = 1024
        self.epochs = 6
        self.local_epochs = 1
        self.unl_epochs = 6
        self.rec_epochs = 4
        self.kd_epochs = 3
        self.retrain_epochs = 10
        self.upga_unlearn = 5
        self.poi_epochs = 1
        self.lr = 0.01
        self.rec_lr = 0.01
        self.unl_lr = 0.06
        self.momentum = 0.9
        # self.no_cuda = False
        self.seed = 1
        self.log_interval = 1
        self.save_model = True
        self.client_number = 10
        # self.shadow_client_number = 2
        self.target_client_ratio = 0.3  # Percentage of malicious clients
        self.w = 6  # Weight adjustments
        self.t = 10
        self.a = 0.5
        self.e = 5
        self.noise_rate = 0.05
        self.percent_poison = 0.8
        self.mu = 2
        self.omega = 0.3
        self.gamma = 0.5
        self.malicious_round = 1
        self.de_radio = 0.3


def Arg ():
    return Arguments()

if __name__ == '__main__':
    print(Arg())
