from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder


def build_network(net_name):
    if net_name == "cifar10_LeNet":
        return CIFAR10_LeNet()
    elif net_name == "cifar10_LeNet_ELU":
        return CIFAR10_LeNet_ELU()
    else:
        raise ValueError("Unsupported network name.")


def build_autoencoder(net_name):
    if net_name == "cifar10_LeNet":
        return CIFAR10_LeNet_Autoencoder()
    elif net_name == "cifar10_LeNet_ELU":
        return CIFAR10_LeNet_ELU_Autoencoder()
    else:
        raise ValueError("Unsupported network name.")
