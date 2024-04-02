import numpy as np
import faiss
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import torch as torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from models import Encoder
from models import Projector
from augmentations import train_transform
from augmentations import test_transform
from torch.utils.data import Dataset

ENCODER_DIM = 128
GAMMA = 1.
EPSILON = 1e-4
LAMBDA = 25.
MU = 25.
NU = 1.
BATCH_SIZE = 256
LR = 3e-4
EPOCHS = 10 # change to 30



class CfirDataset(Dataset):
    def __init__(self, imgs, labels, transform, for_training):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.training = for_training

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        if self.training:
            return self.transform(self.imgs[idx]), self.transform(self.imgs[idx]),\
                   self.labels[idx]
        return self.transform(self.imgs[idx]), self.labels[idx]


class MnistAndCfirDataset(Dataset):
    def __init__(self, cfir_imgs, cfir_labels, cfir_transform, mnist_imgs, mnist_labels, mnist_transform):
        self.cfir_imgs = cfir_imgs
        self.cfir_labels = cfir_labels
        self.cfir_transform = cfir_transform
        self.mnist_imgs = mnist_imgs
        self.mnist_labels = mnist_labels
        self.mnist_transform = mnist_transform
        self.cfir_len = self.cfir_imgs.shape[0]

    def __len__(self):
        return self.cfir_imgs.shape[0] + self.mnist_imgs.shape[0]

    def __getitem__(self, idx): #anomalies = 1 , nomral = 0
        if idx < self.cfir_imgs.shape[0]:
            return self.cfir_transform(self.cfir_imgs[idx]), torch.tensor(self.cfir_labels[idx]), torch.zeros(1)
        else:
            idx -= self.cfir_len
            return self.mnist_transform(self.mnist_imgs[idx]), torch.tensor(self.mnist_labels[idx]), torch.ones(1)


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def get_numpy_representations(model, dataset, mixed_data=False):
    if mixed_data:
        for idx, (batch, labels, anomaly_classification) in tqdm(enumerate(dataset)):
            rep = model(batch.to(device))
            if (idx == 0):
                train_anomaly_classification = anomaly_classification
                train_labels = labels
                train_representation = rep
                train_imgs = batch
                continue
            train_anomaly_classification = torch.cat([train_anomaly_classification, anomaly_classification], dim=0)
            train_labels = torch.cat([train_labels,labels],dim=0)
            train_representation = torch.cat([train_representation,rep],dim=0)
            train_imgs = torch.cat([train_imgs, batch],dim=0)
        train_representation = train_representation.detach().cpu().numpy()
        train_labels = train_labels.detach().cpu().numpy()
        return train_representation, train_labels, train_imgs, train_anomaly_classification

    else:
        for idx, (batch, labels) in tqdm(enumerate(dataset)):
            rep = model(batch.to(device))
            if (idx == 0):
                train_labels = labels
                train_representation = rep
                train_imgs = batch
                continue
            train_labels = torch.cat([train_labels,labels],dim=0)
            train_representation = torch.cat([train_representation,rep],dim=0)
            train_imgs = torch.cat([train_imgs, batch],dim=0)
        train_representation = train_representation.detach().cpu().numpy()
        train_labels = train_labels.detach().cpu().numpy()
        return train_representation, train_labels, train_imgs


def plot_roc_auc(org_classification, org_inverse_density, no_nb_classification=None, no_nb_inverse_density=None):
    org_fpr ,org_tpr, org_thresholds = roc_curve(org_classification, org_inverse_density)
    org_roc_auc = auc(org_fpr, org_tpr)
    plt.plot(org_fpr, org_tpr, label=f'ROC original model (AUC={org_roc_auc})')
    if no_nb_classification is not None:
        no_nb_fpr ,no_nb_tpr, no_nb_thresholds = roc_curve(no_nb_classification, no_nb_inverse_density)
        no_nb_roc_auc = auc(no_nb_fpr, no_nb_tpr)
        plt.plot(no_nb_fpr, no_nb_tpr, label=f'ROC no generated neighors model (AUC={no_nb_roc_auc})')
    plt.plot([0, 1], [0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title("Models ROC Curves")
    plt.show()


def calc_knn_density(index, test_reps, k):
    D, I = index.search(test_reps, k)
    return np.mean(D,axis=1)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workers = 0 if device.type == 'cpu' else 2

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)
    encoder = Encoder(ENCODER_DIM, device).to(device)

    if os.path.exists("encoder_model_weights.pth"):
        encoder.load_state_dict(torch.load("encoder_model_weights.pth", map_location=device))

    freeze_model(encoder)

    train_down_dataset = CfirDataset(trainset.data, trainset.targets,
                                     test_transform, False)  # for downstream
    train_down_loader = torch.utils.data.DataLoader(train_down_dataset,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=workers)


    numpy_train_representation, numpy_train_labels, \
    numpy_train_imgs = get_numpy_representations(encoder,
                                                 train_down_loader)

    index = faiss.IndexFlatL2(ENCODER_DIM)
    index.add(numpy_train_representation)


    mnist_transform = transforms.Compose([
        transforms.Lambda(lambda x: transforms.functional.pad(x, (2, 2, 2, 2),
                                                              fill=0)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        test_transform
    ]) # partly used chatgpt

    mnistset = torchvision.datasets.MNIST(root='./data', train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
    mnist_cfir_test_dataset = MnistAndCfirDataset(testset.data, testset.targets,
                                    test_transform, mnistset.data, mnistset.targets, mnist_transform) # for downstream
    mnist_cfir_test_loader = torch.utils.data.DataLoader(mnist_cfir_test_dataset,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=workers)

    mnist_cfir_org_numpy_test_representation, mnist_cfir_org_numpy_test_labels, \
                    mnist_cfir_org_numpy_test_imgs, mnist_cfir_org_numpy_test_classification\
                     = get_numpy_representations(encoder, mnist_cfir_test_loader, True)
    # Q1
    inverse_density = calc_knn_density(index,
                                       mnist_cfir_org_numpy_test_representation, 2)
    # Q2
    plot_roc_auc(mnist_cfir_org_numpy_test_classification,
                 inverse_density)