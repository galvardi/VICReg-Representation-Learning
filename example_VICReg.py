import numpy as np
import faiss
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tqdm import trange
import torch as torch
import torch.nn as nn
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


def train_linear_prob(model,trainset, model_name):
    fc_prob = nn.Linear(ENCODER_DIM, 10, device=device)
    fc_prob.to(device)
    # if os.path.exists(f"weights/{model_name}linear_prob_weights.pth"):
    #     fc_prob.load_state_dict(torch.load(f"weights/{model_name}linear_prob_weights.pth",
    #                                     map_location=device))
    # else:
    fc_prob.train()
    optimizer = torch.optim.Adam(fc_prob.parameters(), lr=LR,
                                 betas=(0.9, 0.999), weight_decay=1e-6)
    cross_loss = nn.CrossEntropyLoss()
    for epoch in trange(10):
        for batch_idx, (imgs, labels) in enumerate(trainset):
            imgs = imgs.to(device)
            labels = labels.to(device)
            y = model(imgs)
            y = fc_prob(y)
            loss = cross_loss(y, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # torch.save(fc_prob.state_dict(), f"{model_name}linear_prob_weights.pth")
    freeze_model(fc_prob)
    return fc_prob

def test_linear_prob(model, fc_prob, testset, model_name):
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(testset):
            imgs = imgs.to(device)
            labels = labels.to(device)
            y = model(imgs)
            y = fc_prob(y)
            pred_labels = torch.argmax(y, dim=1)
            if i == 0:
                test_labels = labels
                test_pred = pred_labels
                continue
            test_pred = torch.cat([test_pred, pred_labels])
            test_labels = torch.cat([test_labels, labels])
        print(f"{model_name} linear probing accuracy score - ", accuracy_score(
            test_labels.cpu().numpy(),
                             test_pred.cpu().numpy()))


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def imshow(img):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261) #used chatgpt
    unnormalize = transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                                        (1.0 / std[0], 1.0 / std[1], 1.0 / std[2]))
    img = unnormalize(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def get_numpy_representations(model, dataset):
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


def batch_retrieve(imgs, I, i=None):
    img_neighbors = imgs[I[0]].unsqueeze(dim=0)
    for idx in I[1:]:
        img_neighbors = torch.cat([img_neighbors, imgs[idx].unsqueeze(dim=0)])
    return img_neighbors

def retrieve(train_labels, train_reps, index, imgs, k, label):
        indcies = np.arange(train_labels.shape[0])
        first = True
        while first or train_labels[choice] != label:
            choice = np.random.choice(indcies)
            first = False
        _, I = index.search(np.expand_dims(train_reps[choice], axis=0), k+1)
        img_neighbors = batch_retrieve(imgs, I[0], choice)
        imshow(torchvision.utils.make_grid(img_neighbors))



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workers = 0 if device.type == 'cpu' else 2

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)
    org_enc = Encoder(ENCODER_DIM, device).to(device)

    # Q1
    org_enc.load_state_dict(torch.load("encoder_model_weights.pth", map_location=device))

    freeze_model(org_enc)

    train_down_dataset = CfirDataset(trainset.data, trainset.targets,
                                     test_transform, False)  # for downstream
    train_down_loader = torch.utils.data.DataLoader(train_down_dataset,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=workers)

    test_down_dataset = CfirDataset(testset.data, testset.targets,
                                    test_transform, False)  # for downstream

    test_down_loader = torch.utils.data.DataLoader(test_down_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=workers)
    fc_prob = train_linear_prob(org_enc, train_down_loader, "org_")
    test_linear_prob(org_enc, fc_prob, test_down_loader, "original")

    numpy_train_representation, numpy_train_labels, \
    numpy_train_imgs = get_numpy_representations(org_enc,
                                                     train_down_loader)

    index = faiss.IndexFlatL2(ENCODER_DIM)
    index.add(numpy_train_representation)
    print("retrival with original VICReg model")
    for i in range(10):
        retrieve(numpy_train_labels, numpy_train_representation,
                 index, numpy_train_imgs, 5, i)