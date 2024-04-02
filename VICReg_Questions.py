import numpy as np
import faiss
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_curve, auc, silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from tqdm import trange
import torch as torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
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

def cov_obj(z):
    z_mean = torch.mean(z,dim=0,keepdim=True)
    z = (z - z_mean)
    cov_mat = (torch.matmul(z.t(),z) / (z.size(0)-1)).pow(2)
    return (cov_mat.sum() - cov_mat.trace()) / cov_mat.size(0)

def var_obj(z):
    sigmas = ((torch.var(z,dim=0) + EPSILON)).pow(0.5)
    sigmas = (torch.ones(z.size(1)).to(device) * GAMMA) - sigmas
    return torch.where(sigmas < 0, 0, sigmas).mean()

def calculate_loss_terms(z1, z2):

    mse = nn.MSELoss()
    z_inv_loss = mse(z1, z2)
    z1_var_loss = var_obj(z1)
    z2_var_loss = var_obj(z2)
    z1_cov_loss = cov_obj(z1)
    z2_cov_loss = cov_obj(z2)
    return z1_cov_loss, z1_var_loss, z2_cov_loss, z2_var_loss, z_inv_loss

def plot_obj_loss(obj_losses, test_losses, loss_index, test_x_axis):
    loss_titles = {0:'Invariance Objective Loss', 1:'Variance Objective '
                                                    'Loss', 2:'Covariance Objective Loss'}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(obj_losses.shape[1])),y=obj_losses[loss_index][:]))
    fig.add_trace(go.Scatter(x=test_x_axis, y=test_losses[loss_index][:], name='test'))
    fig.update_layout(title=loss_titles[loss_index], xaxis_title='batches', yaxis_title='loss')
    fig.show()



def train_VICReg(enc, proj, train, test, mu=MU, num_epochs=EPOCHS,
                 data_len=None):

    proj.train()
    num_of_batches = int(np.ceil(data_len / BATCH_SIZE))
    test_losses = np.zeros([3,num_epochs])
    obj_losses = np.zeros([3,num_epochs*num_of_batches])
    test_x_axis = []
    optimizer = torch.optim.Adam([*enc.parameters(),*proj.parameters()],
                                 lr=LR,
                                 betas=(0.9, 0.999), weight_decay=1e-6)
    idx = 0
    for epoch in trange(num_epochs):
        enc.train()
        for t1, t2, _ in train:
            optimizer.zero_grad()
            z1 = proj(enc(t1.to(device)))
            z2 = proj(enc(t2.to(device)))
            z1_cov_loss, z1_var_loss, z2_cov_loss, z2_var_loss, z_inv_loss = \
                                                     calculate_loss_terms(z1,z2)
            obj_losses[0][idx] = z_inv_loss.item()
            obj_losses[1][idx] = (z1_var_loss + z2_var_loss).item()
            obj_losses[2][idx] = (z1_cov_loss + z2_cov_loss).item()
            vic_loss = LAMBDA * z_inv_loss + mu * (z1_var_loss + z2_var_loss) + NU * (z1_cov_loss + z2_cov_loss)
            vic_loss.backward()
            optimizer.step()
            idx += 1
        test_x_axis.append(idx)
        enc.eval()
        with torch.no_grad():
            test_data1, test_data2, _ = next(iter(test))
            z1 = enc(test_data1.to(device))
            z2 = enc(test_data2.to(device))
            z1_cov_loss, z1_var_loss, z2_cov_loss, z2_var_loss, z_inv_loss = calculate_loss_terms(z1, z2)
            test_losses[0][epoch] = z_inv_loss.item()
            test_losses[1][epoch] = (z1_var_loss + z2_var_loss).item()
            test_losses[2][epoch] = (z1_cov_loss + z2_cov_loss).item()
    return obj_losses, test_losses, test_x_axis

def train_linear_prob(model,trainset, model_name):
    fc_prob = nn.Linear(ENCODER_DIM, 10, device=device)
    fc_prob.to(device)
    if os.path.exists(f"{model_name}linear_prob_weights.pth"):
        fc_prob.load_state_dict(torch.load(f"{model_name}linear_prob_weights.pth",
                                        map_location=device))
    else:
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


def plot_representation(method,test_rep,test_labels, title):
    reduced = method.fit_transform(test_rep)
    labels = test_labels.astype('str')
    fig = px.scatter(x=reduced[:, 0], y=reduced[:, 1],
                            color=labels, title=title)
    fig.show()

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

class KNNDataset(Dataset):
    def __init__(self, k, index, training_set, train_imgs, train_labels):
        self.imgs = train_imgs
        self.index = index
        self.k = k
        self.train_reps = training_set
        self.train_labels = train_labels
        self.close_dict = {}
        self.calc_closest()

    def calc_closest(self):
        for i in range(int(self.train_reps.shape[0]/BATCH_SIZE)-1):
            reps = self.train_reps[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
            _, I = self.index.search(reps, self.k)
            idx = i*BATCH_SIZE
            for j in range(BATCH_SIZE):
                n = np.random.choice(I[j], size=2, replace=False)
                self.close_dict[idx + j] = n[0] if idx+j != n[0] else n[1]
        _, I = self.index.search(self.train_reps[idx+j+1:,:], self.k)
        for k in range(idx+j+1,self.train_reps.shape[0]):
            n = np.random.choice(I[k - (idx+j+1)], size=2, replace=False)
            self.close_dict[k] = n[0] if k != n[0] else n[1]

    def get_nearest_neighbors(self, k, label=None, idx=None):
        if idx is None:
            indcies = np.arange(self.train_labels.shape[0])
            first = True
            while first or self.train_labels[choice] != label:
                choice = np.random.choice(indcies)
                first = False
        else:
            choice = idx
        _, I = self.index.search(np.expand_dims(self.train_reps[choice], axis=0), k+1)
        return choice, I

    def get_furthest_neighbors(self, idx, k):
        _, I = self.index.search(np.expand_dims(self.train_reps[idx],
                                                axis=0), self.index.ntotal)
        I = I[0][-(k):]
        return idx, I


    def __len__(self):
        return self.train_reps.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx],  self.imgs[self.close_dict[idx]]



def train_no_neighbors_vic_reg(enc, proj, neighbors_dataset,
                               num_epochs=EPOCHS):
    proj.train()
    enc.train()
    optimizer = torch.optim.Adam([*enc.parameters(),*proj.parameters()],
                                 lr=LR,
                                 betas=(0.9, 0.999), weight_decay=1e-6)
    for epoch in trange(num_epochs):
      for z1, z2 in neighbors_dataset:
        z1 = proj(enc(z1.to(device)))
        z2 = proj(enc(z2.to(device)))
        z1_cov_loss, z1_var_loss, z2_cov_loss, z2_var_loss, z_inv_loss = calculate_loss_terms(
            z1, z2)
        vic_loss = LAMBDA * z_inv_loss + MU * (z1_var_loss + z2_var_loss) \
                   + NU * (z1_cov_loss + z2_cov_loss)
        vic_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def batch_neighbors(neighbors_data_set, I, i=None, append=False):
    if append:
        img_neighbors_labels = [neighbors_data_set.train_labels[i]]
        img_neighbors = neighbors_data_set.imgs[i].unsqueeze(dim=0)
        for idx in I:
            img_neighbors = torch.cat([img_neighbors,neighbors_data_set.imgs[idx].unsqueeze(dim=0)])
            img_neighbors_labels.append(neighbors_data_set.train_labels[idx])
    else:
        img_neighbors_labels = [neighbors_data_set.train_labels[I[0]]]
        img_neighbors = neighbors_data_set.imgs[I[0]].unsqueeze(dim=0)
        for idx in I[1:]:
            img_neighbors = torch.cat([img_neighbors,neighbors_data_set.imgs[idx].unsqueeze(dim=0)])
            img_neighbors_labels.append(neighbors_data_set.train_labels[idx])
    return img_neighbors, img_neighbors_labels

def imshow(img):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261) #used chatgpt
    unnormalize = transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                                        (1.0 / std[0], 1.0 / std[1], 1.0 / std[2]))
    img = unnormalize(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_close_and_far(neighbors_data_set, chosen_idxs=None):
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #close
    chosen_indx = []
    print("____closest____")
    if chosen_idxs is None:
        for label in range(10):
            idx, I = neighbors_data_set.get_nearest_neighbors(5, label)
            I = I[0]
            chosen_indx.append(idx)
            img_neighbors, img_neighbors_labels = batch_neighbors(neighbors_data_set, I)
            imshow(torchvision.utils.make_grid(img_neighbors))
            # print(' '.join(f'{classes[img_neighbors_labels[j]]:5s}' for j in range(6)))
    else:
        for i in chosen_idxs:
            idx, I = neighbors_data_set.get_nearest_neighbors(5, idx=i)
            I = I[0]
            img_neighbors, img_neighbors_labels = batch_neighbors(neighbors_data_set, I, i, append=True)
            imshow(torchvision.utils.make_grid(img_neighbors))
        chosen_indx = chosen_idxs
    #far
    print("____furthest____")
    for i in chosen_indx:
        idx, I = neighbors_data_set.get_furthest_neighbors(i,5)
        img_neighbors, img_neighbors_labels = batch_neighbors(neighbors_data_set, I, i, True)
        imshow(torchvision.utils.make_grid(img_neighbors))
        # print(' '.join(f'{classes[img_neighbors_labels[j]]:5s}' for j in range(6)))
    return chosen_indx

def calc_knn_density(index, test_reps, k):
    D, I = index.search(test_reps, k)
    return np.mean(D,axis=1)

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


def show_most_anomalous(inv_density, imgs, k):
    sorted_idxs = np.argsort(inv_density)[-k:]
    anom_imgs = imgs[sorted_idxs]
    imshow(torchvision.utils.make_grid(anom_imgs))

def cluster_kmeans(reps):
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(reps)
    return kmeans.predict(reps), kmeans.cluster_centers_



def plot_tsne_clusters(reduced, kmeans_labels, true_labels, title):

    fig, axes = plt.subplots(1, 2, figsize=(20, 20))

    sns.scatterplot(x=reduced[:-10, 0], y=reduced[:-10, 1], hue=kmeans_labels.astype('str'), ax=axes[0])
    sns.scatterplot(x=reduced[-10:, 0], y=reduced[-10:, 1], color="black", s=130, ax=axes[0])
    axes[0].set_title('Kmeans clusters')

    sns.scatterplot(x=reduced[:-10, 0], y=reduced[:-10, 1], hue=true_labels.astype('str'), ax=axes[1])
    sns.scatterplot(x=reduced[-10:, 0], y=reduced[-10:, 1], color="black", s=130, ax=axes[1])
    axes[1].set_title('True Labels')

    fig.suptitle(title)

    plt.show()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workers = 0 if device.type == 'cpu' else 2



    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
    train_dataset = CfirDataset(trainset.data, trainset.targets,
                                train_transform, True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=workers)

    testset = torchvision.datasets.CIFAR10(root='./data',train=False,
                                                     transform=transforms.ToTensor(),
                                                     download=True)
    test_train_aug_dataset = CfirDataset(testset.data, testset.targets,
                                         train_transform, True)
    test_train_aug_loader = torch.utils.data.DataLoader(test_train_aug_dataset,
                                              batch_size=len(test_train_aug_dataset),
                                            shuffle=True, num_workers=workers)
    torch.cuda.empty_cache()
    org_enc = Encoder(ENCODER_DIM, device).to(device)
    org_proj = Projector(ENCODER_DIM).to(device)

    # Q1
    if os.path.exists("encoder_model_weights.pth"):
        org_enc.load_state_dict(torch.load("encoder_model_weights.pth",
                                        map_location=device))

    else:
        obj_losses,test_losses, test_x_axis = train_VICReg(org_enc, org_proj,
                                                           train_loader,
                                               test_train_aug_loader,
                                                           data_len=len(train_dataset))
        # torch.save(org_enc.state_dict(), "encoder_model_weights.pth")
        [plot_obj_loss(obj_losses,test_losses,i, test_x_axis) for i in range(3)]
    freeze_model(org_enc)
    freeze_model(org_proj)

    train_down_dataset = CfirDataset(trainset.data, trainset.targets,
                                test_transform, False) # for downstream
    train_down_loader = torch.utils.data.DataLoader(train_down_dataset,
                                                    batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                    num_workers=workers)

    test_down_dataset = CfirDataset(testset.data, testset.targets,
                                    test_transform, False) # for downstream



    test_down_loader = torch.utils.data.DataLoader(test_down_dataset,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=workers)

    # Q2
    test_numpy_rep, test_numpy_labels, _ = get_numpy_representations(
        org_enc, test_down_loader)
    # pca
    plot_representation(PCA(n_components=2), test_numpy_rep,
                        test_numpy_labels,
                        "PCA - original VICReg")
    # tsne
    plot_representation(TSNE(n_components=2), test_numpy_rep,
                        test_numpy_labels,
                        "T-SNE - original VICReg")

    # Q3
    fc_prob = train_linear_prob(org_enc, train_down_loader, "org_")
    test_linear_prob(org_enc, fc_prob, test_down_loader, "original")

    # Q4
    torch.cuda.empty_cache()
    no_var_enc = Encoder(ENCODER_DIM, device).to(device)
    no_var_proj = Projector(ENCODER_DIM).to(device)
    if os.path.exists("no_var_enc_weights.pth"):
        no_var_enc.load_state_dict(torch.load("no_var_enc_weights.pth",
                                              map_location=device))

    else:
        train_VICReg(no_var_enc, no_var_proj,
                                                           train_loader,
                                               test_train_aug_loader,mu=0,
                                                           data_len=len(
                                                               train_dataset))
    # torch.save(no_var_enc.state_dict(), "no_var_enc_weights.pth")
    freeze_model(no_var_enc)
    freeze_model(no_var_proj)
    test_numpy_rep_no_var, test_numpy_labels_no_var, _ = get_numpy_representations(no_var_enc, test_down_loader)
    # pca
    plot_representation(PCA(n_components=2), test_numpy_rep_no_var,
                        test_numpy_labels_no_var,
                                        "PCA - No-Variance VICReg")
    # linear prob
    no_var_fc = train_linear_prob(no_var_enc,train_down_loader, "no_var_")
    test_linear_prob(no_var_enc,no_var_fc, test_down_loader, "no variance model")

    # Q6

    org_numpy_train_representation, org_numpy_train_labels, \
                org_numpy_train_imgs = get_numpy_representations(org_enc, train_down_loader)
    org_index = faiss.IndexFlatL2(ENCODER_DIM)
    org_index.add(org_numpy_train_representation)

    torch.cuda.empty_cache()
    no_neighbors_enc = Encoder(ENCODER_DIM, device).to(device)
    no_neighbors_proj = Projector(ENCODER_DIM).to(device)
    org_neighbors_data_set = KNNDataset(3, org_index,
                                        org_numpy_train_representation,
                                        org_numpy_train_imgs,
                                        org_numpy_train_labels)

    org_neighbor_down_loader = torch.utils.data.DataLoader(org_neighbors_data_set,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=2)

    if os.path.exists("no_neighbors_enc_weights.pth"):
        no_neighbors_enc.load_state_dict(torch.load("no_neighbors_enc_weights.pth", map_location=device))
    else:
        no_neighbors_enc.train()
        no_neighbors_proj.train()
        train_no_neighbors_vic_reg(no_neighbors_enc, no_neighbors_proj, org_neighbor_down_loader, 1)
    # torch.save(no_neighbors_enc.state_dict(), "no_neighbors_enc_weights.pth")
    freeze_model(no_neighbors_enc)
    freeze_model(no_neighbors_proj)

    fc_prob_neighbors = train_linear_prob(no_neighbors_enc, train_down_loader, "no_neighbors_")
    test_linear_prob(no_neighbors_enc,fc_prob_neighbors, test_down_loader, "no neighbors")


    # Q8
    no_nb_numpy_train_representation, no_nb_numpy_train_labels, \
                    no_nb_numpy_train_imgs = get_numpy_representations(no_neighbors_enc, train_down_loader)
    no_nb_index = faiss.IndexFlatL2(ENCODER_DIM)
    no_nb_index.add(org_numpy_train_representation)

    no_nb_neighbors_data_set = KNNDataset(3, no_nb_index,
                                        no_nb_numpy_train_representation,
                                        no_nb_numpy_train_imgs,
                                        no_nb_numpy_train_labels)
    print("org model")
    chosen_idxs = show_close_and_far(org_neighbors_data_set)
    print("no neighbors model")
    _ = show_close_and_far(no_nb_neighbors_data_set, chosen_idxs)

    # part II Anomaly Detection

    # Q1
    mnist_transform = transforms.Compose([
        transforms.Lambda(lambda x: transforms.functional.pad(x, (2, 2, 2, 2),
                                                        fill=0)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        test_transform
    ])
    mnistset = torchvision.datasets.MNIST(root='./data', train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
    mnist_cfir_test_dataset = MnistAndCfirDataset(testset.data, testset.targets,
                                    test_transform, mnistset.data, mnistset.targets, mnist_transform) # for downstream
    mnist_cfir_test_loader = torch.utils.data.DataLoader(mnist_cfir_test_dataset,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=2)
    mnist_cfir_org_numpy_test_representation, mnist_cfir_org_numpy_test_labels, \
                    mnist_cfir_org_numpy_test_imgs, mnist_cfir_org_numpy_test_classification\
                     = get_numpy_representations(org_enc, mnist_cfir_test_loader, True)
    mnist_cfir_no_nb_numpy_test_representation, mnist_cfir_no_nb_numpy_test_labels, \
                    mnist_cfir_no_nb_numpy_test_imgs, mnist_cfir_no_nb_numpy_test_classification\
                     = get_numpy_representations(no_neighbors_enc, mnist_cfir_test_loader, True)

    org_inverse_density = calc_knn_density(org_index, mnist_cfir_org_numpy_test_representation, 2)
    no_nb_inverse_density = calc_knn_density(no_nb_index, mnist_cfir_no_nb_numpy_test_representation, 2)

    # Q2
    plot_roc_auc(mnist_cfir_org_numpy_test_classification, org_inverse_density, mnist_cfir_no_nb_numpy_test_classification, no_nb_inverse_density)

    # Q3
    print("org model most anomalous")
    show_most_anomalous(org_inverse_density, mnist_cfir_org_numpy_test_imgs, 7)
    print("no neighbor model most anomalous")
    show_most_anomalous(no_nb_inverse_density, mnist_cfir_no_nb_numpy_test_imgs, 7)

    # Part III Clustering
    # Q1
    org_kmeans_labels, org_cluster_centers = cluster_kmeans(org_numpy_train_representation)
    no_nb_kmeans_labels, no_nb_cluster_centers = cluster_kmeans(no_nb_numpy_train_representation)



    # Q2
    torch.cuda.empty_cache()
    tsne = TSNE(n_components=2)
    org_reduced = tsne.fit_transform(np.vstack([org_numpy_train_representation, org_cluster_centers]))
    tsne = TSNE(n_components=2)
    no_nb_reduced = tsne.fit_transform(np.vstack([no_nb_numpy_train_representation, no_nb_cluster_centers]))

    plot_tsne_clusters(org_reduced, org_kmeans_labels, org_numpy_train_labels, "original model cluster representation")
    plot_tsne_clusters(no_nb_reduced, no_nb_kmeans_labels, no_nb_numpy_train_labels, "no neighbors model cluster representation")

    # Q3
    print("Silhouette Scores:")
    print("orginal model", silhouette_score(org_numpy_train_representation, org_kmeans_labels))
    print("no neighbors model", silhouette_score(no_nb_numpy_train_representation, no_nb_kmeans_labels))