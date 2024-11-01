import json
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():  # dont compute gradients
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            if 'LeNet' not in str(model):
                x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100: .2f}')

    model.train()


def get_qa_datatset():
    directory = os.path.dirname(os.getcwd())
    filepath = os.path.join(directory, 'kai', 'data', 'questions_answers.json')

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def get_modified_mnist_data(batch_size=64):
    # Load Data
    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = PaddedDataLoader(train_loader)
    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    test_loader1 = PaddedDataLoader(test_loader)
    test_loader2 = PaddedDataLoader(test_loader, padding='other')

    return train_loader, test_loader1, test_loader2


def plot_loss(loss):
    n_iterations = len(loss)
    iterations = np.arange(1, n_iterations+1)
    plt.plot(iterations, loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


def plot_images(images, labels):
    # Visualise images
    # Create layout
    my_dpi = 192
    fig = plt.figure(figsize=(2250 // my_dpi, 1200 // my_dpi), dpi=150)
    outer = gridspec.GridSpec(1, 4)

    # Create axes
    axes = []
    axes.append(plt.Subplot(fig, outer[0]))
    axes.append(plt.Subplot(fig, outer[1]))
    axes.append(plt.Subplot(fig, outer[2]))
    axes.append(plt.Subplot(fig, outer[3]))

    # Add axes to figure
    for ax in axes:
        fig.add_subplot(ax)

    for ida, ax in enumerate(axes):
        ax.imshow(images[ida][0], cmap='gray')
        ax.set_title(f'Label: {labels[ida]}')
    plt.show()


def train_mnist_model(num_epochs, model, objective, optimizer, train_loader, device):
    losses = []
    for epoch in range(num_epochs):
    # 1 epoch => Network has seen all the images in the dataset

        print(f'Epoch: {epoch}')

        for batch_idx, (data, targets) in enumerate(train_loader):

            data = data.to(device=device)
            targets = targets.to(device=device)

            if 'LeNet' not in str(model):
                data = data.reshape(data.shape[0], -1)  # Flatten

            scores = model(data)
            loss = objective(scores, targets)

            # backward
            optimizer.zero_grad()  # set all gradients to zero for each batch
            loss.backward()

            # gradient descent
            optimizer.step()
            losses.append(loss.detach().numpy())

    plot_loss(losses)



class PaddedDataLoader:
    def __init__(self, dataloader, padding='top-left'):
         self.dataloader = dataloader
         self.padding = padding

    def __iter__(self):
        for images, labels in self.dataloader:
            # Padd images
            n_batch, n_channels, n_width, n_height = images.shape
            padded = torch.zeros(size=(n_batch, n_channels, n_width*2, n_height*2))
            if self.padding == 'top-left':
                padded[:, :, :n_width, :n_height] = images
            else:
                padded[:, :, -n_width:, -n_height:] = images

            yield padded, labels
