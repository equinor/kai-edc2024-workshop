import pickle
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
sns.set_theme()


def check_accuracy_language(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():  # dont compute gradients
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        acc = float(num_correct)/float(num_samples)*100
        print(f'Got {num_correct}/{num_samples} with accuracy {acc: .2f}')

    model.train()

    return acc


def get_shakespeare_dataloader(window_size=10, batch_size=16):
    data = get_shakespeare_text()

    # Partition data
    n_letters = len(data)
    n_size = int(n_letters * 0.8)
    train = data[:n_size]
    test = data[n_size:]

    train_loader = ShakespeareDataLoader(train, window_size, batch_size)
    test_loader = ShakespeareDataLoader(test, window_size, batch_size)

    return train_loader, test_loader


def get_shakespeare_text():
    directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(directory, 'data', 'shakespeare.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def get_midsummer_night_dream():
    directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(directory, 'data', 'midsummer-night.txt')
    with open(path, 'r') as f:
        data = f.read()

    return data


def plot_accuracy(accuracy):
    n_epochs = len(accuracy)
    epoch = np.arange(1, n_epochs+1)
    plt.plot(epoch, accuracy, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def plot_letter_count(bin1, bin2, bin3):
    # Create layout
    my_dpi = 192
    fig = plt.figure(figsize=(2250 // my_dpi, 900 // my_dpi), dpi=150)
    outer = gridspec.GridSpec(1, 3)

    # Create axes
    axes = []
    axes.append(plt.Subplot(fig, outer[0]))
    axes.append(plt.Subplot(fig, outer[1]))
    axes.append(plt.Subplot(fig, outer[2]))

    # Add axes to figure
    for ax in axes:
        fig.add_subplot(ax)

    letters, counts = np.unique(bin1, return_counts=True)
    idx = np.argsort(counts)[::-1]
    sns.barplot(x=letters[idx], y=counts[idx], ax=axes[0])

    letters, counts = np.unique(bin2, return_counts=True)
    idx = np.argsort(counts)[::-1]
    sns.barplot(x=letters[idx], y=counts[idx], ax=axes[1])

    letters, counts = np.unique(bin3, return_counts=True)
    idx = np.argsort(counts)[::-1]
    sns.barplot(x=letters[idx], y=counts[idx], ax=axes[2])

    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Next letter')
    axes[1].set_xlabel('Next letter')
    axes[2].set_xlabel('Next letter')

    plt.show()


class Vocabulary():
    def __init__(self, text):
        unique_letters = np.unique(list(text))
        self.letter2index = {letter: idx for idx, letter in enumerate(unique_letters)}
        self.index2letter = {idx: letter for idx, letter in enumerate(unique_letters)}

    def get_letters(self, indices):
        out = []
        for idx in indices:
            out.append(self.index2letter[idx])

        return out

    def get_indices(self, letters):
        out = []
        for letter in letters:
            out.append(self.letter2index[letter])

        return out


class ShakespeareDataLoader:
    def __init__(self, text, window_size=10, batch_size=32):
        text = text.lower()
        self.vocab = Vocabulary(text)
        data = self._batch_data(text, window_size, batch_size)
        self.data = torch.tensor(data)

    def __iter__(self):
        # Shuffle
        n_batches = len(self.data)
        idx = np.arange(n_batches)
        np.random.shuffle(idx)
        self.data = self.data[idx]

        for data in self.data:
            context = data[:, :-1]
            letter = data[:, -1].to(dtype=int)

            yield context, letter

    def _batch_data(self, data, window_size, batch_size):
        # Encode datasets
        data = self.vocab.get_indices(data)

        # Batch data
        n_samples = len(data) - window_size - 1
        samples = np.empty(shape=(n_samples, window_size+1))
        for idx in range(n_samples):
            samples[idx] = data[idx:idx+window_size+1]

        n_batches = n_samples // batch_size
        batches = np.empty((n_batches, batch_size, window_size+1))
        for idb in range(n_batches):
            for ids in range(batch_size):
                batches[idb, ids] = samples[idb*batch_size+ids]

        return batches
