import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import itertools


# Class with reusable methods to plot data in different styles
class Vizualizer:
    @staticmethod
    def plot_line(data, index_col, columns, n, m, label=None):
        if label is None:
            label = columns
        fig, axs = plt.subplots(nrows=n, ncols=m, sharex=True, sharey=False)

        for i in range(0, len(columns)):
            if m > 1:
                axs[int(i / m)][i % m].plot(data[index_col], data[columns[i]], label=label[i])
                axs[int(i / m)][i % m].legend()
            else:
                axs[i].plot(data[index_col], data[columns[i]], label=label[i])
                axs[i].legend()
        plt.show()

    @staticmethod
    def plot_scatter(data, index_col, columns, n, m, label=None):
        if label is None:
            label = columns
        fig, axs = plt.subplots(nrows=n, ncols=m, sharex=True, sharey=False)

        for i in range(0, len(columns)):
            axs[i].scatter(data[index_col], data[columns[i]])
            axs[i].legend(label[i])

        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False):
        # Select the colormap.
        cmap = plt.cm.Blues
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
