import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_image_pair(image, processed_image, directory, name):
    """

    :param image: tuple (image np_array, title, cmap)
    :param processed_image: tuple (image np_array, title, cmap)
    :param directory: The directory to save
    :param name: The file name
    :return:
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    fig.subplots_adjust(hspace=.05, wspace=.05)
    axes[0].imshow(image[0], image[2])
    #axes[0].axis('off')
    axes[0].set_title(image[1], fontsize=8)
    axes[1].imshow(((processed_image[0]+0.5)*255).astype(np.uint8), processed_image[2])
    #axes[1].axis('off')
    axes[1].set_title(processed_image[1], fontsize=8)
    filename = directory + name
    plt.tight_layout()
    fig.savefig(filename)


def save_training_images(imgs, steer, filename):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 5.5))
    fig.subplots_adjust(hspace=.2, wspace=.05)
    for i, img in enumerate(imgs):
        axes[i // 4, i % 4].imshow(imgs[i])
        axes[i // 4, i % 4].axis('off')
        axes[i // 4, i % 4].set_title("Steering {0:.4f}".format(steer[i]), fontsize=8)
    fig.savefig(filename)


def save_steering_histogram(y_train, title, filename):
    fig = plt.figure(figsize=(5, 5))
    hist, bins = np.histogram(y_train, bins=50, range=(-1, 1))
    plt.bar(bins[:-1], hist, align='center', width=0.1)
    plt.title(title)
    plt.ylabel('Number of Occurrences')
    #plt.xticks(bins, , rotation='vertical')
    plt.tight_layout()
    fig.savefig(filename)


def save_steering_signal(signal, time_range, filename):
    fig = plt.figure(figsize=(8, 4))
    plt.title('Steering Signal')
    plt.xlabel("Time (event {} to {})".format(time_range[0], time_range[1]))
    plt.ylim(-1, 1)
    plt.plot(signal[time_range[0]:time_range[1]], linewidth=1)
    pos = [r for r in range(0, time_range[1]-time_range[0], 100)]
    values = [str(r) for r in range(time_range[0], time_range[1], 100)]
    plt.xticks(pos, values)
    plt.tight_layout()
    fig.savefig(filename)