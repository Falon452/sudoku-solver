"""
Example of use Hopfield Recurrent network
=========================================

Task: Recognition of Simpsons

"""
import io
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from hopfieldnetwork import HopfieldNetwork
from hopfieldnetwork import images2xi

TEST_PATH = './data/test/*.png'
TRAIN_PATH = './data/train/*.png'


def show_images(images):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(images[0])
    ax.set_title('Test')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(images[1])
    ax.set_title('Result')

    plt.show()


# Utility function, which changes xi from the hopfield network library to a PIL image
def xi_to_PIL(xi, N):
    N_sqrt = int(np.sqrt(N))
    image_arr = np.uint8((xi.reshape(N_sqrt, N_sqrt) + 1) * 90)
    plt.matshow(image_arr, cmap="Blues")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    im = Image.open(img_buf)
    # img_buf.close()
    return im


if __name__ == '__main__':
    N = 100**2
    iterations = 10
    target = []
    for file in glob.glob(TRAIN_PATH):
        if file not in ("./data/train\shrek_color.png", "./data/train\\fiona_color.png"):
            continue
        target.append(file)

    target = images2xi(target, N)
    network = HopfieldNetwork(N=N)

    print("TRAINING")
    network.train_pattern(target)
    network.save_network("./hopefield_network.npz")

    print("TESTING")
    network.load_network("./hopefield_network.npz")

    img_test = images2xi(['./data/test/fiona_color.png'], N)

    network.set_initial_neurons_state(np.copy(img_test[:, 0]))
    network.update_neurons(iterations, 'async')
    test_pil = xi_to_PIL(img_test[:, 0], N)
    res_pil = xi_to_PIL(network.S, N)

    show_images([test_pil, res_pil])
