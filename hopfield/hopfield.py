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
from random import randint
from PIL import Image
from hopfieldnetwork import HopfieldNetwork
from hopfieldnetwork import images2xi

TEST_PATH = './data/test/*.png'
TRAIN_PATH = './data/train/*.png'
REF_PATH = './data/reference/*.png'


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


def noise_image(file, percent):
    print(file)

    img = Image.open(file).convert('L')
    img = np.array(img)
    w, h = img.shape
    for i in range(w):
        for j in range(h):
            if randint(0, 100) < percent:
                img[i][j] = 255 - img[i][j]
    img = Image.fromarray(img)

    file = file.split(".png")[0]
    file = file.split("\\")[1]
    filename = ".\\data\\test\\" + file + str(percent) + ".png"

    img.save(filename)


def get_train_images_as_reference():
    for file in glob.glob(TRAIN_PATH):
        print(file)
        img_test = images2xi([file], N)
        network.set_initial_neurons_state(np.copy(img_test[:, 0]))
        network.update_neurons(iterations, 'sync')

        test_pil = xi_to_PIL(img_test[:, 0], N)
        res_pil = xi_to_PIL(network.S, N)

        file = file.split(".png")[0]
        file = file.split("\\")[1]
        filename = ".\\data\\reference\\" + file + ".png"
        test_pil.save(filename)


def predict(res_pil):
    mses = {}
    res_pil.save("result.png")
    res_pil = Image.open("result.png").convert('L')
    for file in glob.glob(REF_PATH):
        reference_img = Image.open(file).convert('L')
        mses[file] = mse_img(res_pil, reference_img)

    lowest_mse = float('inf')
    lowest_name = None
    for filename, mse in mses.items():
        if mse < lowest_mse:
            lowest_mse = mse
            lowest_name = filename

    lowest_name = lowest_name.split("\\")[1]
    lowest_name = lowest_name.split(".png")[0]
    lowest_name = lowest_name.split("_")[0]
    print(f"Predykcja -> {lowest_name.capitalize()} z mse={lowest_mse}")
    return mses


def mse_img(img1, img2):
    img1, img2 = np.array(img1), np.array(img2)
    sum = 0
    w, h = img1.shape[:2]
    for x in range(w):
        for y in range(h):
            diff = img1[x, y] - img2[x, y]
            sum += diff
    return sum / (w * h)


if __name__ == '__main__':
    N = 100 ** 2
    iterations = 5
    target = []

    for file in glob.glob(TRAIN_PATH):
        target.append(file)

    target = images2xi(target, N)
    network = HopfieldNetwork(N=N)

    print("TRAINING")
    network.train_pattern(target)
    network.save_network("./hopefield_network.npz")


    print("TESTING")
    network.load_network("./hopefield_network.npz")
    for file in glob.glob(TEST_PATH):
        print(file)
        img_test = images2xi([file], N)
        network.set_initial_neurons_state(np.copy(img_test[:, 0]))
        network.update_neurons(iterations, 'async')

        test_pil = xi_to_PIL(img_test[:, 0], N)
        res_pil = xi_to_PIL(network.S, N)

        predict(res_pil)

        show_images([test_pil, res_pil])
