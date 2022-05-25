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
from hopfieldnetwork import HopfieldNetwork
from hopfieldnetwork import images2xi

IMAGE_SIZE = (30, 30, 3)
TEST_PATH = './data/test/*.png'
TRAIN_PATH = './data/train/*.png'


def convert_to_black_and_white(name):
    img = Image.open(name)
    thresh = 200
    fn = lambda x: 255 if x > thresh else 0
    r = img.convert('L').point(fn, mode='1')
    r.save('out2.png')


from PIL import Image


def img2array(name):
    convert_to_black_and_white(name)
    img = cv2.imread("out2.png")
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_AREA)
    img = img.flatten()
    binary_image = []
    for i in img:
        if i > 230:
            binary_image.append(1)
        else:
            binary_image.append(0)

    return binary_image


def array2img(array):
    array[array == -1] = 0
    array *= 255
    img = np.reshape(array, IMAGE_SIZE)
    cv2.imshow("array2img", img)
    cv2.waitKey(0)
    return img


def array2float(array):
    tmp = np.asfarray(array)
    tmp[tmp == 0] = -1.0
    return tmp


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
    iterations = 5
    target = []
    for file in glob.glob(TRAIN_PATH):
        # array = img2array(file)
        if file not in ("./data/train\shrek_color.png", "./data/train\\fiona_color.png"):
            continue
        target.append(file)

    target = images2xi(target, N)
    network = HopfieldNetwork(N=N)

    print("TRAINING")
    network.train_pattern(target)
    network.save_network("./hopefield_network.npz")

    print("TESTING")
    # network.load_network("./hopefield_network.npz")

    img_test = images2xi(['./data/test/fiona_color.png'], N)
    half_image = np.copy(img_test)
    half_image[: int(N / (100 / 50))] = -1

    network.set_initial_neurons_state(np.copy(img_test[:, 0]))
    # network.set_initial_neurons_state(np.copy(half_image[:, 0]))
    network.update_neurons(iterations, 'async')
    test_pil = xi_to_PIL(img_test[:, 0], N)
    # test_pil = xi_to_PIL(half_image, N)
    res_pil = xi_to_PIL(network.S, N)

    # out_image = array2img(out[0]) #output network image
    # img_test = array2img(test) #test image
    show_images([test_pil, res_pil])
