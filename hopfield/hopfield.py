"""
Example of use Hopfield Recurrent network
=========================================

Task: Recognition of Simpsons

"""
import cv2, glob
import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt
from PIL import Image

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
  if name == './data/train/shrek.png':
    cv2.imshow("x", img)
    cv2.waitKey(0)
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



target = []
for file in glob.glob(TRAIN_PATH):
  array = img2array(file)
  target.append(array)

target = array2float(target)

net = nl.load("noise_cancellation_model30x30maxinit10.net")
# net = nl.net.newhop(target, max_init=1) # Create and train network
# net.save("noise_cancellation_model30x30maxinit1.net")

img_test = img2array('./data/test/shrek.png')

test = array2float(img_test)


out = net.sim([test]) #test network

out_image = array2img(out[0]) #output network image
img_test = array2img(test) #test image
show_images([img_test, out_image])