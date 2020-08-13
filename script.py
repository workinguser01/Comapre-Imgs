import os
# from skimage.measure import structural_similarity as ssim
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2

base_dir = os.path.dirname(os.path.abspath(__file__))


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
	return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = measure.compare_ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()


# load the images -- the original, the original + contrast,
# and the original + photoshop
original = cv2.imread(base_dir + "/images/origin.png")
contrast = cv2.imread(base_dir + "/images/contrast.png")
shopped = cv2.imread(base_dir + "/images/photoshoped.png")
jis1 = cv2.imread(base_dir + "/images/jis1.png")
jis2 = cv2.imread(base_dir + "/images/jis2.png")
jis3 = cv2.imread(base_dir + "/images/jis3.png")

# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)
jis1 = cv2.cvtColor(jis1, cv2.COLOR_BGR2GRAY)
jis2 = cv2.cvtColor(jis2, cv2.COLOR_BGR2GRAY)
jis3 = cv2.cvtColor(jis3, cv2.COLOR_BGR2GRAY)

compare_images(original, jis1, 'o-c')
