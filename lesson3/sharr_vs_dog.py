import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d as conv2


def precision_recall(original, result):
    result[result > 0] = 1

    tp = np.sum((result[original == 1]) == 1)
    fp = np.sum((result[original == 0]) == 1)
    fn = np.sum((result[original == 1]) == 0)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def gauss(size, sigma):
    center = (size - 1.) / 2.
    y, x = np.ogrid[-center:center + 1, -center:center + 1]
    psf = np.exp(-(x**2 + y**2) / (2. * sigma**2))
    psf[psf < np.finfo(psf.dtype).eps * psf.max()] = 0
    sm = psf.sum()
    if sm != 0:
        psf /= sm
    return psf


def diff_of_gaussian(size=5, sigma1=20, sigma2=15):
    if sigma2 > sigma1:
        sigma1, sigma2 = sigma2, sigma1
    return gauss(size, sigma1) - gauss(size, sigma2)


if __name__ == '__main__':
    win_size = 5

    image = np.array(cv2.imread('kovrov_gray.bmp', 0))
    mask = np.array(cv2.imread('kovrov_edges.bmp', 0))

    # Show input data
    plt.figure()
    plt.imshow(mask, cmap=plt.cm.gray)
    plt.title('Mask')
    plt.axis('off')

    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Original image')
    plt.axis('off')

    mask[mask < mask.max()] = 0
    mask[mask > 0] = 1

    # Find edges
    scharr_edges = np.sqrt(cv2.Scharr(image, cv2.CV_64F, 1, 0)**2 +
                           cv2.Scharr(image, cv2.CV_64F, 0, 1)**2)

    dog = diff_of_gaussian()
    dog_edges = conv2(image.astype(float), dog, 'same')

    precision_sharr = []
    recall_sharr = []

    precision_dog = []
    recall_dog = []

    for threshold in np.arange(0, 0.11, 0.01):

        sch = scharr_edges
        sch[sch < threshold * np.max(sch)] = 0
        sch = sch / np.max(sch)

        dg = dog_edges
        dg[dg < threshold * np.max(dg)] = 0
        dg = dg / np.max(dg)

        p, r = precision_recall(mask, sch)
        precision_sharr.append(p)
        recall_sharr.append(r)

        p, r = precision_recall(mask, dg)
        precision_dog.append(p)
        recall_dog.append(r)

    # Plot last results
    plt.figure()
    plt.imshow(dg, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('DoG. Threshold = {} * max'.format(threshold))

    plt.figure()
    plt.imshow(sch, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Scharr. Threshold = {} * max'.format(threshold))

    # Plot precision-recall curve
    plt.figure()
    plt.plot(recall_sharr, precision_sharr, '-*b', label='Scharr')
    plt.plot(recall_dog, precision_dog, '-*r', label='DoG')

    plt.plot(recall_sharr[-1], precision_sharr[-1], 'oy', label='Last point of Scharr')
    plt.plot(recall_dog[-1], precision_dog[-1], 'og', label='Last point of DoG')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc=1, fontsize='x-small')
    plt.grid()

    plt.show()
