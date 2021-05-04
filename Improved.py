import sys
import numpy as np
import copy
from PIL import Image
import math
import random
import Color_Coded as cc

### Model Idea: For every 3x3 greyscale patch, multiply
### each pixel val by a weight vector to obtain prediction
### values for each of the  k colors from get_clusters fxn


def initialize_weights(k):
    weights = np.empty((9,k))
    for i in range(9):
        for j in range(k):
            weights[i][j] = random.uniform(-1,1)

    return weights


def color_classifier(weights, patch):
    raw_preds = np.dot(patch, weights)

    # Max portion here
    return


def main():
    # Relative path
    original = Image.open("C:Images\\nature.jpeg")


    img = original.convert("L")  # grayscale copy
    width, height = img.size

    # Create left (training) and right (testing) image halves
    training = original.crop((1, 1, (width / 2) + 1, height + 1))
    testing = img.crop(((width / 2), 1, width, height + 1))  #!plus 1 to get them to equal size
    training_grayscale = img.crop((1, 1, (width / 2) + 1, height + 1))

    # need sizes of all halves
    half_width, half_height = training.size

    # Convert to numpy array
    testing_as_list = np.array(list(testing.getdata()))
    training_as_list = np.array(list(training.getdata()))
    training_grayscale_as_list = np.array(list(training_grayscale.getdata()))

    clusters = cc.get_clusters(5, training_as_list)
    tuple_clusters = {}
    for x, c in enumerate(clusters):
        RGB = (clusters[c][0], clusters[c][1], clusters[c][2])
        tuple_clusters.update({x: RGB})
    
    
    return

main()