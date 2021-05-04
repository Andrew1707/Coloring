import sys
import numpy as np
import copy
from PIL import Image
import math
import random
import Color_Coded as cc
import matplotlib.pyplot as plt

### Model Idea: For every 3x3 greyscale patch, multiply
### each pixel val by a weight vector to obtain prediction
### values for each of the  k colors from get_clusters fxn


# Returns dictionary of (pixel index:cluster center) pairs using Lloyd's k-means clustering algorithm
# INPUTS: k is the number of clusters, data is a list of data RGB tuples
def recolor(k, data):
    # Pick random centers from the data list to start
    centers = cc.random_picks(k, data)

    # RSS is the residual sum of squares (total error squared of each point's dist to its cluster center)
    old_RSS, RSS = 0, 0
    centers_changed = True

    while centers_changed:
        # Create dictionary mapping data tuple locations to centers
        dictionary = dict()
        old_RSS = RSS
        RSS = 0

        # RGB distances are weighted evenly (using square norm for dist)
        for i in range(len(data)):
            # The index of the closest center to i
            closest = 0
            # Initialize distance between i and closest center
            dist = float("inf")

            # Iterate through centers
            for c in range(0, k):
                temp = 0
                # Iterate through each R/G/B pixel value
                for val in range(3):
                    temp += (centers[c][val] - data[i][val]) ** 2
                # If distance to center c is shorter
                if temp < dist:
                    dist = temp
                    closest = c

            dictionary.update({i: closest})

        # Now adjust center locations
        # cluster is a list of [count, avg] lists corresponding to the center at its index
        cluster = []
        # Initialize cluster structure
        for i in range(k):
            cluster.append([0, [0, 0, 0]])
        # Sum up which color triplets belong to which cluster
        for i in range(len(data)):
            centers_index = dictionary.get(i)
            cluster[centers_index][0] += 1

            for rgb in range(3):
                cluster[centers_index][1][rgb] += data[i][rgb]

                # Update RSS
                RSS += (centers[centers_index][rgb] - data[i][rgb]) ** 2
        # Average out the [~, avg] part of each pair in cluster
        for i in range(k):
            for rgb in range(3):
                if cluster[i][0] != 0:
                    cluster[i][1][rgb] = int(cluster[i][1][rgb] / cluster[i][0])

        # Updates centers
        centers_changed = False
        for i in range(k):
            if centers[i] != cluster[i][1]:
                centers[i] = cluster[i][1]
                centers_changed = True

        # Early termination condition (close enough to solution according to prev testing)
        if abs(old_RSS - RSS) < (old_RSS * 0.0001):
            centers_changed = False

    # Replace centers indicies in dictionary with finalized colors
    # rep_colors is copy of data except w/ RGB vals replaced w/ corresponding cluster colors
    rep_colors = np.array(data)
    for key in dictionary:
        new_val = np.array(centers[dictionary.get(key)])
        rep_colors[key] = new_val

    return rep_colors, RSS


def initialize_weights(k):
    weights = np.empty((9, k))
    for i in range(9):
        for j in range(k):
            weights[i][j] = random.uniform(-1, 1)

    return weights


# Loss calculator
def loss():
    return


# Loss gradient calculator
def loss_grad():
    return


def trainer(weights, patch):
    raw_preds = np.cross(patch, weights)

    # Max portion here
    return


def start():
    # Relative path
    original = Image.open("C:Nature\small_nature.jpeg")
    # original = Image.open("C:Images\EvenSmaller.png")
    k = 5  #! Might not need

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
    training_as_list = list(training.getdata())
    training_grayscale_as_list = np.array(list(training_grayscale.getdata()))

    # Run get clusters, get np array of RGB vals that replace the original RGBs in training_as_list
    RSS_list = []
    num_list = []
    for num in range(1, 20):
        print(num)
        training_recolored, RSS = recolor(num, training_as_list)
        RSS_list.append(RSS)
        num_list.append(num)
    plt.xlabel("k clusters")
    plt.ylabel("RSS")
    plt.plot(num_list, RSS_list, color="red", label="Smart")
    plt.legend(loc="best")
    plt.show()
    return

    # Append training arrays so cols 1-3 are recolored RGB, col 4 is corresponding grey val
    training_grayscale_as_list = np.reshape(training_grayscale_as_list, (training_grayscale_as_list.size, 1))
    combined_ds = np.append(training_recolored, training_grayscale_as_list, axis=1)

    # Shuffle combined_ds, then split into train_ds and val_ds
    np.random.shuffle(combined_ds)
    train_ds = combined_ds[:, -1]
    val_ds = combined_ds[:, :-1]

    # Initalize weights
    weights = initialize_weights(k)  #! Might not need k, so reupdate

    return


start()
