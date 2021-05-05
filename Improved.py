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
    centers = np.array(centers)
    rep_colors = np.array(data)
    for key in dictionary:
        new_val = np.array(centers[dictionary.get(key)])
        rep_colors[key] = new_val

    return rep_colors, RSS, centers


def initialize_weights(k):
    weights = np.empty((9, k))
    for i in range(9):
        for j in range(k):
            weights[i][j] = random.uniform(-1, 1)

    return weights


# Loss calculator
def loss(softmaxed_arr, index):
    return -1 * math.log10(softmaxed_arr[index])


# Loss gradient calculator
def softmax_loss_grad(softmaxed_arr, val_arr):
    grad = softmaxed_arr - val_arr
    return grad


# Returns softmaxed array (adjusted implementation to prevent overflow)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def improvedModel(weights, patch):
    raw_preds = np.matmul(patch, weights)
    softmaxed_arr = softmax(raw_preds)
    return softmaxed_arr


def update_weights(lr, weights, patch, soft_gradient):
    for i in range(len(weights[0])):
        weights[:,i] = weights[:,i] - (lr * soft_gradient[i] * patch)
    return weights


# Trainer
def train(weights, train_ds, val_ds, patch_ds, height, width, centers):
    # Initialize learning rate (lr)
    lr = 1e5
    
    # Call model and update_weights here
    for i in range(len(patch_ds)):
        train_patch = patch_ds[i]

        # Get predictions through model
        predictions = improvedModel(weights, train_patch)

        # val_ds[i] is corresponding RGB array, make one hot encoded arr (validation array)
        one_hot_encoded_arr = np.zeros(weights.shape[1])
        one_index = 0
        for j in range(len(centers)):
            if np.array_equal(centers[j], val_ds[i]):
                one_index = j
                break
        one_hot_encoded_arr[one_index] = 1

        # Forced learning rate decay
        if i == int(len(train_ds)/4):
            lr = 1e4
        elif i == int(len(train_ds)/2):
            lr = 1e3

        # Update weights
        # loss_val = loss(predictions, one_index)
        # print(f'i: {i}, loss_val: {loss_val}') #! For plotting
        soft_gradient = softmax_loss_grad(predictions, one_hot_encoded_arr)
        weights = update_weights(lr, weights, train_patch, soft_gradient)
    
    return weights


def start():
    # Relative path
    original = Image.open("C:Nature\\small_nature.jpeg")
    # original = Image.open("C:Images\\EvenSmaller.png")
    k = 10  # Adjusted number of clusters

    img = original.convert("L")  # grayscale copy
    width, height = img.size

    # Create left (training) and right (testing) image halves
    training = original.crop((1, 1, (width / 2) + 1, height + 1))
    testing = img.crop(((width / 2), 1, width, height + 1))  #!plus 1 to get them to equal size
    training_grayscale = img.crop((1, 1, (width / 2) + 1, height + 1))
    testing.show()
    original.close()

    # need sizes of all halves
    half_width, half_height = training.size

    # Convert to numpy array
    testing_as_list = np.array(list(testing.getdata()))
    training_as_list = list(training.getdata()) #! Do not convert to np array
    training_grayscale_as_list = np.array(list(training_grayscale.getdata()))

    # Run get clusters, get np array of RGB vals that replace the original RGBs in training_as_list
    training_recolored, _, centers = recolor(k, training_as_list)
    
    # # Code for RSS vs k plot
    # RSS_list = []
    # num_list = []
    # for num in range(1, 20):
    #     print(num)
    #     training_recolored, RSS = recolor(num, training_as_list)
    #     RSS_list.append(RSS)
    #     num_list.append(num)
    # plt.xlabel("k clusters")
    # plt.ylabel("RSS")
    # plt.plot(num_list, RSS_list, color="red", label="Smart")
    # plt.legend(loc="best")
    # plt.show()
    # return


    # Append training arrays so cols 1-3 are recolored RGB
    # col 4 is corresponding grey val
    # col 5 w/ added 3rd dimension for entire arr, 9 deep (for patches)
    # skim the top surface (3rd dim 0th index) for cols 1-4, grab col 5 w/ 3rd dim
    combined_obj = np.zeros((training_grayscale_as_list.size, 5, 9))
    combined_obj[:, :3, 0] = training_recolored
    combined_obj[:, 3, 0] = training_grayscale_as_list
    
    # Add greyscale patches to combined_obj
    for i in range(len(training_grayscale_as_list)):
        combined_obj[i, 4, :] = np.array(cc.get3x3(training_grayscale_as_list, half_height, half_width, i))
    
    # Clear out all edges (no corresponding 3x3 patch)
    i = 0
    stop_index = len(training_grayscale_as_list)
    while i < stop_index:
        if math.isnan(combined_obj[i, 4, 0]):
            combined_obj = np.delete(combined_obj, i, 0)
            i -= 1
            stop_index -= 1
        i += 1
    
    # Rescale vals from [0, 255] to [0, 1], shuffle combined_ds, then split
    combined_obj = combined_obj / 255
    centers = centers / 255

    np.random.shuffle(combined_obj)
    train_ds = combined_obj[:, 3, 0]
    val_ds = combined_obj[:, :3, 0]
    patch_ds = combined_obj[:, 4, :]

    # Initalize weights
    weights = initialize_weights(k)
    weights = train(weights, train_ds, val_ds, patch_ds, half_width, half_height, centers)

    # Apply to test image
    test_rgb = np.zeros((testing_as_list.size,3))
    for i in range(len(testing_as_list)):
        temp_patch = np.array(cc.get3x3(testing_as_list, half_height, half_width, i))
        if temp_patch.size == 1:
            continue
        temp_patch = temp_patch / 255

        temp_pred = improvedModel(weights, temp_patch)
        new_color = centers[np.argmax(temp_pred)]
        test_rgb[i] = new_color

    # Rescale new RGB image back up to [0, 255] and display
    test_rgb = test_rgb * 255
    test_rgb = np.reshape(test_rgb, (half_height, half_width, 3))

    new_img = Image.fromarray(np.uint8(test_rgb), mode='RGB')
    new_img.show()
    
    return


start()