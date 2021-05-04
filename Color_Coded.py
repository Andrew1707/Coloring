import sys
import numpy as np
import copy
from PIL import Image
import math
import random


# return a 3x3 list of the pixels around num
def get3x3(img_as_list, height, width, num):
    if num < width:
        return None
    if num >= len(img_as_list) - width:
        return None
    if num % width == 0:
        return None
    if (num + 1) % width == 0:
        return None

    topleft = img_as_list[num - width - 1]
    top = img_as_list[num - width]
    topright = img_as_list[num - width + 1]
    midleft = img_as_list[num - 1]
    mid = img_as_list[num]
    midright = img_as_list[num + 1]
    bottomleft = img_as_list[num + width - 1]
    bottom = img_as_list[num + width]
    bottomright = img_as_list[num + width + 1]

    new_list = [topleft, top, topright, midleft, mid, midright, bottomleft, bottom, bottomright]
    return new_list


# RGB list to GrayScale list converter (grayscale in 0-1 format not 0-255)
def dulldown(RGB):
    grayscale = []
    for x in RGB:
        gray = 0.21 * x[0] + 0.72 * x[1] + 0.07 * x[2]
        grayscale.append(gray / 255)
    return grayscale


# Returns a list of k randomly picked elements from a list l
def random_picks(k, l):
    output = []
    already_picked = set()
    i = 0
    while i < k:
        x = random.randint(0, len(l) - 1)
        # If the element in data hasn't already been picked
        if already_picked & {x} == set():
            output.append(l[x])
            already_picked.add(x)
        else:
            i -= 1  # we try again
        i += 1
    return output


# Returns dictionary of (pixel index:cluster center) pairs using Lloyd's k-means clustering algorithm
# INPUTS: k is the number of clusters, data is a list of data RGB tuples
def get_clusters(k, data):
    # Pick random centers from the data list to start
    centers = random_picks(k, data)

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
    # rep_colors is the dictionary of (pixel index : representative) colors for the training image
    rep_colors = dict()
    for key in dictionary:
        new_val = centers[dictionary.get(key)]
        rep_colors.update({key: new_val})

    return rep_colors


# num patches is how many best patches do you want to save for determination of rep color
# find the most similar patches to represent the gray scale
# training and testing are list of data
# height and width are size of image
# rep colors is each pixels rep color from k clustering
# returns data of colored testing image
def color_mapping(num_patches, training, training_gray, testing, height, width, rep_colors):

    training_gray_patches = []

    # get a list of all the patches in the gray training data
    for x in range(len(training_gray)):
        patch = get3x3(training_gray, height, width, x)
        training_gray_patches.append(patch)

    new_image = []
    # compare the testing patch to training patch and keep best numpatches representatives
    for x in range(len(testing)):
        # time stamp for every 5 percent
        if (x / len(testing)) % 0.05 < 0.0001:
            print(x / len(testing))
        testing_patch = get3x3(testing, height, width, x)
        best = []

        # set up 'best' representative patches
        # tuple is (index of best rep, total differene between patches)
        # We use infinity because that will always be replaced
        for p in range(num_patches):
            best.append((None, float("inf")))

        # if the patch is on the boarder make it black or color it
        if testing_patch == None:
            # if its on the border make it black
            new_image.append((0, 0, 0))
        else:
            # comparing with every training gray scale patch
            for n, training_patch in enumerate(training_gray_patches):

                if training_patch != None:
                    total_difference = 0

                    for i in range(9):
                        difference = abs(testing_patch[i] - training_patch[i]) ** 2
                        total_difference += difference
                    # benton for smart AI
                    #! # try and find the pixel in other 3x3 that gives the least difference but you can only use each compare pixel once
                    #! # for example 1 2 3 and 6 7 8 will compare 1 to 6, 2 to 7, 3 to 8
                    #! # will not work well if there is not similar 3x3
                    #! used = []
                    #! for i in range(9):
                    #!     min_diff=(0,float('inf'))#(index,diff)
                    #!     for j in range(9):
                    #!         if j not in used:
                    #!             difference = abs(testing_patch[j] - training_patch[i]) ** 2
                    #!             if difference<min_diff[1]:
                    #!                 min_diff = (j,difference)

                    #!     used.append(min_diff[0])
                    #!     total_difference += min_diff[1]
                    # see if total difference of patch is less worst in best
                    if total_difference < best[num_patches - 1][1]:
                        best.pop()
                        best.append((n, total_difference))
                        # sort to keep worst of best at the end
                        best.sort(key=lambda x: x[1])

            best_rep = {}
            mode_color = []
            mode_value = 0
            # get count of all the rep colors
            for b in best:
                # b[0] is the index needed to get color from training_as_data which has a rep color
                cluster_color = rep_colors[b[0]]
                if cluster_color not in best_rep:
                    best_rep.update({cluster_color: 1})
                else:
                    curr = best_rep.pop(cluster_color)
                    best_rep.update({cluster_color: curr + 1})

                # keep track of the highest mode color
                if mode_value == best_rep[cluster_color]:
                    mode_color.append((cluster_color, mode_value))
                elif mode_value < best_rep[cluster_color]:
                    mode_color = [cluster_color]
                    mode_value = best_rep[cluster_color]
            # if there is a mode color use it else find the closest relative
            if len(mode_color) == 1:
                new_image.append(mode_color[0])
            else:
                # best [0] is tuple with (index, distance) where since sorted should be the shortest distance
                # best[0][0] is said index so we get the training color and then its rep color
                new_image.append(rep_colors[best[0][0]])

    return new_image


def play():
    try:
        # * original = Image.open("C:Images\EvenSmaller.png")
        original = Image.open("C:Images\EvenSmaller.png")

        img = original.convert("L")  # grayscale copy
        width, height = img.size

        # Create left (training) and right (testing) image halves
        training = original.crop((1, 1, (width / 2) + 1, height + 1))
        testing = img.crop(((width / 2), 1, width, height + 1))  #!plus 1 to get them to equal size
        training_grayscale = img.crop((1, 1, (width / 2) + 1, height + 1))

        # need sizes of all halves
        half_width, half_height = training.size

        # Convert to list
        testing_as_list = list(testing.getdata())
        training_as_list = list(training.getdata())
        training_grayscale_as_list = list(training_grayscale.getdata())

        numDiffPixels = set()
        for x in training_as_list:
            numDiffPixels.add(x)
        num = int(math.sqrt(math.sqrt(len(numDiffPixels))))
        print(num)

        clusters = get_clusters(num, training_as_list)
        tuple_clusters = {}
        clusters_img = []
        cluster_colors = set()
        for x, c in enumerate(clusters):
            RGB = (clusters[c][0], clusters[c][1], clusters[c][2])
            tuple_clusters.update({x: RGB})
            clusters_img.append(RGB)
            cluster_colors.add(RGB)
        clusters_image = Image.new("RGB", training.size)
        clusters_image.putdata(clusters_img)
        clusters_image.show()
        print(cluster_colors)
        return  #!

        output = color_mapping(
            6, training_as_list, training_grayscale_as_list, testing_as_list, half_height, half_width, tuple_clusters
        )

        thing = Image.new("RGB", testing.size)
        thing.putdata(output)

        thing.show()
        training.show()
        testing.show()
        #! end of tandrew test
        return

        # get the 3x3 pixel and show it
        # benton what is in this list data WTF

        # output = Image.new("L", square.shape)
        # output.show()
        # testing = original_as_list[: len(original_as_list)] + img_as_list[len(img_as_list) :]

        # # color is from 0-255: convert to color here
        # testing = testing.reshape(img.size)

        # # Convert back to image
        # img_mul = testing
        # img_ints = np.rint(img_mul)
        # img2 = Image.new("P", testing.shape)
        # img2.putdata(img_ints.astype(int).flatten())

        # original.show()
        # img2.show()

    except IOError:
        print("passed")
        pass


# main()
