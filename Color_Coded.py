import sys
import numpy as np
import copy
from PIL import Image
# from sympy import symbols, solve, Eq
import random

# return a 3x3 list of the pixels around num
def get3x3(img_as_list, height, width, num):
    if num < width:
        return None
    if num > len(img_as_list) - width:
        return None
    if num % width == 0:
        return None
    if (num - 1) % width == 0:
        return None

    try:
        topleft = img_as_list[num - width - 1]
        top = img_as_list[num - width]
        topright = img_as_list[num - width + 1]
        midleft = img_as_list[num - 1]
        mid = img_as_list[num]
        midright = img_as_list[num + 1]
        bottomleft = img_as_list[num + width - 1]
        bottom = img_as_list[num + width]
        bottomright = img_as_list[num + width + 1]

    except IndexError:
        print("not possible")

    new_list = [topleft, top, topright, midleft, mid, midright, bottomleft, bottom, bottomright]
    return new_list


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
    RSS = 0
    centers_changed = True
    
    while centers_changed:
        # Create dictionary mapping data tuple locations to centers 
        dictionary = dict()
        RSS = 0

        # RGB distances are weighted evenly (using square norm for dist)
        for i in range(len(data)):
            # The index of the closest center to i
            closest = 0
            # Initialize distance between i and closest center
            dist = float('inf')

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

            dictionary.update({i:closest})

        # Now adjust center locations
        # cluster is a list of [count, avg] lists corresponding to the center at its index
        cluster = []
        # Initialize cluster structure
        for i in range(k):
            cluster.append([0,[0,0,0]])
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
                cluster[i][1][rgb] = int(cluster[i][1][rgb] / cluster[i][0])
        
        # Updates centers
        centers_changed = False
        for i in range(k):
            if centers[i] != cluster[i][1]:
                centers[i] = cluster[i][1]
                centers_changed = True
        
        # Early termination condition (close enough to solution according to prev testing)
        if RSS < 940000000:
            centers_changed = False
        #! print(f'centers: {centers}') #!!!!!
        #! print(f'RSS: {RSS}') #!!!!!

    # Replace centers indicies in dictionary with finalized colors
    # rep_colors is the dictionary of (pixel index : representative) colors for the training image
    rep_colors = dict()
    for key in dictionary:
        new_val = centers[dictionary.get(key)]
        rep_colors.update({key:new_val})

    return rep_colors


def main():
    try:
        # Relative Path
        original = Image.open("C:Images\Cartoon-Zoom-Backgrounds-Funny-SpongeBob-Images-to-Download-For-Free-1200x720.png")
        
        img = original.convert("L")  # grayscale copy
        width, height = img.size

        # Create left (training) and right (testing) image halves
        left = original.crop((1, 1, width / 2, height))
        right = img.crop((width / 2, 1, width, height))

        # Convert to list
        img_as_list = list(img.getdata())
        original_as_list = list(original.getdata())

        #! test_list = [(0,0,0), (1,2,4), (4,0,0), (5,0,0), (10,2,5), (3,6,9), (1,3,7)]
        rep_colors = get_clusters(5, original_as_list)
        return

        # get the 3x3 pixel and show it
        # benton what is in this list data WTF
        square = get3x3(img_as_list, height, width, 100000)
        if square == None:
            print("it was none")
        square = np.asarray(square)
        square.shape = (3, 3)
        test = Image.fromarray(square)
        test.show()

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

main()