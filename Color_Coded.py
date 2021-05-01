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


# Returns a list of k randomly picked elements from a list l
# For rgb
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


# find the most similar patches to represent the gray scale
# training and testing are list of data
# height and width are size of image
def color_mapping(num_patches, training, training_gray, testing, height, width, rep_colors):

    training_gray_patches = []

    # get a list of all the patches in the gray training data
    for x in range(len(training_gray)):
        patch = get3x3(training_gray, height, width, x)
        training_gray_patches.append(patch)

    new_image = []
    # compare the testing patch to training patch and keep best numpatches representatives
    for x in range(len(testing)):
        print(x / len(testing))
        testing_patch = get3x3(testing, height, width, x)
        best = []

        # set up 'best' representative patches
        # tuple is (index of best rep, total differene between patches)
        # We use infinity because that will always be replaced
        for p in range(num_patches):
            best.append((None, float("inf")))

        # if the patch is on the boarder
        if testing_patch == None:
            # if its on the border make it black
            new_image.append((0, 0, 0))
        else:
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
                    #! see if total difference of patch is less worst in best
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
                cluster_color = rep_colors[training[b[0]]]
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
                new_image.append(rep_colors[training[best[0][0]]])

    return new_image


# Returns list of cluster centers using Lloyd's k-means clustering algorithm
# INPUTS: k is the number of clusters, data is a list of data RGB tuples
def get_clusters(k, data):
    # Pick random centers from the data list to start
    centers = random_picks(k, data)

    # Create dictionary mapping data tuple locations to centers
    dictionary = dict()

    # RGB distances are weighted evenly (using square norm for dist)
    for pixel in data:
        # The index of the closest center to i
        closest = 0
        # Get distance between i and first center
        dist = float("inf")

        # Iterate through centers
        for rep_color in range(0, k):
            temp = 0
            # Iterate through each R/G/B value of i
            for val in range(3):
                temp += (centers[rep_color][val] - pixel[val]) ** 2
            temp = math.sqrt(temp)
            # If distance to center c is shorter
            if temp < dist:
                dist = temp
                closest = rep_color

        dictionary.update({pixel: closest})

    # Now adjust center locations
    #! For each center c, add up all i's that fall under c
    #! Then average that summation and make that the new c
    #! NEED TO ADD LOOP TO BE ABLE TO ITERATIVELY CONVERGE TO SOLUTION

    return centers


def main():
    try:
        # * Relative Path for Tandrew
        # original = Image.open("C:Images\Cartoon-Zoom-Backgrounds-Funny-SpongeBob-Images-to-Download-For-Free-1200x720.png")
        # * Relative Path for Benton
        original = Image.open(
            "C:Images\Cartoon-Zoom-Backgrounds-Funny-SpongeBob-Images-to-Download-For-Free-1200x720.png"
        )

        img = original.convert("L")  # grayscale copy
        width, height = img.size

        # Create left (training) and right (testing) image halves
        # training = original.crop((1, 1, (width / 2) + 1, height + 1))
        # testing = img.crop(((width / 2), 1, width, height + 1))  #!plus 1 to get them to equal size
        # training_grayscale = img.crop((1, 1, (width / 2) + 1, height + 1))

        training = original.crop((0, 0, 100, 100))  #!delete
        testing = img.crop((width / 2, 0, (width / 2) + 100, 100))  #!delete
        training_grayscale = img.crop((0, 0, 100, 100))  #!delete

        # need sizes of all halves
        half_width, half_height = training.size

        # Convert to list
        testing_as_list = list(testing.getdata())
        training_as_list = list(training.getdata())
        training_grayscale_as_list = list(training_grayscale.getdata())

        #! temp for tandrew testing
        # get_clusters(5, training_as_list)

        # andrews test for his shit
        mapping = {}
        for x, pixel in enumerate(training_as_list):
            if x % 5 == 0:
                mapping.update({pixel: (255, 0, 0)})
            elif x % 4 == 0:
                mapping.update({pixel: (255, 255, 0)})
            elif x % 3 == 0:
                mapping.update({pixel: (0, 255, 0)})
            elif x % 2 == 0:
                mapping.update({pixel: (0, 255, 255)})
            else:
                mapping.update({pixel: (0, 0, 255)})

        output = color_mapping(
            6, training_as_list, training_grayscale_as_list, testing_as_list, half_height, half_width, mapping
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


main()
