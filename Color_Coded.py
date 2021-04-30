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


# Returns list of cluster centers using Lloyd's k-means clustering algorithm
# INPUTS: k is the number of clusters, data is a list of data RGB tuples
def get_clusters(k, data):
    # Pick random centers from the data list to start 
    centers = random_picks(k, data)

    # Create dictionary mapping data tuple locations to centers 
    dictionary = dict()

    # RGB distances are weighted evenly (using square norm for dist)
    for i in data:
        # The index of the closest center to i
        closest = 0
        # Get distance between i and first center
        dist = float('inf') #!CHECK THIS

        # Iterate through centers
        for c in range(0, k):
            temp = 0
            # Iterate through each R/G/B value of i
            for val in range(3):
                temp += (centers[c][val] - i[val]) ** 2
            
            # If distance to center c is shorter
            if temp < dist:
                dist = temp
                closest = c

        dictionary.update({i:closest})
    
    # Now adjust center locations
    #! For each center c, add up all i's that fall under c
    #! Then average that summation and make that the new c
    #! NEED TO ADD LOOP TO BE ABLE TO ITERATIVELY CONVERGE TO SOLUTION

    return centers


def main():
    try:
        #* Relative Path for Tandrew
        # original = Image.open("C:Images\Cartoon-Zoom-Backgrounds-Funny-SpongeBob-Images-to-Download-For-Free-1200x720.png")
        #* Relative Path for Benton
        original = Image.open("rotated_picture.jpg") # benton @Tandrew this line should work for u too, it's in the git repo
        
        img = original.convert("L")  # grayscale copy
        width, height = img.size

        # Create left (training) and right (testing) image halves
        left = original.crop((1, 1, width / 2, height))
        right = img.crop((width / 2, 1, width, height))

        # Convert to list
        img_as_list = list(img.getdata())
        original_as_list = list(original.getdata())

        get_clusters(5, original_as_list)
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