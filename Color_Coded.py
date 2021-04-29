import sys
import numpy as np
import copy
from PIL import Image
from sympy import symbols, solve, Eq
import random

# return a 3x3 list of the pixels around num
# num is


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


try:
    # Relative Path
    original = Image.open("C:Images\Cartoon-Zoom-Backgrounds-Funny-SpongeBob-Images-to-Download-For-Free-1200x720.png")
    img = original.convert("L")  # this make the grey scale
    original_copy = original.convert("P")

    width, height = img.size

    # benton fuck the way they set these parameters
    right_crop = original_copy.crop((1, 1, width / 2, height))
    left_crop = img.crop((width / 2, 1, width, height))

    # Convert to list
    img_data = img.getdata()
    img_as_list = list(img_data)
    original_data = original_copy.getdata()
    original_as_list = list(original_data)

    # get the 3x3 pixel and show it
    # benton what is in this list data WTF
    square = get9x9(img_as_list, height, width, 100000)
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
