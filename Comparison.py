import numpy as np
import copy
from PIL import Image
import math
import random
import pathlib


# Input img arrays must be same shape
def loss_calc(img1, img2):
    loss = 0
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            for k in range(len(img1[0][0])):
                loss += (img1[i][j][k] - img2[i][j][k]) ** 2
    return loss


def main():
    with Image.open("C:Nature\\right_basic.png") as basic, Image.open("C:Nature\\right_original.png") as original:
        basic_rgb = basic.convert(mode="RGB")
        original_rgb = original.convert(mode="RGB")

        basic_data = np.asarray(basic_rgb) / 255
        original_data = np.asarray(original_rgb) / 255
        
        min_loss = float('inf')

        x_variants = 6   # 418 to 423 endings
        y_variants = 5   # 211 to 215 endings
        for i in range(x_variants):
            for j in range(y_variants):
                loss = loss_calc(basic_data, original_data[0 + i : 418 + i, 0 + j : 211 + j, :])
                print(f'Basic [{i}][{j}] loss: {loss}')
                if loss < min_loss:
                    min_loss = loss

        print(f'Basic min_loss: {min_loss}')


    with Image.open("C:Nature\\right_adv.png") as adv, Image.open("C:Nature\\right_original.png") as original:
        adv_rgb = adv.convert(mode="RGB")
        original_rgb = original.convert(mode="RGB")

        adv_data = np.asarray(adv_rgb) /255
        original_data = np.asarray(original_rgb) /255
        
        min_loss = float('inf')

        x_variants = 6   # 418 to 423 endings
        y_variants = 5   # 211 to 215 endings
        for i in range(x_variants):
            for j in range(y_variants):
                loss = loss_calc(adv_data, original_data[0 + i : 418 + i, 0 + j : 211 + j, :])
                print(f'Adv [{i}][{j}] loss: {loss}')
                if loss < min_loss:
                    min_loss = loss

        print(f'Adv min_loss: {min_loss}')


# Basic min_loss: 1391.7868512108064
# Adv min_loss: 11106.260638213464
main()