# In[]
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Guide:
    def __init__(self):
        self.black_mask = np.array([[[0, 256],
                                [90, 256],
                                [100, 170],
                                [156, 170],
                                [166, 256],
                                [256, 256],
                                [256, 0],
                                [0, 0]]])

        self.black_mask_left = np.array([[[0, 256],
                                     [42, 256],
                                     [52, 170],
                                     [108, 170],
                                     [118, 256],
                                     [256, 256],
                                     [256, 0],
                                     [0, 0]]])

        self.black_mask_right = np.array([[[0, 256],
                                      [138, 256],
                                      [148, 170],
                                      [204, 170],
                                      [214, 256],
                                      [256, 256],
                                      [256, 0],
                                      [0, 0]]])

    def guide_safe_path(self, image):
        """
        This method will guide the agent to choose a safer path
        It will tell if the agent should go straight, right or left
        :param image: frame/image with footpath prediction drawn w/ GREEN color
                      COLOR GREEN IS VERY IMPORTANT
        :return: 0: Go straight
                 1: Go left
                 2: Go right
        """
        image_copy_left = image.copy()
        image_copy_right = image.copy()

        # Filling the image with black color rather the region of interest
        cv2.fillPoly(image, self.black_mask, (0, 0, 0))
        cv2.fillPoly(image_copy_left, self.black_mask_left, (0, 0, 0))
        cv2.fillPoly(image_copy_right, self.black_mask_right, (0, 0, 0))

        # Calculating free space by splitting layers (BGR)
        # green area is footpath
        # the greater, the better
        _, g, _ = cv2.split(image)
        _, g_left, _ = cv2.split(image_copy_left)
        _, g_right, _ = cv2.split(image_copy_right)

        free_area_straight = (g==255).sum()
        free_area_left = (g_left == 255).sum()
        free_area_right = (g_right == 255).sum()

        free_space = np.array([free_area_straight, free_area_left, free_area_right])

        return np.argmax(free_space)


