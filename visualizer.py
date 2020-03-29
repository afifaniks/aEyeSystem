import cv2
import numpy as np
from dist_calculator import DistanceCalculator


class Visualizer:
    def __init__(self):
        self.dist_calc = DistanceCalculator("weights/distance_model.pkl")
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.TEXT_WEIGHT = 1
        self.BORDER_WIDTH = 2
        self.COLOR_RED = (255, 0, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_GREEN = (0, 255, 0)

    def draw_contour(self, original_image, segmentation_mask):
        """
        This method draws biggest contour of the segmentation mask
        :param original_image: Image where the contour should be drawn
        :param segmentation_mask: Inference result from segmentation model
        :return: Original image with segmentation mask and contour drawn
        """

        contours, _ = cv2.findContours(segmentation_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        original_image[segmentation_mask.astype(np.uint8) > 0] = (0.3, 0.5, 0.4)
        biggest_area = 0

        # Finding the biggest contour
        for contour in contours:
            area = cv2.contourArea(contour)

            if area > biggest_area:
                biggest_area = area
                biggest_contour = contour

        original_image = cv2.drawContours(original_image, biggest_contour, -1, self.COLOR_GREEN, self.BORDER_WIDTH)

        return original_image

    def draw_detection_boxes(self, image, detection_boxes):
        for det in detection_boxes:
            x, y, w, h, label, confidence = det
            cv2.rectangle(image, (x, y), (x + w, y + h), self.COLOR_WHITE, self.BORDER_WIDTH)
            print("Y", y + h)
            dist = self.dist_calc.predict(y + h)[0][0]
            print(label, dist)

            cv2.putText(image, label + " " + str(round(dist, 2)), (x, y + h + 10), self.FONT, 0.35, self.COLOR_WHITE, self.TEXT_WEIGHT)

        return image