from wand.image import Image
from wand.color import Color
import numpy as np
import cv2

def crop(image_path, destination_path):
    stitched = cv2.imread(image_path)

    stitched2 = stitched.copy()
    stitched2 = Image.from_array(stitched2)
    stitched2.trim(color=Color('rgb(0,0,0)'), percent_background=0.0, fuzz=0)
    trimmed_image = np.array(stitched2)
    cv2.imwrite(destination_path, trimmed_image)
