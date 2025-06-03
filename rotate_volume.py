import numpy as np
import cv2
from scipy.ndimage import rotate
from utilities import angle

def rotate_volume(volume):

    angle_YZ, _=angle.angles_estimation(volume)
    rotated_volume_YZ = rotate(volume, angle=angle_YZ, axes=(0, 1), reshape=False) #Plano YZ
