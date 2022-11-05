
#Import library
import torch
import torchvision
import kornia as ki
import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(input: torch.Tensor):
    out = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = ki.utils.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()


#load input and read image color 
#Cv2 represent the output in numpy array
img_input= cv2.imread('../input/edge-detection-dlgeneral-approach/edgedetection(general and dl)/input.png',cv2.IMREAD_COLOR)

# Kornia generally torch based one! so we are convert numpy to torch

img_input = ki.utils.image_to_tensor(img_input)  # CxHxWx
img_input = img_input[None,...].float() / 255.
imshow(img_input)
#Covert first bgr to rgb
#And rgb to gray
img_rgb = ki.color.bgr_to_rgb(img_input)
img_gray = ki.color.rgb_to_grayscale(img_rgb)
imshow(img_gray)
#canny edges
img_canny= ki.filters.canny(img_gray)[0]
imshow(1. - img_canny.clamp(0., 1.))
