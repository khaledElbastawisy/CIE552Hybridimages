# Project Image Filtering and Hybrid Images - Generate Hybrid Image
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from helpers import vis_hybrid_image, load_image, save_image, my_imfilter, gen_hybrid_image

# Before trying to construct hybrid images, it is suggested that you
# implement my_imfilter in helpers.py and then debug it using proj1_part1.py

# Debugging tip: You can split your python code and print in between
# to check if the current states of variables are expected or use a proper debugger.

## Setup
# Read images and convert to floating point format
image1 = load_image('../data/dog.bmp')
image2 = load_image('../data/cat.bmp')
image3 = load_image('../data/submarine.bmp')
image4 = load_image('../data/fish.bmp')
image5 = load_image('../data/bird.bmp')
image6 = load_image('../data/plane.bmp')
image7 = load_image('../data/motorcycle.bmp')
image8 = load_image('../data/bicycle.bmp')
image9 = load_image('../data/marilyn.bmp')
image10 = load_image('../data/einstein.bmp')

# display the dog and cat images
plt.figure(figsize=(3,3))
plt.imshow((image1*255).astype(np.uint8))
plt.figure(figsize=(3,3))
plt.imshow((image2*255).astype(np.uint8))

# For your write up, there are several additional test cases in 'data'.
# Feel free to make your own, too (you'll need to align the images in a
# photo editor such as Photoshop).
# The hybrid images will differ depending on which image you
# assign as image1 (which will provide the low frequencies) and which image
# you asign as image2 (which will provide the high frequencies)

## Hybrid Image Construction ##
# cutoff_frequency is the standard deviation, in pixels, of the Gaussian#
# blur that will remove high frequencies. You may tune this per image pair
# to achieve better results.

#a scale parameter is used increase the gap betwewen the cutoff freq of the high and low frequency images

#cat_dog hybrid
cutoff_frequency1 = 4
scale1 = 2
low_frequencies1, high_frequencies1, hybrid_image1 = gen_hybrid_image(image1, image2, cutoff_frequency1,scale1)

#fish_submarine
cutoff_frequency2 = 2
scale2 = 3
low_frequencies2, high_frequencies2, hybrid_image2 = gen_hybrid_image(image3, image4, cutoff_frequency2,scale2)

#plane_bird
cutoff_frequency3 = 1.7
scale3 = 3
low_frequencies3, high_frequencies3, hybrid_image3 = gen_hybrid_image(image5, image6, cutoff_frequency3,scale3)

#motorcycle_bicycle
cutoff_frequency4 = 1
scale4 = 10
low_frequencies4, high_frequencies4, hybrid_image4 = gen_hybrid_image(image7, image8, cutoff_frequency4,scale4)

#marilyn_einstein
cutoff_frequency5 = 2
scale5 = 2
low_frequencies5, high_frequencies5, hybrid_image5 = gen_hybrid_image(image9, image10, cutoff_frequency5,scale5)




## Visualize and save outputs ##
plt.figure()
plt.imshow((low_frequencies1*255).astype(np.uint8))
plt.figure()
plt.imshow(((high_frequencies1+0.5)*255).astype(np.uint8))
vis1 = vis_hybrid_image(hybrid_image1)
vis2 = vis_hybrid_image(hybrid_image2)
vis3 = vis_hybrid_image(hybrid_image3)
vis4 = vis_hybrid_image(hybrid_image4)
vis5 = vis_hybrid_image(hybrid_image5)

plt.figure(figsize=(20, 20))
plt.imshow(vis1)

save_image('../results/low_frequencies1.jpg', low_frequencies1)
save_image('../results/high_frequencies1.jpg', np.clip(high_frequencies1+0.5,0,1))
save_image('../results/hybrid_image1.jpg', hybrid_image1)
save_image('../results/hybrid_image_scales1.jpg', vis1)

save_image('../results/low_frequencies2.jpg', low_frequencies2)
save_image('../results/high_frequencies2.jpg', np.clip(high_frequencies2+0.5,0,1))
save_image('../results/hybrid_image2.jpg', hybrid_image2)
save_image('../results/hybrid_image_scales2.jpg', vis2)

save_image('../results/low_frequencies3.jpg', low_frequencies3)
save_image('../results/high_frequencies3.jpg', np.clip(high_frequencies3+0.5,0,1))
save_image('../results/hybrid_image3.jpg', hybrid_image3)
save_image('../results/hybrid_image_scales3.jpg', vis3)

save_image('../results/low_frequencies4.jpg', low_frequencies4)
save_image('../results/high_frequencies4.jpg', np.clip(high_frequencies4+0.5,0,1))
save_image('../results/hybrid_image4.jpg', hybrid_image4)
save_image('../results/hybrid_image_scales4.jpg', vis4)

save_image('../results/low_frequencies5.jpg', low_frequencies5)
save_image('../results/high_frequencies5.jpg', np.clip(high_frequencies5+0.5,0,1))
save_image('../results/hybrid_image5.jpg', hybrid_image5)
save_image('../results/hybrid_image_scales5.jpg', vis5)
