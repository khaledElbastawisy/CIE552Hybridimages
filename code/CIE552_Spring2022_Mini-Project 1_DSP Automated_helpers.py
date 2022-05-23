# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import scipy.fft


def my_imfilter(image: np.ndarray, filter: np.ndarray, fft = False):
    
    """
    Inputs:
    - image -> numpy nd-array of dim (m, n, c) for RGB images or numpy nd-array of dim (m, n) for gray scale images
    - filter -> numpy nd-array of odd dim (k, l)
    Returns
    - filtered_image -> numpy nd-array of dim (m, n, c) or numpy nd-array of dim (m, n)
    Errors if:
    - filter has any even dimension -> raise an Exception with a suitable error message.
    
    Function supports both normal convolution and fft based convolution
    """
    #filtered_image = np.asarray([0])

    ##################
    # Your code here #
    
    filtered_image = np.empty(image.shape)   
    k,l = filter.shape
   
    if k % 2 == 0 | l % 2 == 0:   #allow only odd dimensions filters
        print ("Error! The dimensions of the filter kernel should be odd.")
        return
    
    if len(image.shape) == 3:    #RGB image
        m, n, c = image.shape

    else:                        #gray scale image
        m,n = image.shape
        c = 1
        image = image.expand_dims(image,axis=2)
        
    #implementation of normal convolution
    if (~fft):
        
        #Zero Padding
        padmat = np.zeros( (m + k - 1, n + l - 1, c) , dtype=image.dtype) #the shape of the output matrix after convolution

        #add the original image into the padded matrix
        kk = (k-1)//2
        ll = (l-1)//2

        padmat[kk : kk + m, ll : ll + n] = image

        #flipping the kernel horizontally and vertically to apply convolution
        filter = np.flipud(filter)
        filter = np.fliplr(filter)
        
        '''
        Now the kernel is flipped and ready to be convolved with the image.
        To apply the convolution we are going to spread the kernel into a 1-D vector and
        do the same with the corresponding portion of the image (the cross image that should
        be correlated with the kernel) and perform a dot product between the two 1-D vectors
        and append the value to the anchor pixel in the filtered image
        '''      
        
        filter = np.reshape(filter, -1) #conver kernel to 1-D vector
        for i in range(n):
            for j in range (m):
                image_kernel = padmat[j:j+k, i:i+l]
                image_kernel = image_kernel.reshape((k*l,c)) #convert the image multiple 1-D vectors depending on no of channels
                filtered_image[j,i] = np.dot(filter,image_kernel)
                
    #implementation of fft based convolution that can be chosen by asserting fft parameter in the function arguments     
    elif (fft):
        kernel_fft = scipy.fft.fft2(filter, (m,n))    

        for i in range(c):
            img = image[:,:,i].reshape(m,n)
            img_fft = scipy.fft.fft2(img)
            prod_fft = np.multiply(img_fft , kernel_fft)
            prod = scipy.fft.ifft2(prod_fft , (m,n))
            if c > 1:
                filtered_image[:,:,i] = np.absolute(prod)
            else:
                filtered_image = np.absolute(prod)
        

    return filtered_image

def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float, scale: float):
    """
    Inputs:
    - image1 -> The image from which to take the low frequencies.
    - image2 -> The image from which to take the high frequencies.
    - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

    Task:
    - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
    - Combine them to create 'hybrid_image'.
    """
    def create_gaussian_filter(ksize = (3,3), sigma = 0.5):
        m = (ksize[0]-1.)/2.
        n = (ksize[1]-1.)/2
        x,y = np.mgrid[-m:m+1, -n:n+1]
        kernel = np.exp(-(x*x+y*y)/(2.*(sigma**2)))
        sumh = kernel.sum()
        if sumh!=0:
            kernel /= sumh
        return kernel
    
    assert image1.shape == image2.shape

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
    # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel
    kernel = create_gaussian_filter (ksize = (25,25), sigma = cutoff_frequency*scale)
    # Your code here:
    low_frequencies = my_imfilter(image1,kernel, fft = True) # Replace with your implementation

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    kernel2= create_gaussian_filter(ksize = (25,25),sigma=cutoff_frequency)
    high_frequencies = image2-my_imfilter(image2,kernel2, fft=True)
    # Replace with your implementation

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = low_frequencies + high_frequencies # Replace with your implementation
    low_frequencies = np.clip(low_frequencies,0,1)
    high_frequencies = np.clip(high_frequencies,0,1)
    hybrid_image = np.clip(hybrid_image,0,1)
    assert hybrid_image.shape == image1.shape
    assert hybrid_image.shape == image2.shape

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0,
    # and all values larger than 1.0 to 1.0.
    # (5) As a good software development practice you may add some checks (assertions) for the shapes
    # and ranges of your results. This can be performed as test for the code during development or even
    # at production!

    return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image: np.ndarray):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect', multichannel=True)
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
