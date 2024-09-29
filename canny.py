import cv2 
import numpy as np

def filteredGradient(im, sigma):
    # Computes the smoothed horizontal and vertical gradient images for a given
    # input image and standard deviation. The convolution operation should use
    # the default border handling provided by cv2.
    #
    # im: 2D float32 array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.

    # Returns:
    # Fx: 2D double array with shape (height, width). The horizontal
    #     gradients.
    # Fy: 2D double array with shape (height, width). The vertical
    #     gradients.
        
    # Our default values and makes sure it is odd
    filter_size = int(6 * sigma)
    if filter_size % 2 == 0:
        filter_size += 1
    
    
    def derivative_of_gaussian_kernel_1d(sigma, size):
        # Create a 1D Gaussian kernel
        x = np.linspace(-size // 2, size // 2, num = size)
        gaussian = 1/(np.sqrt(2 * np.pi * sigma))*np.exp(-0.5 * (x / sigma)**2)
        kernel = gaussian * (-x / sigma**2)
        return kernel
    
    
    kernel_x_derivative = derivative_of_gaussian_kernel_1d(sigma, filter_size)
    kernel_y_derivative = derivative_of_gaussian_kernel_1d(sigma, filter_size)
    
    kernel_x = np.expand_dims(kernel_x_derivative, axis=0)  # 1 x N matrix
    kernel_y = np.expand_dims(kernel_y_derivative, axis=1)  # N x 1 matrix
    
    Fx = cv2.filter2D(im,ddepth=-1, kernel = kernel_x)
    Fy = cv2.filter2D(im,ddepth=-1, kernel = kernel_y)

    return Fx, Fy


def edgeStrengthAndOrientation(Fx, Fy):
    # Given horizontal and vertical gradients for an image, computes the edge
    # strength and orientation images.
    #
    # Fx: 2D double array with shape (height, width). The horizontal gradients.
    # Fy: 2D double array with shape (height, width). The vertical gradients.

    # Returns:
    # F: 2D double array with shape (height, width). The edge strength
    #        image.
    # D: 2D double array with shape (height, width). The edge orientation
    #        image.
    
    height, width = Fx.shape
    F = np.zeros((height, width), dtype=np.float64)
    D = np.zeros((height, width), dtype=np.float64)

    # Calculate the edge strength and orientation for each pixel
    for i in range(height):
        for j in range(width):
            x = Fx[i, j]
            y = Fy[i, j]

            # edge strength (magnitude)
            F[i, j] = np.sqrt(x**2 + y**2)

            # edge orientation and making sure it's between 0 and pi
            angle = np.arctan2(y, x)
            if angle < 0:
                angle += np.pi
            D[i, j] = angle

    return F, D


def suppression(F, D):
    # Runs nonmaximum suppression to create a thinned edge image.
    #
    # F: 2D double array with shape (height, width). The edge strength values
    #    for the input image.
    # D: 2D double array with shape (height, width). The edge orientation
    #    values for the input image.

    # Returns:
    # I: 2D double array with shape (height, width). The output thinned
    #        edge image.

    height, width = F.shape
    I = np.zeros((height, width), dtype=np.float64)

    # for i in range(height):
    #     for j in range(width):

    #         angle_deg = D[i, j] * (180 / np.pi)
    #         # angle_deg = angle_deg % 180

    #         # neighbor in direction of D*
    #         r = -1
    #         # neighbor in direction of -D*
    #         p = -1

    #         # finding pixels within D* 
    #         if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg <= 180):
    #             # D* = 0 or pi
    #             if j - 1 >= 0:     
    #                 r = F[i, j - 1]
    #             if j + 1 < width:
    #                 p = F[i, j + 1]
    #         elif 22.5 <= angle_deg < 67.5: 
    #             # D* = pi/4
    #             if i - 1 >= 0 and j + 1 < width:
    #                 r = F[i - 1, j + 1]
    #             if i + 1 < height and j - 1 >= 0:
    #                 p = F[i + 1, j - 1]
    #         elif 67.5 <= angle_deg < 112.5: 
    #             # D* = pi/2
    #             if i - 1 >= 0:
    #                 r = F[i - 1, j]
    #             if i + 1 < height:
    #                 p = F[i + 1, j]
    #         elif 112.5 <= angle_deg < 157.5: 
    #             # D* = 3pi/4
    #             if i - 1 >= 0 and j - 1 >= 0:
    #                 r = F[i - 1, j - 1]
    #             if i + 1 < height and j + 1 < width:
    #                 p = F[i + 1, j + 1]

    #         if r != -1 and F[i, j] < r and p != -1 and F[i, j] < p :
    #             I[i, j] = 0
    #         elif r != -1 and F[i, j] < r:
    #             I[i, j] = 0
    #         elif p != -1 and F[i, j] < p:
    #             I[i , j] = 0
    #         else:
    #             I[i ,j] = F[i ,j]




    for i in range(1, height - 1):
        for j in range(1, width - 1):

            angle_deg = D[i, j] * (180 / np.pi)
            angle_deg = angle_deg % 180

            # neighbor in direction of D*
            r = 0
            # neighbor in direction of -D*
            p = 0

            # finding pixels within D* 
            if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg <= 180):
                # D* = 0 or pi
                r = F[i, j - 1]
                p = F[i, j + 1]
            elif 22.5 <= angle_deg < 67.5: 
                # D* = pi/4
                r = F[i - 1, j + 1]
                p = F[i + 1, j - 1]
            elif 67.5 <= angle_deg < 112.5: 
                # D* = pi/2
                r = F[i - 1, j]
                p = F[i + 1, j]
            elif 112.5 <= angle_deg < 157.5: 
                # D* = 3pi/4
                r = F[i - 1, j - 1]
                p = F[i + 1, j + 1]

            if F[i, j] >= r and F[i, j] >= p:
                I[i, j] = F[i, j]
            else:
                I[i, j] = 0
    return I


def hysteresisThresholding(I, D, tL, tH):
    # Runs hysteresis thresholding on the input image.

    # I: 2D double array with shape (height, width). The input's edge image
    #    after thinning with nonmaximum suppression.
    # D: 2D double array with shape (height, width). The edge orientation
    #    image.
    # tL: double. The low threshold for detection.
    # tH: double. The high threshold for detection.

    # Returns:
    # edgeMap: 2D binary array with shape (height, width). Output edge map,
    #          where edges are 1 and other pixels are 0. 
    
    visited = set()
    height, width = I.shape
    edgeMap = np.zeros((height, width), dtype=np.float64)
    normalized_I = np.zeros((height, width), dtype=np.float64)


    # Normalize I
    max_val = np.max(I)
    normalized_I = I/max_val

    high_threshold_pixels = []

    for i in range(height):
        for j in range(width):
            if (i, j) in visited:
                continue
                
            if normalized_I[i, j] > tH:
                high_threshold_pixels.append((i, j))
                edgeMap[i, j] = 1
                visited.add((i, j))

    # Recursivly track edge
    def trace(i, j):
        neighboring_pixels = [(1,0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]

        for dir_i, dir_j in neighboring_pixels:
            neighbor_i, neighbor_j = i + dir_i, j + dir_j
            # If valid edge
            if height > neighbor_i >= 0 and width > neighbor_j >= 0 and (neighbor_i, neighbor_j) not in visited and tH >= normalized_I[neighbor_i, neighbor_j] > tL:
                edgeMap[neighbor_i, neighbor_j] = 1
                visited.add((neighbor_i, neighbor_j))
                trace(neighbor_i, neighbor_j)

    for i, j in high_threshold_pixels:
        trace(i, j)
            
    return edgeMap

def cannyEdgeDetection(im, sigma, tL, tH):
    # Runs the canny edge detector on the input image. This function should
    # not duplicate your implementations of the edge detector components. It
    # should just call the provided helper functions, which you fill in.
    #
    # IMPORTANT: We have broken up the code this way so that you can get
    # better partial credit if there is a bug in the implementation. Make sure
    # that all of the work the algorithm does is in the proper helper
    # functions, and do not change any of the provided interfaces. You
    # shouldn't need to create any new .py files, unless they are for testing
    # these provided functions.
    # 
    # im: 2D double array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.
    # tL: double. The low threshold for detection.
    # tH: double. The high threshold for detection.

    # Returns:
    # edgeMap: 2D binary image of shape (height, width). Output edge map,
    #          where edges are 1 and other pixels are 0.

    Fx, Fy = filteredGradient(im, sigma)
    F, D = edgeStrengthAndOrientation(Fx, Fy)
    I = suppression(F, D)
    edgeMap = hysteresisThresholding(I, D, tL, tH)

    return edgeMap
