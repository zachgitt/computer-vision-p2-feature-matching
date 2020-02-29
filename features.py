import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
from scipy.spatial import distance
import math
import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def get_mirrored_image(self, srcImage, n=2):
        sq1 = np.flip(srcImage[:n, :n])
        sq2 = np.flip(srcImage[:n, -n:])
        sq3= np.flip(srcImage[-n:, :n])
        sq4 = np.flip(srcImage[-n:, -n:])

        first_two_cols = np.flip(srcImage[:, :n], axis=1)
        last_two_cols = np.flip(srcImage[:,-n:], axis=1)
        first_two_cols = np.vstack((np.vstack((sq1, first_two_cols)), sq3))
        last_two_cols = np.vstack((np.vstack((sq2, last_two_cols)), sq4))

        first_two_rows = np.flip(srcImage[:n, :], axis=0)
        last_two_rows = np.flip(srcImage[-n:, :], axis=0)

        srcImage = np.vstack((first_two_rows, srcImage))
        srcImage = np.vstack((srcImage, last_two_rows))

        srcImage = np.hstack((first_two_cols, srcImage))
        srcImage = np.hstack((srcImage, last_two_cols))

        return srcImage

    def get_random_image(self, shape):
        return np.random.randint(0, 255, (shape))

    def gaussian_blur_kernel_2d(self, sigma=0.5, height=5, width=5):
        G = np.zeros((height, width))
        const = 2 * (sigma ** 2)
        num = 1 / (math.pi * const)
        array1 = np.concatenate(
            ([abs(height // 2 - i) for i in range(height // 2 + 1)], [i for i in range(1, height // 2 + 1)]))
        array2 = np.concatenate(
            ([abs(width // 2 - i) for i in range(width // 2 + 1)], [i for i in range(1, width // 2 + 1)]))
        for x in range(len(array1)):
            for y in range(len(array2)):
                G[x][y] = math.e ** ((-1) * (array1[x] ** 2 + array2[y] ** 2) / float(const))
        return G / np.sum(G)

    def flip_kernel(self, kernel):
        kernel = np.flip(kernel, 0)
        return np.flip(kernel, 1)


    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]
        harrisImage = np.zeros((height, width))
        orientationImage = np.zeros((height, width))

        dx = ndimage.sobel(srcImage, 0)  # horizontal derivative
        dy = ndimage.sobel(srcImage, 1)  # vertical derivative

        ix_sq = dx**2
        iy_sq = dy**2
        ix_iy = np.multiply(dx, dy)

        s = 0.5
        g_ix_sq = ndimage.gaussian_filter(ix_sq, sigma=s)
        g_iy_sq = ndimage.gaussian_filter(iy_sq, sigma=s)
        g_ix_iy = ndimage.gaussian_filter(ix_iy, sigma=s)

        for h in range(height):
            for w in range(width):
                H = np.array([[g_ix_sq[h][w], g_ix_iy[h][w]], [g_ix_iy[h][w], g_iy_sq[h][w]]])
                c = np.linalg.det(H) - (0.1 * (np.trace(H)**2))
                harrisImage[h][w] = c
                if dx[h][w] == 0 and dy[h][w] == 0:
                    orientationImage[h][w] = 0
                else:
                    orientationImage[h][w] = math.degrees(math.atan2(dx[h][w], dy[h][w]))
        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        return harrisImage == ndimage.filters.maximum_filter(harrisImage, size=(7,7), mode="constant")


    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                f.size = 10
                f.pt = (x,y)
                f.angle = orientationImage[y][x]
                f.response = harrisImage[y][x]
                features.append(f)
        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        copy = np.zeros((grayImage.shape[0]+4, grayImage.shape[1]+4))
        copy[2:-2, 2:-2] = grayImage
        for i, f in enumerate(keypoints):
            x, y = int(f.pt[0]), int(f.pt[1])
            x += 2
            y += 2
            desc[i] = copy[y-2:y+3, x-2:x+3].flatten()
            # The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            x = -1* f.pt[0]
            y = -1 * f.pt[1]
            theta = math.radians(f.angle)

            # height, width = grayImage.shape[0]/2, grayImage.shape[1]/2
            R = transformations.get_rot_mx(0, 0, -theta)
            T1 = transformations.get_trans_mx(np.array([x, y, 0]))
            S = transformations.get_scale_mx(1/5, 1/5, 0)
            T2 = transformations.get_trans_mx(np.array([4, 4, 0]))
            transMx = (T2 @ (S @ (R @ T1)))[:2,[0,1,3]]

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # Normalize the descriptor to have zero mean and unit
            # variance. If the variance is negligibly small (which we 
            # define as less than 1e-10) then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            destImage = destImage.flatten()
            mean = np.mean(destImage)
            std = np.std(destImage)
            if std < 1e-5:
                destImage = np.zeros(windowSize**2)
            else:
                destImage = (destImage - mean)/std
            desc[i] = destImage
        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        Y = distance.cdist(desc1, desc2, 'sqeuclidean')
        min_dist_indices = np.argmin(Y, axis=1)
        for i, min_idx in enumerate(min_dist_indices):
            matches.append(cv2.DMatch(i, min_idx, Y[i][min_idx]))
        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        Y = distance.cdist(desc1, desc2, 'sqeuclidean')
        min_dist_indices = np.argpartition(Y, 2)
        for i, min_idx in enumerate(min_dist_indices):
            min_idx_first = min_idx[0]
            min_idx_second = min_idx[1]
            matches.append(cv2.DMatch(i, min_idx_first, Y[i][min_idx_first]/Y[i][min_idx_second]))
        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))
