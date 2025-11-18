# pylint: disable=no-member
# import the necessary packages
import numpy as np
import cv2

class FeatureExtractor:

    #######################################################################################################################
    # Function extract(self, image)
    # Function to extract features for an image
    #
    # Input arguments:
    #   - [image] image: input image
    # Output argument:
    #   - [list] features: list with extracted features
    #######################################################################################################################
    def extract(self, image):
        spatial = self.extract_features_spatial(image, factor=50)
        thumb = self.thumbnail_features(image)
        part = self.partitionbased_histograms(image)
        features = np.concatenate([spatial, thumb, part]).astype(np.float32)

        # Normalize the feature vector to unit length (L2 normalization).
        # After this step, the vector has length 1, which allows cosine similarity
        # to be computed simply as a dot product: cos_sim(x, y) = x · y
        # This removes the need to compute norms during search time.
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm

        return features

    #######################################################################################################################
    # Function histogram(self, image)
    # Function to extract histogram features for an image
    #
    # Tipp:
    # 	- use 'cv2.calcHist'
    #	- create a list out of the histogram (hist.tolist())
    #	- return a flatten list
    #
    # Input arguments:
    #   - [image] image: input image
    # Output argument:
    #   - [list] features: list with extracted features
    #######################################################################################################################
    def histogram(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist.ravel()

    #######################################################################################################################
    # Function partitionbased_histograms(self, image, factor = 4):
    # Function to create partition based histograms
    #
    # Help:
    #   - Resize image (dim(100, 100))
    #	- Observe (factor * factor) image parts
    #	- calculate a histogramm for each part and add to feature list
    #
    # Input arguments:
    #   - [image] image: input image
    #	- [int] factor: facto to split images into parts
    # Output argument:
    #   - [list] feature_list: list with extracted features
    #######################################################################################################################
    def partitionbased_histograms(self, image, factor=2, bins=32):
        dim = (64, 64)  # kleiner als vorher, reicht
        resized = cv2.resize(image, dim)
        h, w = resized.shape
        h_step = h // factor
        w_step = w // factor

        feature_list = []
        for i in range(factor):
            for j in range(factor):
                patch = resized[i * h_step:(i + 1) * h_step,
                        j * w_step:(j + 1) * w_step]

                hist = cv2.calcHist([patch], [0], None, [bins], [0, 256])
                hist = cv2.normalize(hist, hist).flatten().tolist()
                feature_list.extend(hist)

        return feature_list

    #######################################################################################################################
    # Function thumbnail_features(self, image)
    # Function to create a thumbnail of an image and return the image values (features)
    #
    # Help:
    #   - Resize image (dim(30,30))
    #	- Create a list from the image (np array)
    #	- flatten and return the list
    #
    # Input arguments:
    #   - [image] image: input image
    # Output argument:
    #   - [list] feature_list: list with extracted features
    #######################################################################################################################
    def thumbnail_features(self, image):
        resized_image = cv2.resize(image, (8, 8))
        return resized_image.ravel()

    #######################################################################################################################
    # Function extract_features_spatial(self, image, factor = 10)
    # Function to create spatial features
    #
    # Help:
    #   - Resize image (dim(200,200))
    #	- Observe (factor * factor) image parts
    #	- calculate max, min and mean for each part and add to feature list
    #
    # Input arguments:
    #   - [image] image: input image
    #	- [int] factor: facto to split images into parts
    # Output argument:
    #   - [list] feature_list: list with extracted features
    #######################################################################################################################
    def extract_features_spatial(self, image, factor=50):
        features = []
        resized_image = cv2.resize(image, (200, 200))
        area_cnt = 200 // factor  # 4 bei factor=50

        for i in range(area_cnt):
            for j in range(area_cnt):
                img_area = resized_image[
                           j * factor:(j + 1) * factor,
                           i * factor:(i + 1) * factor
                           ]
                img_area = img_area.astype(np.float32)
                features.append(np.max(img_area))
                features.append(np.min(img_area))
                features.append(np.mean(img_area))

        return np.array(features, dtype=np.float32)

    #######################################################################################################################
    # Function image_pixels(self, image):
    # Function to return the image pixels as features
    #
    # Example of a **bad** implementation. The use of pixels as features is highly inefficient!
    #
    # Input arguments:
    #   - [image] image: input image
    # Output argument:
    #   - [list] feature_list: list with extracted features
    #######################################################################################################################
    def image_pixels(self, image):
        # cast image to list of lists
        features = image.tolist()
        # flatten the list of lists
        features = [item for sublist in features for item in sublist]
        # return
        return features


if __name__ == '__main__':
    # Read the test image
    example_image = cv2.imread("./ImageCLEFmed2007_test/3145.png", cv2.IMREAD_GRAYSCALE)

    # Assert image read was successful
    assert example_image is not None

    # create extractor
    feature_extractor = FeatureExtractor()

    # describe the image
    features = feature_extractor.extract(example_image)

    # print the features
    print("Features: ", features)
    print("Length: ", len(features))
