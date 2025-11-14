<<<<<<< HEAD
# import the necessary packages
import csv
import math
from pathlib import Path
import numpy as np

#######################################################################################################################
# Function square_rooted(x)
# Function to calculate the root of the sum of all swared elements of a list
#
# Help: 
#   - for each element calculate the square value. Sum all values up. Calculate the root
# 
# Input arguments:
#   - [list] x: list of input numbers
# Output argument:
#   - [float] result
#######################################################################################################################
def square_rooted(x):
        return math.sqrt(sum([a*a for a in x]))

class Searcher:

    #######################################################################################################################
	# Function __init__(self, index_path):
	# Init function. Just set the index path
	#######################################################################################################################
    def __init__(self, index_path):
        # store our index path
        self.index_path = index_path
        
    #######################################################################################################################
	# Function search(self, queryFeatures, limit = 5)
	# Function retrieve similar images based on the queryFeatures
	#
    # 	# Tasks:
    #   - If there is no index file -> Print error and return False [Hint: Path(*String*).exists()]
    #   - Open the index file
    #   - Read in CSV file [Hint: csv.reader()]
    #   - Iterate over every row of the CSV file
    #       - Collect the features and cast to float
    #       - Calculate distance between query_features and current features list
    #       - Save the result in a dictionary: key = image_path, Item = distance
    #   - Close file
    #   - Sort the results according their distance
    #   - Return limited results
	# 
	# Input arguments:
	#   - [list] query_features: Lost of query features
    #   - [int] limit: Limit the retrieved results
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def search(self, query_features, limit = 5):
        # TODO 
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("Index does not exist")
        distances = []
        with open(self.index_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if not row:
                    continue
                image_path[0]
                features = [float(x) for x in row[1:]]
                distances.append({"name":image_path, "distance":self.euclidean_distance(features, query_features)})
        distances.sort(key=lambda x: x["distance"])
        return distances[:limit]


    
    #######################################################################################################################
	# Function euclidean_distance(self, x, y):
	# Function to calculate the euclidean distance for two lists
	#
    # 	# Help: https://pythonprogramming.net/euclidean-distance-machine-learning-tutorial/
	# 
	# Input arguments:
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def euclidean_distance(self, x, y):
        # TODO 
        distance = 0
        if len(x) != len(y):
            raise ValueError("Length mismatch")
        for i in range(len(x)):
                distance = distance + (x[i] - y[i]) ** 2
        return math.sqrt(distance)



    #######################################################################################################################
	# Function manhattan_distance(self, x, y):
	# Function to calculate the manhattan distance for two lists
	# 
	# Input arguments:
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def manhattan_distance(self, x, y):
        # TODO 
        distance = 0
        if len(x) != len(y):
            raise ValueError("Length mismatch")
        for i in range(len(x)):
            distance = distance + abs(x[i] - y[i])
        return distance
        

    #######################################################################################################################
	# Function minkowski_distance(self, p, x, y):
	# Function to calculate the minkowski distance for two lists
	# 
    # 	# Help: We expect w to be 1
    # 
	# Input arguments:
    #   - [int] p: P-value from slide
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def minkowski_distance(self, p, x, y):
        # TODO 
        distance =0
        distance = (p(sum(pow(abs(a-b), p)
                    for a, b in zip(x,y)), p))
        return distance


    #######################################################################################################################
	# Function cosine_similarity(self, x, y):
	# Function to calculate the cosine similarity for two lists
	#
    # 	# Help:
    #       Compute numerator
    #       Compute denominator with the help of "square_rooted"
    #       Calculate similarity
    #       Change range to [0,1] rather than [-1,1]
	# 
	# Input arguments:
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed similarity
	#######################################################################################################################
    def cosine_similarity(self, x, y):
        # TODO 
        pass

    #######################################################################################################################
	# Function cosine_distance(self, x, y):
	# Function to calculate the cosine distance with help of cosine similarity
	#######################################################################################################################
    def cosine_distance(self, x, y):
        # TODO 
        pass
=======
# import the necessary packages
import csv
import math
import os
import cv2
import numpy as np
from pathlib import Path
from feature_extractor import FeatureExtractor

#######################################################################################################################
# Function square_rooted(x)
# Function to calculate the root of the sum of all swared elements of a list
#
# Help: 
#   - for each element calculate the square value. Sum all values up. Calculate the root
# 
# Input arguments:
#   - [list] x: list of input numbers
# Output argument:
#   - [float] result
#######################################################################################################################
def square_rooted(x):
        return math.sqrt(sum([a*a for a in x]))

class Searcher:

    #######################################################################################################################
	# Function __init__(self, index_path):
	# Init function. Just set the index path
	#######################################################################################################################
    def __init__(self, index_path):
        # store our index path
        self.index_path = index_path

    #######################################################################################################################
	# Function search(self, queryFeatures, limit = 5)
	# Function retrieve similar images based on the queryFeatures
	#
    # 	# Tasks:
    #   - If there is no index file -> Print error and return False [Hint: Path(*String*).exists()]
    #   - Open the index file
    #   - Read in CSV file [Hint: csv.reader()]
    #   - Iterate over every row of the CSV file
    #       - Collect the features and cast to float
    #       - Calculate distance between query_features and current features list
    #       - Save the result in a dictionary: key = image_path, Item = distance
    #   - Close file
    #   - Sort the results according their distance
    #   - Return limited results
	# 
	# Input arguments:
	#   - [list] query_features: Lost of query features
    #   - [int] limit: Limit the retrieved results
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def search(self, query_features, limit = 5):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("Index does not exist")
        distances = []
        with open(self.index_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if not row:
                    continue  # skip empty lines
                image_path = row[0]
                features = [float(x) for x in row[1:]]
                distances.append({"name":image_path, "distance":self.cosine_distance(features, query_features)})
        distances.sort(key=lambda x: x["distance"])
        return distances[:limit]

    #######################################################################################################################
	# Function euclidean_distance(self, x, y):
	# Function to calculate the euclidean distance for two lists
	#
    # 	# Help: https://pythonprogramming.net/euclidean-distance-machine-learning-tutorial/
	# 
	# Input arguments:
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def euclidean_distance(self, x, y):
        distance = 0
        if len(x) != len(y):
            raise ValueError("Length mismatch")
        for i in range(len(x)):
                distance = distance + (x[i] - y[i]) ** 2
        return math.sqrt(distance)


    #######################################################################################################################
	# Function manhattan_distance(self, x, y):
	# Function to calculate the manhattan distance for two lists
	# 
	# Input arguments:
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def manhattan_distance(self, x, y):
        distance = 0
        if len(x) != len(y):
            raise ValueError("Length mismatch")
        for i in range(len(x)):
            distance = distance + abs(x[i] - y[i])
        return distance

    #######################################################################################################################
	# Function minkowski_distance(self, p, x, y):
	# Function to calculate the minkowski distance for two lists
	# 
    # 	# Help: We expect w to be 1
    # 
	# Input arguments:
    #   - [int] p: P-value from slide
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed distance
	#######################################################################################################################
    def minkowski_distance(self, p, x, y):
        distance = 0
        if len(x) != len(y):
            raise ValueError("Length mismatch")
        for i in range(len(x)):
            distance = distance + abs(x[i] - y[i])**p
        return distance**(1/p)


    #######################################################################################################################
	# Function cosine_similarity(self, x, y):
	# Function to calculate the cosine similarity for two lists
	#
    # 	# Help:
    #       Compute numerator
    #       Compute denominator with the help of "square_rooted"
    #       Calculate similarity
    #       Change range to [0,1] rather than [-1,1]
	# 
	# Input arguments:
	#   - [list] x: List one
    #   - [list] y: List two
	# Output argument:
	#   - [float] result: Computed similarity
	#######################################################################################################################
    def cosine_similarity(self, x, y):
        # sum(Ai*Bi)/sqrt(sum(A^2))*sqrt(sum(B^2))
        a = 0
        b = 0
        c = 0
        for i in range(len(x)):
            a = a + x[i] * y[i]
            b = b + x[i]**2
            c = c + y[i]**2
        return a / (math.sqrt(b) * math.sqrt(c))


    #######################################################################################################################
	# Function cosine_distance(self, x, y):
	# Function to calculate the cosine distance with help of cosine similarity
	#######################################################################################################################
    def cosine_distance(self, x, y):
        return 1 - self.cosine_similarity(x, y)


def to_gray_uint8(im):
    if im is None:
        return None
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.uint8)

def resize_to_height(im, h):
    if im.shape[0] == h:
        return im
    w = int(round(im.shape[1] * (h / im.shape[0])))
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

def vsep(h, w=4, value=255):
    return np.full((h, w), value, dtype=np.uint8)

PAD = 24

def label_below(im, text, pad=PAD):
    strip = np.full((pad, im.shape[1]), 255, np.uint8)
    cv2.putText(strip, text, (5, pad-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
    return cv2.vconcat([im, strip])  # Höhe + pad

if __name__ == '__main__':
    searcher = Searcher("index.csv")
    feature_extractor = FeatureExtractor()
    img_path = "ImageCLEFmed2007_test/3216.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    limit = 5
    distances = searcher.search(feature_extractor.extract(img), limit)
    images = [img]
    for distance in distances:
        distance_name = distance["name"]
        images.append(cv2.imread(distance["name"], cv2.IMREAD_GRAYSCALE))
    img = to_gray_uint8(img)
    target_h = min(img.shape[0], 128)
    labeled_h = target_h + PAD

    # Query vorbereiten
    query_resized = resize_to_height(img, target_h)
    query_labeled = label_below(query_resized, "query", pad=PAD)

    # Kandidaten laden, auf gleiche Höhe bringen, labeln
    tiles = [query_labeled]
    for d in distances:
        im = to_gray_uint8(cv2.imread(d["name"], cv2.IMREAD_GRAYSCALE))
        if im is None:
            continue
        im = resize_to_height(im, target_h)
        name = d['name'].split("/")[-1]
        print(name)
        im = label_below(im, name, pad=PAD)
        tiles += [vsep(labeled_h), im]

    canvas = cv2.hconcat(tiles)
    cv2.imshow("Results", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
>>>>>>> origin/main
