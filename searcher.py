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
        self._image_paths = None
        self._index = None

    # def _load_index(self):
    #     if self._index is not None:
    #         return
    #
    #     if not os.path.exists(self.index_path):
    #         raise FileNotFoundError("Index does not exist")
    #
    #     with open(self.index_path) as f:
    #         n_rows = sum(1 for _ in f)
    #
    #     index = np.zeros(n_rows)
    #     # index = []
    #     with open(self.index_path) as csvfile:
    #         reader = csv.reader(csvfile, delimiter=",")
    #         for idx, row in enumerate(reader):
    #             if not row:
    #                 continue
    #             image_path = row[0]
    #             features = [np.float32(x) for x in row[1:]]
    #             index[idx] = features
    #             # index.append((image_path, features))
    #
    #     self._index = index

    def _load_index(self):
        if self._index is not None:
            return

        if not os.path.exists(self.index_path):
            raise FileNotFoundError("Index does not exist")
        # Erstmal die Zeilen holen (ohne leere)
        with open(self.index_path) as f:
            rows = [row for row in csv.reader(f, delimiter=",") if row]

        n_rows = len(rows)
        if n_rows == 0:
            raise ValueError("Index file is empty")

        feature_dim = len(rows[0]) - 1  # alles außer image_path

        # Matrix für Features + Liste für Pfade
        index = np.empty((n_rows, feature_dim), dtype=np.float32)
        image_paths = []

        for i, row in enumerate(rows):
            image_paths.append(row[0])
            # Hier die Features – das ist der entscheidende Teil:
            index[i] = np.asarray(row[1:], dtype=np.float32)
        self._index = index
        self._image_paths = image_paths


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
    def search(self, query_features, limit=5):
        self._load_index()

        q = np.asarray(query_features, dtype=np.float32)

        # Compute cosine similarities for all images at once using matrix–vector multiplication.
        #
        # Explanation:
        #   - self._index is a 2D NumPy array of shape (N, D)
        #       N = number of images in the database
        #       D = dimensionality of each feature vector
        #
        #   - q is the query feature vector with shape (D,)
        #
        #   - The expression "self._index @ q" performs a matrix–vector dot product.
        #       For each row x_i in self._index, NumPy computes:
        #
        #           x_i · q     (the dot product)
        #
        #       This results in a 1D array of length N:
        #
        #           [ x_0 · q,
        #             x_1 · q,
        #             ...
        #             x_(N-1) · q ]
        #
        #   - Because all feature vectors (including q) are L2-normalized,
        #       the dot product x_i · q is exactly the cosine similarity:
        #
        #           cos_sim(x_i, q) = x_i · q
        #
        #   - This is extremely fast because the entire computation is executed
        #       in optimized C/BLAS code, not in Python loops.
        #
        #   - In summary:
        #         self._index @ q
        #       gives you cosine similarity for all images in one shot.
        sims = self._index @ q  # Shape: (N,)
        dists = 1.0 - sims  # cosine distance

        # get smallest distance
        idx = np.argsort(dists)[:limit]

        # return in a dictionary
        return [
            {"name": self._image_paths[i], "distance": dists[i]}
            for i in idx
        ]

    # def search(self, query_features, limit = 5):
    #     if not os.path.exists(self.index_path):
    #         raise FileNotFoundError("Index does not exist")
    #     distances = []
    #     with open(self.index_path) as csvfile:
    #         reader = csv.reader(csvfile, delimiter=",")
    #         for row in reader:
    #             if not row:
    #                 continue  # skip empty lines
    #             image_path = row[0]
    #             features = [float(x) for x in row[1:]]
    #             distances.append({"name":image_path, "distance":self.cosine_distance(features, query_features)})
    #     distances.sort(key=lambda x: x["distance"])
    #     return distances[:limit]

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
        # distance = 0
        # if len(x) != len(y):
        #     raise ValueError("Length mismatch")
        # for i in range(len(x)):
        #         distance = distance + (x[i] - y[i]) ** 2
        # return math.sqrt(distance)
        return np.linalg.norm(x - y)


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
        # distance = 0
        # if len(x) != len(y):
        #     raise ValueError("Length mismatch")
        # for i in range(len(x)):
        #     distance = distance + abs(x[i] - y[i])
        # return distance
        return np.sum(np.abs(x - y))


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
        if x.shape != y.shape:
            raise ValueError("Length mismatch")

        diff = np.abs(x - y) ** p
        return np.sum(diff) ** (1.0 / p)

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
        # a = 0
        # b = 0
        # c = 0
        # for i in range(len(x)):
        #     a = a + x[i] * y[i]
        #     b = b + x[i]**2
        #     c = c + y[i]**2
        # return a / (math.sqrt(b) * math.sqrt(c))

        # Since both vectors are L2-normalized, the cosine similarity reduces to
        # a simple dot product. For normalized vectors: cos_sim(x, y) = x · y
        return np.dot(x, y)


    #######################################################################################################################
	# Function cosine_distance(self, x, y):
	# Function to calculate the cosine distance with help of cosine similarity
	#######################################################################################################################
    def cosine_distance(self, x, y):
        return np.float32(1) - self.cosine_similarity(x, y)


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
