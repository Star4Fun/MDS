# pylint: disable=no-member
import numpy as np
import cv2
from feature_extractor import FeatureExtractor
from searcher import Searcher
import easygui
from pathlib import Path
import csv
import os

#######################################################################################################################
# Function get_filename_from_path(self, path_to_file, file_ending = '.png'):
# Function to retrieve a file name from a file path
#
# Hint: "./images/739609.png" -> "739609". 
# You can use 'os.path.basename'
# 
# Input arguments:
#   - [string] path_to_file: Path to a file.
#   - [string] file_ending: String of the file type. Default = '.png'
# Output argument:
#   - [string] name: Name of the file
######################################################################################################################
def get_filename_from_path(path_to_file, file_ending = '.png'):
    filename = os.path.basename(path_to_file)

    if filename.endswith(file_ending):
        filename = filename.replace(file_ending,'')

    return filename

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

def vsep(height, width=10):
    return np.full((height, width, 3), 255, np.uint8)  # weißer vertikaler Trenner

PAD = 24

def label_below(im, text, pad=PAD):
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    h, w = im.shape[:2]
    strip = np.full((pad, w, 3), 255, np.uint8)  # weißer Balken unten

    cv2.putText(
        strip,
        text,
        (5, pad - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),   # schwarzer Text
        1,
        cv2.LINE_AA
    )

    return cv2.vconcat([im, strip])



def add_frame(img, thickness=4, color=(0, 0, 255)):
    """
    Add a colored frame around an image.

    img:      grayscale (H, W) oder BGR (H, W, 3)
    thickness: Rahmenbreite in Pixeln
    color:    BGR-Farbe als Tuple, z.B. (0, 0, 255) = rot
    """
    # Falls Graubild: erst zu BGR machen
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return cv2.copyMakeBorder(
        img,
        thickness, thickness, thickness, thickness,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

codes = None
code_path = "codes.csv"

def get_codes():
    global codes
    if codes is not None:
        return codes
    codes = {}

    # open the code file for reading
    with open(code_path) as f:
        # initialize the CSV reader
        reader = csv.reader(f)

        # loop over the rows in the index
        for row in reader:
            # add to dictionary; Key: file name, Item: IRMA code
            codes[row[0]] = row[1]
    return codes

class Query:
    output_name = "index.csv"
    image_directory = "./ImageCLEFmed2007_test/*"
    searcher = Searcher(output_name)

    #######################################################################################################################
	# Function __init__(self, query_image_name = None):
	# Init function. Just call set_image_name
	#######################################################################################################################
    def __init__(self, query_image_name = None, limit = 10):
        self.limit = limit
        self.image_features = None
        self.query_image_name = None
        self.query_image_code = None
        self.query_image = None
        self.features = None
        self.set_image_name(query_image_name)

    #######################################################################################################################
	# Function set_image_name(self, query_image_name = "./images/739562.png"):
	# Function to set the image name. Afterwards the image is loaded
	#
	# Steps: Set variable "self.query_image_name". Load image and set to "self.query_image". Call "calculate_features(self)"
    # Hint: You can also use a file dialogue to automatic enter the filename 
    #       -> http://easygui.sourceforge.net/api.html?highlight=fileopenbox#easygui.fileopenbox 
	# 
	# Input arguments:
	#   - [string] query_image_name: Path to image name. Default: like "./images/739609.png"
	#######################################################################################################################
    def set_image_name(self, query_image_name = None):
        if query_image_name is None:
            query_image_name = easygui.fileopenbox(default = self.image_directory, filetypes = ["*.png"])
        self.query_image_name = query_image_name
        self.query_image = cv2.imread(query_image_name, cv2.IMREAD_GRAYSCALE)
        self.calculate_features()

     #######################################################################################################################
	# Function calculate_features(self):
	# Function to calculate featres
	#
	# Steps: Check if "self.query_image" is None -> exit(). Extract features wit "FeatureExtractor" and set to "self.features"
	#######################################################################################################################
    def calculate_features(self):
        if self.query_image is None:
            print("There is no image named: ", self.query_image_name)
            exit()
        # initialize the FeatureExtractor
        feature_extractor = FeatureExtractor()
        # describe the query_image
        self.features = feature_extractor.extract(self.query_image)

    #######################################################################################################################
	# Function run(self):
	# Function to start a query
	#
	# Steps: 
    #   Check if "self.query_image" or self.features is None -> return. 
    #   Create a Searcher and run a search
	#######################################################################################################################
    def run(self):
        if self.query_image is None:
            print("Error: Image not found")
            return
        if self.features is None:
            print("Error: Feature not found")
            return
        # perform the search
        results = self.searcher.search(self.features, self.limit)

        # If we do not get any results, we quit
        if results is False:
            quit()
        return results

    

    #######################################################################################################################
	# Function check_code(self, query_result):
	# Function to check if the codes of the retrieved images are equal to the code of the query_image
	#
    # Steps:
    #   - Read in the csv file with the codes
    #   - Create a Dictionary 'codes' similar to the csv file -> key : file name , item : IRMA code
    #   - Creat a Dictionary 'coorect_prediciton' with all retrieved images -> key : file name, item : boolen if the same to query_image
	# 
	# Input arguments:
	#   - [list] query_result: Result of 'run'
    # Output argument:
    #   - [Dictionary] correct_prediction:  key : file name, item : boolen
	#######################################################################################################################
    def check_code(self, query_result):
        # check if there is a csv file
        if not Path(code_path).exists():
            print("There is no code file: ", code_path)
            return {}

        codes = get_codes()

        # get the name 
        query_image_name = get_filename_from_path(self.query_image_name)

        # If we cannot find the code for the query image
        if query_image_name not in codes:
            print("There is no code for: ", query_image_name)
            # Return empty dictionary
            return {}

        # get the code of the query image
        self.query_image_code = codes[query_image_name]

        correct_prediction = {}

        # loop over each retrieved element
        for result_image in query_result:
            # get path to file
            path_to_image = result_image['name']
            # get name of file
            image_name = get_filename_from_path(path_to_image)

            # check if there is a code for the image
            if image_name in codes:
                # get code of file
                image_code = codes[image_name]
                # save in dictionary if it was a correct prediction
                correct_prediction[image_name] = (image_code == self.query_image_code)
            else:
                print("There is no code for: ", image_name)

        return correct_prediction
            

    #######################################################################################################################
	# Function visualize_result(self, query_result, correct_prediction_dictionary):
	# Function tovisualize the results of the previous functions
	#
    # Steps:
    #   - Read in and resize (200, 200) the query image (color)
    #   - Loop over the retrieved results:
    #       - Read in the retrieved image (color)
    #       [- Retrieve whether the code is similar to query_image]
    #       [- Add a border depending on the code around the image (cv2.copyMakeBorder)]
    #       - Resize the image (200, 200)
    #       - Concatenate it to the query_image
    #   - Display the result
    #   - WaitKey
    #   - destroyWindow
	# 
	# Input arguments:
	#   - [list] query_result: Result of 'run'
    #   [- [Dictionary] correct_prediction:  Results of check_code]
	#######################################################################################################################
    def visualize_result(self, query_result, correct_prediction_dictionary, image_size = (150,150)):
        assert self.query_image is not None
        limit = 5
        img = to_gray_uint8(self.query_image)
        target_h = min(img.shape[0], image_size[0])
        labeled_h = target_h + PAD

        # Query vorbereiten
        query_resized = resize_to_height(img, target_h)
        query_labeled = label_below(query_resized, "query", pad=PAD)

        # Kandidaten laden, auf gleiche Höhe bringen, labeln
        tiles = [query_labeled]
        for idx, d in enumerate(query_result):
            im = to_gray_uint8(cv2.imread(d["name"], cv2.IMREAD_GRAYSCALE))
            if im is None:
                continue
            name = d['name'].split("/")[-1]
            print(name)
            color = ((0, 255, 0) if correct_prediction_dictionary[name.split(".")[0]] else (0, 0, 255))
            im = add_frame(im, color=color)
            im = resize_to_height(im, target_h)
            im = label_below(im, name, pad=PAD)
            tiles += [vsep(labeled_h), im]

        canvas = cv2.hconcat(tiles)
        cv2.imshow("Results", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    while(True):
        query = Query()
        query_result = query.run()
        print("Retrieved images: ", query_result)
        correct_prediction_dictionary = query.check_code(query_result)
        print("correct_prediction_dictionary:")
        print(correct_prediction_dictionary)
        query.visualize_result(query_result, correct_prediction_dictionary)
