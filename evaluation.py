from preprocessing import get_images_paths
from query import Query
import cv2
import sys
import numpy as np
from pathlib import Path
import os
import csv

code_path = "codes.csv"


#######################################################################################################################
# Function precision_at_k(correct_prediction_list, k = None)
# Function to calculate the precision@k for this query
#
#   Tasks:
#   - If k is not defined -> k should be length of list
#   - If k > length -> Error
#   - If k < length -> cut off correct_prediction_list at k
#   - Calculate precision for list
#
#
# Input arguments:
#   - [dict] correct_prediction_list: list with the True/False for the retrieved images
#   - [int] k
# Output argument:
#   - [float] average precision
#######################################################################################################################
def precision_at_k(correct_prediction_list, k=None):
    cutoff_prediction_list = []
    true_positive = 0
    if k == None:
        k = len(correct_prediction_list)
    if k > len(correct_prediction_list):
        raise Exception("k needs to be smaller than length of list")
    if k <= len(correct_prediction_list):
        cutoff_prediction_list = correct_prediction_list[:k + 1]

    for x in cutoff_prediction_list:
        if x == True:
            true_positive += 1
    precision_at_k = (true_positive) / k

    return precision_at_k


#######################################################################################################################
# Function average_precision(self, amount_relevant):
# Function to calculate the average precision for correct_prediction_list
#
#   Task:
#   - Calculate precision for list
#
#
# Input arguments:
#   - [dict] correct_prediction_list: list with the True/False for the retrieved images
#   - [int] amount_relevant: # relevant documents for the query
# Output argument:
#   - [float] average precision
#######################################################################################################################
def average_precision(correct_prediction_list, amount_relevant):
    if amount_relevant == 0:
        return 0.0
    average_precision = 0.0
    for idx, rel in enumerate(correct_prediction_list):
        average_precision += precision_at_k(correct_prediction_list, idx + 1)
    return (1 / amount_relevant) * average_precision


#######################################################################################################################
# Function amount_relevant_images(self, image_name):
# Function to retrieve the amount of relavant images for a image name
#
#   Tasks:
#   - Check if path to "code_path" exists: if not -> print error and return False
#   - Iterate over every row of code file
#   - Count the amount of codes queal to "query_image_code" TTTT-DDD-AAA-BBB
#
# Input arguments:
#   - [String] image_name: Name of the image
# Output argument:
#   - [int] amount
#######################################################################################################################
def amount_relevant_images(query_image_code):
    if not os.path.exists(code_path):
        print("Error: codes.csv does not exist")
        return False
    codes = {}

    # open the code file for reading
    with open(code_path) as f:
        # initialize the CSV reader
        reader = csv.reader(f)

        # loop over the rows in the index
        for row in reader:
            # add to dictionary; Key: file name, Item: IRMA code
            codes[row[0]] = row[1]
    if not query_image_code in codes.keys():
        return 0
    else:
        image_code = codes[query_image_code]
    cnt = 0
    for item_code, item in codes.items():
        if item == image_code:
            cnt += 1
    return cnt


#######################################################################################################################
# Function mean_average_precision():
# Function to calcualte the mean average precision
#
#   Tasks:
#   - Iterate over every image path
#      - Create and run a query for each image
#      - Compute correct_prediction_dictionary
#      - Create a list from the dict
#      - Remove the first element (its the query image)
#      - Compute amount of relevant images (function)
#      - Compute AP (function) and save the value
#   - Compute mean of APs
#
# Input arguments:
# Output argument:
#   - [float] mean average precision
#######################################################################################################################
def mean_average_precision(limit=20):
    ap_list = []
    # get image paths of all  images
    image_paths = get_images_paths(image_directory="./images/", file_extensions=(".png"))
    # retrieve amount of images
    amount_images = len(image_paths)

    # TODO
    pass


if __name__ == "__main__":
    test = [True, True, False, False]
    print("P@K: ", precision_at_k(test))
    # amount_relevant_images("3145")
    print("AveP: ", average_precision(test, 5))

    result = mean_average_precision(limit=10)
    print("\nMAP: ", result)