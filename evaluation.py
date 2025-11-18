from itertools import repeat

from preprocessing import get_images_paths
from query import Query
import cv2
import sys
import numpy as np
from pathlib import Path
import os
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading

code_path = "codes.csv"
_codes_lock = threading.Lock()

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
        cutoff_prediction_list = correct_prediction_list[:k]

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
        if rel:
            average_precision += precision_at_k(correct_prediction_list, idx + 1)
    return (1 / amount_relevant) * average_precision

codes = None
code_cnt = None

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
    global codes, code_cnt
    with _codes_lock:
        if codes is None:
            codes = {}
            code_cnt = {}

            # open the code file for reading
            with open(code_path) as f:
                # initialize the CSV reader
                reader = csv.reader(f)

                # loop over the rows in the index
                for row in reader:
                    # add to dictionary; Key: file name, Item: IRMA code
                    codes[row[0]] = row[1]
                    if row[1] not in code_cnt:
                        code_cnt[row[1]] = 1
                    else:
                        code_cnt[row[1]] += 1
            if not query_image_code in codes.keys():
                return 0
            else:
                return code_cnt[codes[query_image_code]]
        else:
            return code_cnt[codes[query_image_code]]


def run_query(image_path, limit):
    query = Query(image_path, limit)
    results = query.run()
    if results:
        results.pop(0)
    return results

def get_relevant_cnt(image_path):
    code = os.path.basename(image_path.split(".")[0])
    return amount_relevant_images(code)

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
    # get image paths of all images
    image_paths = get_images_paths(image_directory="./ImageCLEFmed2007_test/", file_extensions=(".png"))
    # retrieve amount of images
    amount_images = len(image_paths)
    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(run_query, image_paths, repeat(limit)),
                total=len(image_paths),
                desc="Processing queries",
            )
        )
    with ThreadPoolExecutor() as executor:
        relevant_results = list(
            tqdm(
                executor.map(get_relevant_cnt, image_paths),
                total=len(image_paths),
                desc="Processing relevant images",
            )
        )
    relevant_mask = list()
    aps = np.zeros(len(results))
    for q_idx, hits in enumerate(results):
        cur_code = os.path.basename(image_paths[q_idx].split(".")[0])
        try:
            relevant_code = codes[cur_code]
            mask = np.zeros(len(hits), dtype=bool)
        except KeyError:
            raise Exception(f"Code {cur_code} not found in codes.csv")
        for r_idx, hit in enumerate(hits):
            hit_code = codes[os.path.basename(hit['name']).split(".")[0]]
            if relevant_code == hit_code:
                mask[r_idx] = True
        aps[q_idx] = average_precision(mask, len(hits))
    return float(np.mean(aps))


if __name__ == "__main__":
    test = [True, True, False, False]
    print("P@K: ", precision_at_k(test))
    print(amount_relevant_images("3145"))
    print("AveP: ", average_precision(test, 5))

    result = mean_average_precision(limit=10)
    print("\nMAP: ", result)