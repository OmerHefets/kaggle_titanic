import numpy as np


# count the number of different predictions
def compare_two_predictions(set_1, set_2):
    # counter
    diff_results = 0
    if len(set_1) != len(set_2):
        exit("two prediction sets with different sizes")
    for i in range(len(set_1)):
        # Use XOR to identify when prediction is different
        if bool(set_1[i]) ^ bool(set_2[i]):
            diff_results += 1
    print("The percentage of difference between the sets is: {}".format(diff_results / len(set_1)))
