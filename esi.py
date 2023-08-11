import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## BUILD CONFUSION MATRIX ##
# The main return uses the confusion matrix conf_matrix, however, there are
# potentially many confusion matrices that could be set up, for example the
# Steiner 2020 Fig 2D multi-reader dataset affords the opportunity to generate
# many confusion matrices.  So it may be useful to extract various confusion
# matrices from the data, and swap them into the final conf_matrix as we figure
# out what it is we want to draft.
#
# This crosstab method of constructing a confusion matrix from here:
# https://datatofish.com/confusion-matrix-python/. This works as long as all
# cells are populated. For production, may need to check for the input for 
# empty rows and NaN cells and delete them:
# https://stackoverflow.com/a/17092718

df = pd.read_csv('./Steiner2020_assisted.csv')
print(df.nunique())
df.head(15)
confusion_matrix_A = pd.crosstab(df['GU_majority_Ground_truth'], df['B'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix_A)

## FUNCTIONS ##

def error_severity(conf_matrix, esi_weights):
    '''
    Error severity takes in a confusion matrix and returns a measure of the overall predictive value of a given classifier. TODO: check that the conf_matrix is of the same shape as the esi_weights
    '''
    score = 0
    num_elements = sum([sum(r) for r in conf_matrix])
    if num_elements == 0:
      raise("No elements")
      
    for i, row in enumerate(conf_matrix):
        for j, x in enumerate(row):
            score += x * esi_weights[j][i]

    return score / num_elements

def error_severity_index(esi_weights, conf_matrix):
    '''
    Error severity index takes in a weight matrix and error serverity and returns a measure of the overall predictive value of a given classifier. TODO: check that the conf_matrix is of the same shape as the esi_weights
    '''
    error_severity = 0
    error_severity_index = 0
    weight_min = 0
    weight_min = min(np.hstack(esi_weights)[np.hstack(esi_weights)!=0])
    num_elements = sum([sum(r) for r in conf_matrix])
    if num_elements == 0:
      raise("No elements")

    if ((np.sum(conf_matrix)-np.trace(conf_matrix)) != 0):
      
      for i, row in enumerate(conf_matrix):
          for j, x in enumerate(row):
              error_severity += x * esi_weights[j][i]
      
      error_severity = error_severity /(np.sum(conf_matrix)-np.trace(conf_matrix))
      error_severity_index = (10 - 1) * (error_severity - weight_min) / (1 - weight_min) + 1
      
    return np.round(error_severity, 2), np.round(error_severity_index, 2)   

## BUILD WEIGHTS MATRICES ##
# Similar to conf_matrix, it may be useful to have various weights matrices to
# swap in.

weights_naive = [[1,0,0], 
                 [0,1,0], 
                 [0,0,1]] 

esi_weights =   [[0,0.3,0.6,1],
                 [0.3,0,0.3,0.6],
                 [0.6,0.3,0,0.3],
                 [1,0.6,0.3,0]]

table2a = [[20,0,5,0],
           [0,20,0,0],
           [0,0,20,0],
           [5,5,0,25]]

table2weights = [[1,0.5,0],
                 [0.5,1,0.5],
                 [0.0,0.5,1]]
table2a = [[80,10,0],
           [20,80,12],
           [0,10,8]]
table2b = [[80,10,6],
           [10,80,6],
           [10,10,8]]
table2c = [[80,10,12],
           [0,80,0],
           [20,10,8]]

table3_75_weights = [[1,0.5,0],
                     [0.5,1,0.5],
                     [0.0,0.5,1]] 
table3_75_a = [[90,20,25],
               [20,50,15],
               [15,15,50]]
table3_75_b = [[90,5,5],
               [10,90,10],
               [5,10,50]]
table3_90_weights = [[1,0.5,0],
                     [0.5,1,0.5],
                     [0.0,0.5,1]] 
table3_90_a = [[90,5,5],
               [10,90,10],
               [5,10,50]]
table3_90_b = [[90,10,0],
               [15,80,15],
               [0,15,50]]
table3_89_weights = [[1,0.3,0],
                     [0.7,1,0.3],
                     [0.4,0.6,1]] 
table3_89_a = [[41,0,0],
              [35,70,7],
              [4,10,33]]
table3_89_b = [[80,10,9],
              [0,70,0],
              [0,0,31]]
figEA1_a =  [[0,0.3,0.6,1],
             [0.3,0,0.3,0.6],
             [0.6,0.3,0,0.3],
             [1,0.6,0.3,0]]
figEA1_b =[[20,0,0,0],
          [5,20,5,0],
          [0,5,20,0],
          [0,0,0,25]]
figEA1_c =[[20,2,3,0],
           [2,20,2,0],
           [3,3,20,0],
           [0,0,0,25]]
figEA1_d =[[20,0,5,0],
           [0,20,0,0],
           [0,0,20,0],
           [5,5,0,25]]

## MAIN: CALCULATE THE FIGURES OF MERIT ##
# Confusion matrix follows the convention here
# https://i.stack.imgur.com/a3hnS.png. The element in the ith row and jth column
# is the the number of elements with ground truth j which were classified as i 
# by the model.

# Confusion matrix must have num_classes rows and columns. All elements should
# be non-negative integers.

num_classes = 5
conf_matrix = confusion_matrix_A.to_numpy(dtype=int)
weights = weights_naive

print(conf_matrix)
print(weights)
print("Error severity index:", error_severity_index(conf_matrix, esi_weights))