'''
Implementation for Image
@author Ajay Anand, Yashas Ravi, Steven Tan
'''

import math
import random
import numpy as np
from Image import Image

'''
INITIALIZATION
'''
# Learning Rate
ALPHA = 0.1

# Number of examples 
NUM_OF_EXAMPLES = 500

# Threshold for error
THRESHOLD = 0.000001

# Array of NUM_OF_EXAMPLES numpy arrays (each 1X400), where each array represents an 
# example (20X20 grid) and the elements represent the colors of the 400 pixels in the example. 
training_set = []

# Array of NUM_OF_EXAMPLE arrays, where each array has [R, G, B, Y] for booleans R, G, B, Y.
# This is basically one-hot encoding. 
outputs = []

# Populates training_set[] and outputs[] with pixel data and corresponding outputs
def create_training_set():
    # Loop control variable for number of iterations
    num_iterations = 0

    # Iterate over all the examples
    while (num_iterations < NUM_OF_EXAMPLES):
        # Create a new image, and extract data
        new_image = Image()
        # Data: example -> 20x20 array of pixels, output -> dangerous or not, color -> wire to be cut
        example, output, color = new_image.create_image()

        # If image is not dangerous, skip this example, and we need an extra iteration
        if (output == False):
            continue

        # Else, store the example and output color into training_set and outputs
        else:
            # Reshape example[][], which is a 20X20 array, into a 1X400 array
            flattened_example = np.array(example).reshape(-1)
            # Store this flattened array into training_set[]
            training_set.append(flattened_example)

            # Use one-hot encoding to store the wire to be cut into outputs[]
            if (color == 1):
                # Append red
                outputs.append([1,0,0,0])
            elif (color == 2):
                # Append blue
                outputs.append([0,1,0,0])
            elif (color == 3):
                # Append yellow
                outputs.append([0,0,1,0])
            elif (color == 4):
                # Append green
                outputs.append([0,0,0,1])
            else:
                # Invalid color and abort method
                print("INVALID COLOR")
                return
            
            # Update loop control variable SINCE we had a dangerous wire
            num_iterations = num_iterations + 1

'''
SOFTMAX REGRESSION
'''

# Initialize a 400x4 array of 0s for the weight_matrix[][]
weight_matrix = np.zeros((4,400))
# Initialize a 1x4 array of 0s for the bias for each color
bias_vector = np.zeros(4)

# Weight Matrix details:
    # Contains 4 arrays, one for each color (R, B, Y, G)
    # The contents in each array dictate the weights required for each pixel to predict that color.
    # Ex: If weight_matrix[0] = [w1, w2, w3 ... w400] is for red, then weight_matrix[0] * training_set[n] is predictor for red color in nth example.

# Here, we calculate the predicted outputs for each color using Softmax Regression.
# Calculation:
    # y_m = e^(w_m * x + b_m) / SUM(over j) e^(w_k * x + b_j)
def softmax_regression(example_number):
    # Use example_number to extract the example array and output
    current_example = training_set[example_number]
    current_output = outputs[example_number]

    #print(len(current_example))

    # Contains P(Y=k | data) for k = Red, Blue, Yellow, Green
    predicted_output = [0, 0, 0, 0]

    # Formula: y_k = e^(dot_product + bias_term for k) / SUM(over j)[e^(dot_product + bias term for j)]
        # We simplify this to y_k = numerator / denominator
    denominator = 0
    numerator = 0

    # Loop iterating through k+1=1,2,3,4 corresponding to colors R,G,B,Y
    for k in range(0,4):
        # Compute the dot product w_k * x
        dot_product = 0
        current_exponential = 0
        for pixel in range(400):
            dot_product = dot_product + current_example[pixel] * weight_matrix[k][pixel]
        
        # Extract the bias term b_k
        bias_term = bias_vector[k]

        # Compute the exponential e^(w_k * x + b_k)
        current_exponential = math.exp(dot_product + bias_term)

        # Store each exponential term in predicted_output[]
        predicted_output[k] = current_exponential
        
        # Append the exponential term to the denominator for future division
        denominator = denominator + current_exponential
    
    # Perform normalization on predicted_output[] by dividing every term by the denominator
    normalized_output = [numerator / denominator for numerator in predicted_output]

    # Return this array 
    return normalized_output

# The Loss function computes the loss between each data point and the predicted output
    # The formula L = SUM(over j) (y_j * log(yhat_j)) is used
def loss_function(example_number):
    # Counter for summation
    loss_summation = 0

    # Extract the data point from outputs[]
    actual_output = outputs[example_number]

    # Iterate over the colors and use softmax_regression() to get the predicted output
    for k in range(4):
        normalized_output = softmax_regression(example_number)
        loss_summation = loss_summation + (actual_output[k]) * math.log(normalized_output[k])
    
    # Return the final sum
    return loss_summation

# For weight_matrix = [[w11, w12, w13 ...], [w21, w22, w23 ...] ... ] for wkh for color k, data point h
    # We compute dL/dwkh = SUM(for j=1,2,3,4) [-y_j * 1/yhat_j * dyhat_j/dwkh] where yhat = normalized_output(example number, j)
    # dyhat_j/dwkh is different when j = k and when j is not equal to j
def dL_dwkh(example_number, k, h):
    # Counter for summation
    derivative_summation = 0

    # Predicted output using softmax_regression()
    normalized_output = softmax_regression(example_number)

    # Iterate over the colors
    for j in range(4):
        
        # Extract actual data point from outputs[]. This is y_j.
        actual_output = outputs[example_number][j]

        # Predicted value for color j using features (x1, x2, x3 ...) for current example
        yhat_j = normalized_output[j]

        # Predicted value for color k using features (x1, x2, x3 ...) for current example
        yhat_k = normalized_output[k]

        # The hth feature for the data point
        x_h =  training_set[example_number][h]

        # dy_j/dw_k,h here is initialized to 0 for future update
        dyj_dwkh = 0

        # Use a different formula to update dy_j/dw_k,h depending on if the current color we are 
        # summing over (SUM(over j)) is EQUAL to the color corresponding to w that we want to take the 
        # derivative of loss with respect to (dL/dw_k,h).
        if (j == k):
            dyj_dwkh = (yhat_j) * (1 - yhat_j) * x_h
        else:
            dyj_dwkh = (-yhat_j) * (yhat_k) * x_h

        # Update derivative_summation using the below formula
        derivative_summation = derivative_summation + (-1)*(actual_output) * (1/yhat_j) * dyj_dwkh
    
    # Return the final value for dL/dw_k,h
    return derivative_summation

# For weight_matrix = [[w11, w12, w13 ...], [w21, w22, w23 ...] ... ] for wkh for color k, data point h
    # We compute dL/dbk = SUM(for j=1,2,3,4) [-y_j * 1/yhat_j * dyhat_j/dwbk] where yhat = normalized_output(example number, j)
    # Simplification leads to dL/dbk = yhat_k - 1
def dL_dbk(example_number, k, h):
    # Extract the predicted output using softmax_regression()
    normalized_output = softmax_regression(example_number)

    # Store the normalized output corresponding to the color k
    yhat_k = normalized_output[k]

    # Use the below formula which uses yhat_k to return dL/db_k
    return yhat_k - 1

'''
GRADIENT DESCENT
'''

# Perform gradient descent on the loss function to optimize the weights so that the loss function is minimized
def stocastic_gradient_descent():
    # old_error initialized to 0, used for difference calculation
    old_error = 0

    # new_error initialized to infinity so that abs(new_error - old_error) >= THRESHOLD
    new_error = float('inf')

    # Iterate until the error becomes minimized, meaning delta(error) = change in error = abs(new_error - old_error) < THRESHOLD
    while abs(new_error - old_error) >= THRESHOLD:
        # Choose a random example for Stochastic Gradient Descent (SGD)

        # chosen_example_num = random.randint(0, NUM_OF_EXAMPLES)
        chosen_example_num = 0

        # Iterate over all features in the chosen example (x1, x2, x3, ...)
        for h in range(400):
            # Iterate over all color choices for each feature
            for k in range(4):
                # Store the weight w_h,k from the weight_matrix. This matrix is UPDATED as the loop iterates and SGD progresses
                weight_hk = weight_matrix[k][h]

                # Compute the dL/dw_k,h using dL_dwkh()
                weight_gradient = dL_dwkh(chosen_example_num, k, h)

                # Compute the new weight by changing it by (learning rate)*(derivative)
                new_weight_hk = weight_hk - ALPHA * weight_gradient

                # Store the new weight in the weight matrix so that the weight matrix better maps inputs to the output data
                weight_matrix[k][h] = new_weight_hk

        # Iterate over all color choices for bias
        for k in range(4):
            # Extract the bias corresponding to current color. This vector is UPDATED as the loop iterates and SGD progresses
            bias_k = bias_vector[k]

            # Compute dL/db_k using dL_dbk()
            bias_gradient = dL_dbk(chosen_example_num, k, h)

            # Compute the new bias by changing it by (learning rate)*(derivative)
            new_bias_k = bias_k - ALPHA * bias_gradient

            # Store the new bias in the bias vector so that the bias vector better maps inputs to the output data
            bias_vector[k] = new_bias_k

        # Compute the loss function using the newly updated weights
        new_error = loss_function(chosen_example_num)

        # Update old_error so that the loop condition computes the correct value in the next iteration
        old_error = new_error

# Compute the sucess rte using the weight matrix OPTIMIZED by SGD
def compute_success_rate():
    # Count the number of sucesses
    success_count = 0

    # Iterate over all examples. This forms the training set
    for training_example in range(NUM_OF_EXAMPLES):
        # Extract the correct outputs (each one is a one-hot encoded array) from outputs[]
        correct_output = outputs[training_example]
        # Calculate the model output (a one-hot encoded array) using the prediction from softmax_regression()
        model_output = softmax_regression(training_example)
        # The index corresponding to the correct color in correct_output. This will be 1.
        correct_color_actual = correct_output.index(max(correct_output))
        # The index corresponding to the correct color using the prediction. This will be 
        # the color with the highest probability for this example computed by softmax_regression()
        correct_color_predicted = model_output.index(max(model_output))

        # If the indices between actual and predicted match, then the model has made a correct prediction and we increment success_rate
        if (correct_color_actual == correct_color_predicted):
            success_count = success_count + 1 
        else:
            continue

    # Compute the rate by dividing count by the TOTAL number of examples
    success_rate = success_count / NUM_OF_EXAMPLES

    # Return the final success rate
    return success_rate

# Runs the entire Multiclass Classification and performs Stochastic Gradient Descent (SGD) to compute the optimal parameters
def run_multiclass_classification():

    # Create the training dataset for NUM_OF_EXAMPLES images, each a 1X400 array of pixels
    create_training_set()

    # Perform Stochastic Gradient Descent (SGD) to compute the loss function for every example
    # using softmax_regression() and loss_function(), and optimize the weights for every example
    stocastic_gradient_descent()

    # Using the weights optimized by SGD, calculate the success rate 
    success_rate = compute_success_rate()

    # Return the success rate
    return success_rate

# Main method to run Multiclass Classification
if __name__ == '__main__':
    
    create_training_set()
    stocastic_gradient_descent()
    final_rate = run_multiclass_classification()
    print(final_rate)


    # print(outputs[0])








    
