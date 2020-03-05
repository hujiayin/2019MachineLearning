import pandas as pd
import numpy as np
from neuralnetwork import NeuralNetwork
from improvedNN import ImprovedNeuralNetwork

adult_data = pd.read_csv('adult_clean.csv')
adult_array = np.array(adult_data.iloc[:, 1:-1])
adult_target = np.array(adult_data.iloc[:, -1])

# Normalize adult data
def normalize(dataframe):
    for col in range(dataframe.shape[1]):
        min = np.min(dataframe.iloc[:, col])
        max = np.max(dataframe.iloc[:, col])
        dataframe.iloc[:, col] = (dataframe.iloc[:, col] - min)/(max - min)
    return dataframe


adult_data.iloc[:, 1:4] = normalize(adult_data.iloc[:, 1:4])
adult_array_norm = np.array(adult_data.iloc[:, 1:-1])


# Split data into 5 groups
data_size = adult_array_norm.shape[0]
idx = np.array(range(data_size))
np.random.seed(100)
np.random.shuffle(idx)
idx_groups = np.array_split(idx, 5)
fold_groups = [(adult_array_norm[idxs], adult_target[idxs]) for idxs in idx_groups]
five_sets = set(range(5))


# Test neural network
def test_nn(hid_node, learning_rate, activation_function, pruning_choice):
    acc_stat = []
    con_mat_stat = []
    for i in range(len(fold_groups)):
        print("Round: " + str(i+1))

        # Select test set and train data
        test_group_idx = i
        test_X, test_y = fold_groups[i]

        train_group_idx = list(five_sets - {i})
        train_X = [fold_groups[idx][0] for idx in train_group_idx]
        train_y = [fold_groups[idx][1] for idx in train_group_idx]

        train_X = np.concatenate(train_X)
        train_y = np.concatenate(train_y)

        # Initialize model
        in_node = train_X.shape[1]
        out_node = 1
        NN = NeuralNetwork(in_node, hid_node, out_node, activation=activation_function, seed=None)

        # Train the model
        if pruning_choice:
            NN.fit_model(train_X, train_y, learning_epochs=20, learning_rate=learning_rate,
                         pruning=pruning_choice, pruning_threshold=0.2,
                         pruning_norm='L1', pruning_epoch=10)
        else:
            NN.fit_model(train_X, train_y, learning_epochs=20, learning_rate=learning_rate,
                         pruning=pruning_choice)
        acc = NN.accuracy(test_X, test_y)

        # Compute accuracy and confusion matrix
        acc = 0
        y_pred = NN.predict_result(test_X)
        con_mat = np.zeros((2, 2))
        for i in range(len(y_pred)):
            con_mat[int(y_pred[i]), int(test_y[i])] += 1
            if test_y[i] == y_pred[i]:
                acc += 1
        con_mat_stat.append(con_mat)

        acc = acc/len(y_pred)
        acc_stat.append(acc)

        print('\nACCURACY: ', acc)
        print('CONFUSION MATRIX: \n', con_mat)
        print("\n")

    print("\nACCURACY AVERAGE: ", np.average(acc_stat))
    print("CONFUSION MATRIX AVERAGE: \n", np.average(con_mat_stat, axis=0))


# Test improved neural network
def test_improvednn(hid_node, learning_rate, activation_function, pruning_choice):
    acc_stat = []
    con_mat_stat = []
    for i in range(len(fold_groups)):
        print("Round: " + str(i+1))

        # Select test set and train data
        test_group_idx = i
        test_X, test_y = fold_groups[i]

        train_group_idx = list(five_sets - {i})
        train_X = [fold_groups[idx][0] for idx in train_group_idx]
        train_y = [fold_groups[idx][1] for idx in train_group_idx]

        train_X = np.concatenate(train_X)
        train_y = np.concatenate(train_y)

        # Initialize model
        in_node = train_X.shape[1]
        out_node = 1
        INN = ImprovedNeuralNetwork(in_node, hid_node, out_node, activation=activation_function, seed=None)

        INN.fit_model(train_X, train_y, learning_epochs=20, learning_rate=learning_rate, pruning=pruning_choice)
        acc = INN.accuracy(test_X, test_y)

        # Compute accuracy and confusion matrix
        acc = 0
        y_pred = NN.predict_result(test_X)
        con_mat = np.zeros((2, 2))
        for i in range(len(y_pred)):
            con_mat[int(y_pred[i]), int(test_y[i])] += 1
            if test_y[i] == y_pred[i]:
                acc += 1
        con_mat_stat.append(con_mat)

        acc = acc/len(y_pred)
        acc_stat.append(acc)

        print('\nACCURACY: ', acc)
        print('CONFUSION MATRIX: \n', con_mat)
        print("\n")

    print("\nACCURACY AVERAGE: ", np.average(acc_stat))
    print("CONFUSION MATRIX AVERAGE: \n", np.average(con_mat_stat, axis=0))


# # Tuning parameter number of neurons in hidden layer
# hid_node_list = [2, 5, 10, 20, 30, 50]
# learning_rate = 0.01
# activation_function = 'sigmoid'
# pruning_choice = False
# for item in hid_node_list:
#     print('number of hidden node:', item)
#     test_nn(item, learning_rate, activation_function, pruning_choice)

# # Tuning parameter learning rate
# hid_node = 10
# learning_rate_list = [0.001, 0.01, 0.1, 1, 5]
# activation_function = 'sigmoid'
# pruning_choice = False
# for item in learning_rate_list:
#     print('learning rate:', item)
#     test_nn(hid_node, item, activation_function, pruning_choice)

# # Change activation function
# hid_node = 10
# learning_rate = 0.01
# activation_function_list = ['sigmoid', 'hyperbolic tangent', 'rectify linear']
# pruning_choice = False
# for item in activation_function_list:
#     print('activation function:', item)
#     test_nn(hid_node, learning_rate, item, pruning_choice)


# # Pruning
# hid_node = 10
# learning_rate = 0.01
# activation_function = 'hyperbolic tangent'
# pruning_choice = True
# test_nn(hid_node, learning_rate, activation_function, pruning_choice)

# # Test Improved NN
# hid_node = 10
# learning_rate = 0.01
# activation_function_list = ['sigmoid', 'hyperbolic tangent']
# pruning_choice = False
# for item in activation_function_list:
#     test_improvednn(hid_node, learning_rate, item, pruning_choice)

