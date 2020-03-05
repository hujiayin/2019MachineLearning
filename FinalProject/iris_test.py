from neuralnetwork import NeuralNetwork
import numpy as np
from sklearn import datasets


# load Iris dataset
iris = datasets.load_iris()
data_array = iris.data
target = iris.target

# Split data into 5 group
data_size = data_array.shape[0]
idx = np.array(range(data_size))
np.random.seed(100)
np.random.shuffle(idx)
idx_groups = np.array_split(idx, 5)
fold_groups = [(data_array[idxs], target[idxs]) for idxs in idx_groups]
five_sets = set(range(5))

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
    hid_node = 7
    out_node = 3
    NN = NeuralNetwork(in_node, hid_node, out_node, activation='sigmoid', seed=None)

    # Train the model
    NN.fit_model(train_X, train_y, learning_epochs=500, learning_rate=0.2,
                 pruning=False, verbose=False)
    acc = NN.accuracy(test_X, test_y)

    # compute accuracy and confusion matrix
    acc = 0
    y_pred = NN.predict_result(test_X)
    con_mat = np.zeros((3, 3))
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

