import numpy as np
import time


class ImprovedNeuralNetwork(object):
    """
    3 layers Neural Network
    """

    def __init__(self, in_node, hid_node, out_node, activation='sigmoid',  seed=None):
        """
        Initialize the weights and methods to train the neural network.

        :param in_node: number of nodes in input layer
        :param hid_node: number of nodes in hidden layer
        :param out_node: number of nodes in output layer
        :param activation: activation function used transform the output of each layer
        :param seed: random seed for repeating test
        """

        self.in_node = in_node
        self.hid_node = hid_node
        self.out_node = out_node
        self.hidden_array = np.ones(hid_node)
        self.out_array = np.ones(out_node)
        # self.competition = competition

        # Weights initialization by random number
        if seed:
            random_seed = seed
        else:
            random_seed = np.random.randint(10000000)

        np.random.seed(random_seed)
        self.weight_in = np.random.randn(in_node + 1, hid_node)
        np.random.seed(random_seed)
        self.weight_out = np.random.randn(hid_node + 1, out_node)

        if activation == 'sigmoid':
            self.activation = self.__sigmoid
            self.activation_derivative = self.__sigmoid_update
        # elif activation == 'step':
        #     self.activation = self.__step
        #     self.activation_derivative = self.__step_update
        elif activation == 'rectify linear':
            self.activation = self.__ReLU
            self.activation_derivative = self.__ReLU_update
        elif activation == 'hyperbolic tangent':
            self.activation = self.__hyperbolic_tangent
            self.activation_derivative = self.__hyperbolic_tangent_update

    # activation function
    @staticmethod
    def __sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def __sigmoid_update(x):
        return np.multiply(x, (1.0 - x))

    # @staticmethod
    # def __step(x):
    #     result = 0
    #     if x >= 0:
    #         result = 1
    #     return result
    #
    # @staticmethod
    # def __step_update(x):
    #     return 0

    @staticmethod
    def __ReLU(x):
        return np.maximum(x, 0.0)

    @staticmethod
    def __ReLU_update(x):
        return 1.0 * (x > 0)

    @staticmethod
    def __hyperbolic_tangent(x):
        return np.tanh(x)

    @staticmethod
    def __hyperbolic_tangent_update(x):
        return 1 - x ** 2

    def predict(self, data_array):
        """
        Calculate the output for all samples

        :param data_array: input data array, (sample_num, in_node) like array
        :return: y: the output of neural network given the input data array, (sample_num, out_node) like array
        """

        sample_num = data_array.shape[0]
        hidden_array = np.zeros([sample_num, self.hid_node])
        out_array = np.zeros([sample_num, self.out_node])
        for i in range(0, sample_num):
            hidden_array[i] = self.forward_feed(data_array[i], self.weight_in)
            out_array[i] = self.forward_feed(hidden_array[i], self.weight_out)
        return out_array

    @staticmethod
    def transform_classify(out_array):
        """
        Transform the output array into specific class

        :param out_array: the output of neural network
        :return: the number of the specific class
        """

        if out_array.shape[1] > 1:
            labels = np.argmax(out_array, axis=1).astype(int)
        else:
            labels = out_array
            labels[labels >= 0.5] = 1
            labels[labels < 0.5] = 0
        return labels

    def predict_result(self, data_array):
        """
        Predict the class based on input samples

        :param data_array: input data array, (sample_num, in_node) like array
        :return: the prediction results for samples
        """
        return self.transform_classify(self.predict(data_array))

    def forward_feed(self, x, weight):
        """
        Forward calculate one step forward feed output of one input

        :param x: input data array from input layer or hidden layer
        :param weight: current weights
        :return: output in hidden layer or output layer
        """

        x_with_bias = np.hstack((x, np.ones(1)))
        forward_out = self.activation(np.dot(x_with_bias, weight))

        return forward_out

    def fit_model(self, data_array, target, learning_epochs=10, learning_rate=0.2, re_time=10,
                  pruning=False, pruning_threshold=0.01, pruning_norm='L1', pruning_epoch=5,
                  verbose=True):

        """
        Fit the model according to the given training data

        :param data_array: input training samples
        :param target: the true target of training samples
        :param learning_epochs: times of training the data
        :param learning_rate: step size for update
        :param re_time: times of repeating calculating hidden layer
        :param pruning: pruning neurons in hidden layer or not
        :param pruning_threshold: the threshold in pruning rule (pruning=True)
        :param pruning_norm: pruning rule (pruning=True)
        :param pruning_epoch: begin pruning after pruning epoch (pruning=True)
        :param verbose: enable verbose output
        :return: the updated weights
        """
        recurrent_time = re_time

        # multi-classification
        if self.out_node > 1:
            target_array = np.zeros((target.shape[0], self.out_node))
            for i in range(target.shape[0]):
                target_array[i, int(target[i])] = 1
            true_target = target_array

        # bi-classification
        else:
            true_target = target

        if pruning:
            if pruning_epoch > learning_epochs:
                print('pruning epoch is greater than learning epoch')
            else:
                start_time = time.time()
                # before pruning
                for i in range(pruning_epoch):
                    self.weight_update(data_array, true_target, learning_rate, recurrent_time, pruning=False)
                    if verbose:
                        print('Epoch: ', i+1, 'Accuracy: ', self.accuracy(data_array, target),
                              'Elapse time[s]: ', time.time()-start_time)

                # begin pruning at pruning_epoch
                self.weight_out, mask = self.neuron_pruning(self.weight_out, pruning_threshold, norm=pruning_norm)
                self.weight_in = self.weight_in * mask[:-1]

                # pruning in the epoch afterwards
                for i in range(pruning_epoch, learning_epochs):
                    self.weight_update(data_array, true_target, learning_rate,
                                         pruning=True, mask=mask)
                    self.weight_out, mask = self.neuron_pruning(self.weight_out, pruning_threshold, norm=pruning_norm)
                    self.weight_in = self.weight_in * mask[:-1]
                    if verbose:
                        print('Epoch: ', i+1, 'Accuracy: ', self.accuracy(data_array, target),
                              'Elapse time[s]: ', time.time()-start_time)

        else:
            start_time = time.time()
            for i in range(learning_epochs):
                self.weight_update(data_array, true_target, learning_rate, pruning=False)
                if verbose:
                    print('Epoch: ', i+1, 'Accuracy: ', self.accuracy(data_array, target),
                          'Elapse time[s]: ', time.time()-start_time)

        return self.weight_in, self.weight_out

    def weight_update(self, data_array, true_target, learning_rate, recurrent_time=10,
                      pruning=False, mask=None):
        """
        Update weights given samples

        :param data_array: input training samples
        :param true_target: the true target of training samples
        :param learning_rate: step size for update
        :param recurrent_time: times of repeating calculating hidden layer
        :param pruning: pruning neurons in hidden layer or not
        :param mask: the pruning matrix
        """
        loop_time = recurrent_time

        hidden_layer_weight = np.random.randn(self.hid_node, self.hid_node)
        for k in range(data_array.shape[0]):
            # initiate outputs in hidden_layer and output_layer
            hidden_layer_out = np.zeros(self.hid_node)

            # loop to calculate hidden_layer
            hidden_layer_loop_result = np.zeros((loop_time, self.hid_node))
            data_with_bias = np.hstack((data_array[k], np.ones(1)))
            for t in range(loop_time):
                hidden_layer_out = self.activation(np.dot(data_with_bias, self.weight_in) +
                                                   np.dot(hidden_layer_out, hidden_layer_weight))
                hidden_layer_loop_result[t] = hidden_layer_out

            # calculate output layer
            hidden_layer_out_with_bias = np.hstack((hidden_layer_out, np.ones(1)))
            if pruning:
                hidden_layer_out_with_bias = hidden_layer_out_with_bias * mask
            output_layer_out = self.activation(np.dot(hidden_layer_out_with_bias, self.weight_out))

            # update the weights from hidden layer to output layer
            weight_out_error_signal = self.activation_derivative(output_layer_out) * (true_target[k] - output_layer_out)

            weight_out_update = learning_rate * np.outer(hidden_layer_out_with_bias, weight_out_error_signal)
            if pruning:
                weight_out_update = weight_out_update * mask.reshape(mask.shape[0], 1)

            # update the weights from input layer to hidden layer
            cumulative_error = np.array([sum(self.weight_out[h, :] * weight_out_error_signal) for h in range(self.weight_out.shape[0])])
            hidden_layer_derivative = np.zeros(self.hid_node) + data_with_bias.reshape(data_with_bias.shape[0], 1)
            hidden_layer_derivative = hidden_layer_derivative * self.activation_derivative(hidden_layer_loop_result[0])
            for t in range(1, loop_time):
                hidden_layer_derivative = np.dot((hidden_layer_derivative + data_with_bias.reshape(data_with_bias.shape[0], 1)), hidden_layer_weight)
                hidden_layer_derivative = hidden_layer_derivative * self.activation_derivative(hidden_layer_loop_result[t])

            weight_in_update = learning_rate * (hidden_layer_derivative * cumulative_error[:-1])

            # update the weights from hidden layer to hidden layer
            hidden_layer_derivative_hid = np.outer(np.zeros(self.hid_node), hidden_layer_loop_result[0])
            for t in range(1, loop_time-1):
                hidden_layer_derivative_hid = hidden_layer_derivative_hid + hidden_layer_loop_result[t].reshape(hidden_layer_loop_result[t].shape[0], 1)
                hidden_layer_derivative_hid = hidden_layer_derivative_hid * self.activation_derivative(hidden_layer_loop_result[t+1])
            for index in range(hidden_layer_derivative_hid.shape[0]):
                hidden_layer_derivative_hid[index, index] = 0

            hidden_layer_weight_update = learning_rate * (hidden_layer_derivative_hid * cumulative_error[:-1])

            if pruning:
                weight_in_update = weight_in_update * mask[:-1]
                hidden_layer_weight_update = hidden_layer_weight_update * mask[:-1]

            hidden_layer_weight += hidden_layer_weight_update
            self.weight_out += weight_out_update
            self.weight_in += weight_in_update

    @staticmethod
    def neuron_pruning(weight_out_matrix, threshold, norm, verbose=True):
        """
        Prune the neurons in hidden layer if pruning rule is satisfied

        :param weight_out_matrix: weights connecting hidden layer to output layer
        :param threshold: pruning threshold
        :param norm: L1 or L2 norm
        :param verbose: enable verbose output
        :return: weight_out_matrix: weights connecting hidden layer to output layer after pruning process
                 mask: pruning matrix
        """
        mask = np.ones(weight_out_matrix.shape[0])
        if norm == 'L1':
            norm = np.linalg.norm(weight_out_matrix, 1, axis=1)
        if norm == 'L2':
            norm = np.linalg.norm(weight_out_matrix, 2, axis=1)
        if np.min(norm) < threshold:
            index = np.where(norm < threshold)[0]
            mask[list(index)] = 0
            weight_out_matrix = weight_out_matrix * mask.reshape(mask.shape[0], 1)
            if verbose:
                print('pruning node #', list(index+1))
        return weight_out_matrix, mask

    def accuracy(self, data_array, target):
        """
        Calculate accuracy of prediction

        :param data_array: test samples
        :param target: true target of test samples
        :return: accuracy of prediction
        """
        count = 0
        predict_label = self.predict_result(data_array)
        for i in range(predict_label.shape[0]):
            if predict_label[i] == target[i]:
                count += 1
        return count/predict_label.shape[0]
