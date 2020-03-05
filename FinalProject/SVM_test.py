import pandas as pd
import numpy as np

adult_data = pd.read_csv('adult_clean.csv')
adult_array = np.array(adult_data.iloc[:, 1:-1])
adult_target = np.array(adult_data.iloc[:, -1])

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

from sklearn import svm
import time

# Tuning parameter C
c_list = [0.001, 0.01, 0.1, 1, 10]
time_list = []
train_acc_list = []
acc_list = []
for c in c_list:
    print('c=', c)
    running_time = []
    train_acc = []
    acc = []
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
        
        start_time = time.time()
        clf = svm.SVC(C=c, kernel='rbf', gamma='auto', random_state=100)
        clf.fit(train_X, train_y)
        
        running_time.append(time.time() - start_time)
        train_acc.append(clf.score(train_X, train_y))
        acc.append(clf.score(test_X, test_y))
    
    print('average time:', np.mean(running_time))
    print('average train accuracy:', np.mean(train_acc))
    print('average test accuracy:', np.mean(acc))
    time_list.append(running_time)
    train_acc_list.append(train_acc)
    acc_list.append(acc)
    
print(time_list)
print(acc_list)
        

# # Tuning parameter gamma
# gamma_list = [0.001, 0.01, 0.1, 1, 10]
# time_list = []
# train_acc_list = []
# acc_list = []
# for gamma in gamma_list:
#     print('gamma=', gamma)
#     running_time = []
#     train_acc = []
#     acc = []
#     for i in range(len(fold_groups)):
#         print("Round: " + str(i+1))
#
#         # Select test set and train data
#         test_group_idx = i
#         test_X, test_y = fold_groups[i]
#
#         train_group_idx = list(five_sets - {i})
#         train_X = [fold_groups[idx][0] for idx in train_group_idx]
#         train_y = [fold_groups[idx][1] for idx in train_group_idx]
#
#         train_X = np.concatenate(train_X)
#         train_y = np.concatenate(train_y)
#
#         start_time = time.time()
#         clf = svm.SVC(C=1, kernel='rbf', gamma=gamma, random_state=100)
#         clf.fit(train_X, train_y)
#
#         running_time.append(time.time() - start_time)
#         train_acc.append(clf.score(train_X, train_y))
#         acc.append(clf.score(test_X, test_y))
#
#     print('average time:', np.mean(running_time))
#     print('average train accuracy:', np.mean(train_acc))
#     print('average test accuracy:', np.mean(acc))
#     time_list.append(running_time)
#     train_acc_list.append(train_acc)
#     acc_list.append(acc)
#
# print(time_list)
# print(acc_list)
#



