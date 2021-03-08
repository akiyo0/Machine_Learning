import numpy as np

def kNN(dataset, X, k, labels):
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(X, (dataset_size, 1)) - dataset
    distances = np.sqrt((diff_mat ** 2).sum(axis=1))
    sorted_dist_index = distances.argsort()

    label_count = {}
    for i in range(k):
        knn_label = labels[sorted_dist_index[i]]
        label_count[knn_label] = label_count.get(knn_label, 0) + 1

    sorted_label_count = sorted(label_count.items(), key = lambda x: x[1], reverse=True)
    return sorted_label_count[0][0]

'''
datasets = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
labels = ['A', 'A', 'B', 'B']
#print(kNN(datasets, [0, 0], 3, labels))

datasets = [
    [3, 104], # California Man
    [2, 100], # He’s Not Reall...
    [1, 81], # Beautiful Woman
    [101, 10], #Kevin Longblade
    [99, 5], #Robo Slayer 3000
    [98, 2], #Amped II
]
labels = ['爱情片', '爱情片', '爱情片', '动作片', '动作片', '动作片']
X = [18,  90]

print(kNN(np.array(datasets), X, 3, labels))
'''