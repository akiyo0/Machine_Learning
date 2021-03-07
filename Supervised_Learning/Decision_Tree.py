from math import log

def entropy(dataset):
    num = len(dataset)
    label = {}
    for i in dataset:
        this_label = i[-1]
        if this_label not in label.keys():
            label[this_label] = 0
        label[this_label] += 1
    
    entropy_value = 0.
    for key in label:
        p = float(label[key]) / num
        entropy_value -= p * log(p, 2)
    
    return entropy_value

dataset = [
    [1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']
]
labels = ['no surfacing', 'flippers']

dataset_2 = [
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1],
    [0, 2, 1, 1, 0],
    [0, 2, 1, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 2, 1, 0, 1],
    [2, 0, 0, 0, 1],
    [2, 1, 0, 1, 1],
    [2, 0, 1, 0, 1],
    [2, 2, 1, 1, 1]
]

labels = ['weather', 'temperature', 'humidity', 'wind']

labels_name = [
    ['雨', '晴', '曇'],
    ['暑', '暖', '涼'],
    ['高', '普通'],
    ['有', '無'],
    ['×', '○']
]

print(entropy(dataset_2))

def split_by_value(dataset, idx, value):
    output = []
    for i in dataset:
        if i[idx] == value:
            reduced_i = i[:idx]
            reduced_i.extend(i[idx+1:])
            output.append(reduced_i)
    return output

print(split_by_value(dataset_2, 0, 1))

def select_best_features(dataset):
    features_num = len(dataset[0]) -1
    base_entropy = entropy(dataset)
    best_gain = 0.; best_features = -1

    for i in range(features_num):
        feat_list = [data[i] for data in dataset] #候选features
        unique_vals = set(feat_list) #该features下所有的值
        new_entoropy = 0.0

        for value in unique_vals:
            sub_dataset = split_by_value(dataset, i, value)
            p = len(sub_dataset) / float(len(dataset))
            new_entoropy += p * entropy(sub_dataset)
        
        gain = base_entropy - new_entoropy

        if(gain > best_gain):
            best_gain = gain
            best_features = i
        
    return best_features

print(select_best_features(dataset_2))

def get_major_label(label_list):
    label_count = {}
    for i in label_list:
        if i not in label_count.keys():
            label_count[i] = 0
        label_count[i] += 1
    
    sorted_label_count = sorted(label_count.items(), key = lambda x: x[1], reverse=True)
    return sorted_label_count[0][0]

def build_tree(dataset, labels):
    label_list = [value[-1] for value in dataset]
    #终止条件1: 所有的类标签完全相同，则直接返回该类标签。
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    #终止条件2: 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataset[0]) == 1:
        return get_major_label(label_list)
    
    best_features = select_best_features(dataset)
    best_label = labels[best_features]

    tree = {best_label:{}}
    del(labels[best_features])

    best_features_value = [value[best_features] for value in dataset]
    for value in set(best_features_value):
        sub_labels  = labels[:]
        tree[best_label][value] = build_tree(
            split_by_value(dataset, best_features, value), sub_labels
        )

    return tree

#label_list = [value[-1] for value in dataset_2]
print(build_tree(dataset_2, labels))