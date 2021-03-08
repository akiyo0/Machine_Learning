import os
import kNN
import numpy as np

def get_np_image(image_path):
    image_output = []
    raw_image = open(image_path)

    for j in range(32):
        image_output.append([int(item) for item in raw_image.readline().replace("\n", "")])
        
    return np.array(image_output)

def handwriting():
    dataset_path = '../../../datasets/Digits/'
    training_list = os.listdir(os.path.join(dataset_path, "trainingDigits"))
    test_list = os.listdir(os.path.join(dataset_path, "testDigits"))

    train_len = len(training_list)
    test_len = len(test_list)

    trainingMat = np.zeros((train_len, 1024))

    train_labels = []
    test_labels = []

    for i in range(train_len):
        splited_path = training_list[i].split('.')[0]
        label = int(splited_path.split('_')[0])
        train_labels.append(label)
        
        train_path = os.path.join(dataset_path, "trainingDigits", training_list[i])
        trainingMat[i, :] = get_np_image(train_path).reshape(-1)

    for i in range(test_len):
        splited_path = test_list[i].split('.')[0]
        label = int(splited_path.split('_')[0])
        test_labels.append(label)

        test_path = os.path.join(dataset_path, "testDigits", test_list[i])
        #print(len(img2vector(test_path)[0]))
        
        testVector = get_np_image(test_path).reshape(-1)
        label_pre = kNN.kNN(trainingMat, testVector, 3, train_labels)
        print(label_pre, test_labels[i])

    print(train_labels)
    print(test_labels)
        

handwriting()