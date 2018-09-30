import tensorflow as tf
import numpy as np
import csv
import sys
import os
from pathlib import Path

# About save path
home = str(Path.home())
csv_folder_name = os.path.join(home, "Activity_Source/Data")
test_file_name = "testdata"

model_folder_name = os.path.join(home, "Activity_Source/Model/model")
model_file_name = "ActivityRNN"

model_info_file_name = os.path.join(home, "Activity_Source/Model/model_info.csv")
model_info_type = ["ModelIndex", "DataIndex", "Test Accuracy", "DataNum", "Timestep", "Batch Size", "Epoch", "Lstm Size", "Dense Size", "Model Layer"]

predict_info_file_name = os.path.join(home, "Activity_Source/Model/predict_info")

class_type = ["Biking", "In Vehicle", "Running", "Still", "Tilting", "Walking"] # 0,1,2,3,4,5
# About ML
num_labels = 6
num_channels = 6

def read_data():
    global csv_folder_name
    global test_file_name

    test_path = os.path.join(csv_folder_name, test_file_name + '.csv')
    test_data = list(csv.reader(open(test_path,'r')))
    test_data = np.array(test_data[1:]).astype(float)

    # act = test_data[:,0].astype(int)
    # one_hot = np.zeros((act.size, len(class_type)))
    # one_hot[np.arange(act.size), act] = 1

    # test_data = np.hstack((one_hot, test_data[:,1:]))
    print (test_data.shape)

    return test_data

def label_rawData(npdata, input_size):

    new_npdata = []
    if npdata.size != 0:
        idx_base = npdata[0, 0]

        data_num = npdata.shape[0]
        same_count = 0
        for i in range(0, data_num):
            idx_temp = npdata[i, 0]
            if idx_temp == idx_base:
                same_count += 1;
                if same_count == input_size:
                    temp_data_last = npdata[i+1-input_size:i+1, 1:-1].flatten()
                    temp_data_last = np.hstack((idx_temp, temp_data_last))
                    new_npdata.append(temp_data_last)
                    same_count = 0;
            else:
                # if i > input_size:
                #     temp_data_last = npdata[i-input_size:i, num_labels:-1].flatten()
                #     temp_data_last = np.hstack((activity_base, temp_data_last))
                #     new_npdata.append(temp_data_last)
                if i+input_size <= data_num:
                    temp_data_next = npdata[i:i+input_size, 1:-1].flatten()
                    temp_data_next = np.hstack((idx_temp, temp_data_next))
                    new_npdata.append(temp_data_next)

                idx_base = idx_temp
                same_count = 0;

            # if (i+1)%input_size == 0:
            #     temp_data_last = npdata[i+1-input_size:i+1, num_labels:-1].flatten()
            #     temp_data_last = np.hstack((activity_temp, temp_data_last))
            #     new_npdata.append(temp_data_last)

            # if idx_temp != idx_base:
            #     if i%input_size != 0:
            #         if i > input_size:
            #             temp_data_last = npdata[i-input_size:i, num_labels:-1].flatten()
            #             temp_data_last = np.hstack((activity_base, temp_data_last))
            #             new_npdata.append(temp_data_last)
            #         if i+input_size <= data_num:
            #             temp_data_next = npdata[i:i+input_size, num_labels:-1].flatten()
            #             temp_data_next = np.hstack((activity_temp, temp_data_next))
            #             new_npdata.append(temp_data_next)

            #     idx_base = idx_temp
            #     activity_base = activity_temp

    new_npdata = np.array(new_npdata)
    print (new_npdata.shape)
    return new_npdata

def getXY(all_dataset):
    all_dataset = all_dataset.astype(float)

    dataNum = all_dataset.shape[0]
    permutation = np.random.permutation(all_dataset.shape[0])
    all_dataset = all_dataset[permutation]

    test_X = all_dataset[:,1:]
    test_label = all_dataset[:,0]

    return test_X, test_label

def read_info():
    global model_info_file_name

    info = list(csv.reader(open(model_info_file_name,'r')))
    info = np.array(info[1:])

    return info

def write_result(model_index, test_accuracy, predict_percent_array, info_item):
    global class_type

    predict_info_file_path = predict_info_file_name + str(model_index) + '.csv'
    predict_info_file = open(predict_info_file_path, "w")
    predict_info_w = csv.writer(predict_info_file)

    predict_info_w.writerow(model_info_type)
    predict_info_w.writerow(info_item)
    predict_info_w.writerow(["Test Accuracy"])
    predict_info_w.writerow([test_accuracy])
    predict_info_w.writerow(["Predict Array"])
    predict_info_w.writerow(["Array"] + class_type)

    for i in range(num_labels):
        row = np.hstack((np.array([class_type[i]]), predict_percent_array[i].astype(str)))
        predict_info_w.writerow(row)


info = read_info()

for info_item in info:
    input_size = int(info_item[5])
    model_index = int(info_item[1])
    pb_path = os.path.join(model_folder_name + str(model_index), model_file_name + str(model_index) + '.pb')

    test_data = read_data()
    labeled_data = label_rawData(test_data, input_size)
    test_X, test_label = getXY(labeled_data)
    test_data_num = test_label.shape[0]

    one_hot_label = np.zeros((test_label.size, num_labels))
    one_hot_label[np.arange(test_label.size), test_label.astype(int)] = 1
    print (class_type)
    print (np.sum(one_hot_label, axis=0))

    tf.reset_default_graph()
    graph_def = tf.GraphDef()
    # Import the TF graph
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    output_layer = 'prediction:0'
    input_node = 'input_x:0'
    predictions = []

    with tf.Session() as sess:
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        predictions = sess.run(prob_tensor, {input_node: test_X})

    predictions = np.array(predictions)
    highest_probability_index = np.argmax(predictions, axis=1)

    correct_num = 0
    predict_array = np.zeros((num_labels, num_labels))
    for i in range(test_data_num):
        if highest_probability_index[i] == test_label[i]:
            correct_num += 1
        predict_array[int(test_label[i]), int(highest_probability_index[i])] += 1
    test_accuracy = correct_num / test_data_num

    predict_percent_array = np.zeros((num_labels, num_labels))
    for i in range(predict_array.shape[0]):
        row_sum = np.sum(predict_array[i])
        if row_sum > 0:
            predict_percent_array[i] = predict_array[i] / row_sum

    write_result(model_index, test_accuracy, predict_percent_array, info_item)

    print (test_accuracy)
    print (predict_array)
    print (predict_percent_array)
