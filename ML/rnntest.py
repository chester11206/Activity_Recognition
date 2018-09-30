import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np
import csv
import sys
import os
from operator import itemgetter
from pathlib import Path
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from google.protobuf import text_format

# About save path
home = str(Path.home())
csv_folder_name = os.path.join(home, "Activity_Source/Data")
csv_file_name = "rawData"
data_file_name = "data"
data_index = 0
model_index = 0

model_folder_name = os.path.join(home, "Activity_Source/Model/model")
model_file_name = "ActivityRNN"

model_info_file_name = os.path.join(home, "Activity_Source/Model/model_info.csv")
model_info_type = ["Model Name", "Model Index", "Data Index", "Test Accuracy", "Data Number", "Timestep", "Batch Size", "Epoch", "Lstm Size", "Dense Size", "Model Layer"]
layer_info = "RNN"
class_type = ["Biking", "In Vehicle", "Running", "Still", "Tilting", "Walking"]

# About ML
DATA_NUM = 0
input_height = 1
input_width = 800
input_size = input_height * input_width
num_channels = 6
num_labels = 6

lstm_size = [256,256]
dense_size = [512,512]

learning_rate = 0.0001
epoch_num = 2000
stop_epoch = 0
epoch_start = 0
iters = 0
batch_size = 128
perm_X = np.empty
perm_Y = np.empty

# Training Data / Raw Data
train_p = 0.9


################
# **Function

# Data Process
def read_data():
    global csv_folder_name
    global data_file_name

    data_index = 2
    data_path = os.path.join(csv_folder_name, csv_file_name + str(data_index) + '.csv')
    new_data = list(csv.reader(open(data_path,'r')))
    new_data = np.array(new_data[1:]).astype(float)

    # act = new_data[:,0].astype(int)
    # one_hot = np.zeros((act.size, num_labels))
    # one_hot[np.arange(act.size), act] = 1

    # new_data = np.hstack((one_hot, new_data[:,1:]))
    print (new_data.shape)

    return new_data

def write_data():
    global csv_folder_name
    global csv_file_name
    global data_file_name
    global data_index

    csv_path_name = os.path.join(csv_folder_name, csv_file_name)
    data_path_name = os.path.join(csv_folder_name, data_file_name)
    data_path = data_path_name + '.csv'

    # get last data path
    while os.path.exists(csv_path_name + str(data_index) + ".csv"):
        data_index += 1

    if os.path.exists(data_path):
        # read new data
        new_data = list(csv.reader(open(data_path,'r')))
        new_data = np.array(new_data[1:]).astype(float)

        act = new_data[:,0].astype(int)
        one_hot = np.zeros((act.size, num_labels))
        one_hot[np.arange(act.size), act] = 1

        new_data = np.hstack((one_hot, new_data[:,1:]))
        print (new_data.shape)

        #write new data to csv
        if data_index != 0:
            csv_last_path = csv_path_name + str(data_index-1) + ".csv"
            csv_path = csv_path_name + str(data_index) + ".csv"

            # combine new and last data, save to new file
            with open(csv_last_path, 'rt') as infile:
                with open(csv_path, 'wt') as outfile:
                    writer = csv.writer(outfile)
                    reader = csv.reader(infile)
                    writer.writerow(next(reader))
                    for row in reader:
                        writer.writerow(row)
                    for item in new_data:
                        writer.writerow(item)
                        
            #os.remove(data_path)
        else:
            csv_path = csv_path_name + str(data_index) + ".csv"
            with open(data_path, 'rt') as infile:
                with open(csv_path, 'wt') as outfile:
                    writer = csv.writer(outfile)
                    reader = csv.reader(infile)
                    for item in new_data:
                        writer.writerow(item)
        
        rename_data_path = data_path_name + str(data_index) + '.csv'
        os.rename(data_path,rename_data_path)
    else:
        data_index -= 1
        csv_path = csv_path_name + str(data_index) + ".csv"

    # read data from new file
    all_data = list(csv.reader(open(csv_path,'r')))
    all_data = np.array(all_data)

    return all_data

def label_rawData(npdata):

    new_npdata = []
    if npdata.size != 0:
        activity_base = npdata[0, :num_labels]
        idx_base = 0
        for act in activity_base:
            if act == 1:
                break
            else:
                idx_base+=1

        data_num = npdata.shape[0]
        same_count = 0
        for i in range(0, data_num):
            activity_temp = npdata[i, :num_labels]
            idx_temp = 0
            for act in activity_temp:
                if act == 1:
                    break
                else:
                    idx_temp+=1
            if idx_temp == idx_base:
                same_count += 1;
                if same_count == input_size:
                    temp_data_last = npdata[i+1-input_size:i+1, num_labels:-1].flatten()
                    temp_data_last = np.hstack((activity_temp, temp_data_last))
                    new_npdata.append(temp_data_last)
                    same_count = 0;
            else:
                # if i > input_size:
                #     temp_data_last = npdata[i-input_size:i, num_labels:-1].flatten()
                #     temp_data_last = np.hstack((activity_base, temp_data_last))
                #     new_npdata.append(temp_data_last)
                if i+input_size <= data_num:
                    temp_data_next = npdata[i:i+input_size, num_labels:-1].flatten()
                    temp_data_next = np.hstack((activity_temp, temp_data_next))
                    new_npdata.append(temp_data_next)

                idx_base = idx_temp
                activity_base = activity_temp
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
    print (class_type)
    print (np.sum(all_dataset[:,:num_labels], axis=0))

    dataNum = all_dataset.shape[0]
    permutation = np.random.permutation(all_dataset.shape[0])
    all_dataset = all_dataset[permutation]

    trainNum = int(dataNum*train_p)
    trainY = all_dataset[:trainNum,:num_labels]
    testY = all_dataset[trainNum:,:num_labels]
    allX = all_dataset[:,num_labels:]

    return allX, trainNum, trainY, testY

def feature_normalize(dataset):

    for i in range(0, num_channels):
        mu = np.mean(dataset[:,range(i,dataset.shape[1],num_channels)].flatten(),axis = 0)
        sigma = np.std(dataset[:,range(i,dataset.shape[1],num_channels)].flatten(),axis = 0)
        for j in range(i,dataset.shape[1],num_channels):
            temp = (dataset[:,j] - mu)/sigma
            dataset[:,j] = temp

    return dataset

# Run Batch
def next_batch(X_train, Y_train, num, start):
    global perm_X
    global perm_Y
    global iters
    if start == 0:
        perm = np.random.permutation(X_train.shape[0])
        perm_X = X_train[perm]
        perm_Y = Y_train[perm]
        batch_X = perm_X[start:start + num]
        batch_Y = perm_Y[start:start + num]
        start += num
    elif start <= X_train.shape[0] - num:
        batch_X = perm_X[start:start + num]
        batch_Y = perm_Y[start:start + num]
        start += num
    else:
        rest_num = X_train.shape[0] - start
        new_part_num = num - rest_num
        batch_X = np.vstack((perm_X[start:], perm_X[:new_part_num]))
        batch_Y = np.vstack((perm_Y[start:], perm_Y[:new_part_num]))
        perm = np.random.permutation(X_train.shape[0])
        perm_X = X_train[perm]
        perm_Y = Y_train[perm]
        start = 0
        iters += 1

    return batch_X, batch_Y, start

def canonical_name(x):
    return x.name.split(":")[0]


# Save Result
def write_model(sess, frozen_graphdef):
    global model_folder_name
    global model_file_name
    global model_index

    # **Models
    # make new models folder
    while os.path.exists(model_folder_name + str(model_index)):
        model_index += 1
    model_folder_path = model_folder_name + str(model_index)
    os.makedirs(model_folder_path)

    model_path = os.path.join(model_folder_path, model_file_name)
    ckpt_path = model_path + str(model_index) + ".ckpt"

    # save model
    # ckpt
    saver = tf.train.Saver()
    saver.save(sess, ckpt_path)
    sess.close()
    # pb
    # tf.train.write_graph(frozen_graphdef, model_folder_name + str(i),
    #                  model_file_name + str(i) + '.pb', as_text=False)
    tf.train.write_graph(sess.graph_def, model_folder_path,
                     model_file_name + str(model_index) + '.pbtxt')

    pb_path = model_path + str(model_index) + '.pb'
    pbtxt_path = model_path + str(model_index) + '.pbtxt'
    opt_pb_path = model_path + "opt" + str(model_index) + ".pb"

    freeze_graph.freeze_graph(input_graph = pbtxt_path,  input_saver = "",
             input_binary = False, input_checkpoint = ckpt_path, output_node_names = "prediction",
             restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0",
             output_graph = pb_path, clear_devices = True, initializer_nodes = "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(pb_path, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    # with tf.gfile.FastGFile(pbtxt_path, "rb") as f:
    #     text_format.Merge(f.read(), input_graph_def)

    # for node in input_graph_def.node:
    #     if node.op == "Switch":
    #         node.op = "Identity"
    #         # del node.input[1]
    # print ("OK")

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            ["input_x"], 
            ["prediction"],
            tf.float32.as_datatype_enum)

    f = tf.gfile.FastGFile(opt_pb_path, "w")
    f.write(output_graph_def.SerializeToString())

    return pb_path

def write_result(test_accuracy):
    global model_folder_name
    global model_file_name
    global model_info_file_name
    global model_index
    global data_index
    global batch_size
    global epoch_num
    global layer_info

    # **Write model information to csv
    minfo = [model_file_name, model_index, data_index, test_accuracy, DATA_NUM, input_width, batch_size, stop_epoch, lstm_size, dense_size]
    print (minfo)
    if not os.path.exists(model_info_file_name):
        model_info_file = open(model_info_file_name, "w")
        model_info_w = csv.writer(model_info_file)
        model_info_w.writerow(model_info_type)
        model_info_w.writerow(minfo)
    else:
        model_info_file = open(model_info_file_name, "a")
        model_info_w = csv.writer(model_info_file)
        model_info_w.writerow(minfo)

# ML layer

def RNN_op(input_op, keep_prob):
    mlstm_cell = []
    for i in range(len(lstm_size)):
        #lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size[i], forget_bias=1.0, state_is_tuple=True, name='basic_lstm_cell')
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
        mlstm_cell.append(lstm_cell)
    mlstm_cell = tf.contrib.rnn.MultiRNNCell(mlstm_cell,state_is_tuple=True)

    # **Step3: Initiate state with zero
    init_state = mlstm_cell.zero_state(tf.shape(input_op)[0], dtype=tf.float32)

    # **Step4: Calculate in timeStep
    outputs = list()
    state = init_state
    with tf.variable_scope('RNN'):
        input_shape = input_op.get_shape().as_list()
        #timestep_X = inference_op(input_op,keep_prob)
        timestep_X = input_op
        shp = timestep_X.get_shape()

        # timestep_X = tf.unstack(timestep_X,axis=2)
        outputs, state = tf.nn.dynamic_rnn(cell=mlstm_cell,
                                   inputs=timestep_X,
                                   dtype=tf.float32)
        # outputs = tf.transpose(outputs, [1, 0, 2])[-1]
        

    #     for timestep in range(shp[2]):
    #         if timestep > 0:
    #             tf.get_variable_scope().reuse_variables()
    #         # Variable "state" store the LSTM state
    #         # timestep_X = tf.reshape(input_op[:,timestep,:,:], [-1,1,input_width,num_channels])
    #         # timestep_cell = inference_op(timestep_X,keep_prob)
    #         # timestep_cell = timestep_X[:,:,timestep,:]
    #         # timestep_cell = tf.reshape(timestep_cell, [-1,timestep_cell.get_shape()[-1]])

    #         (cell_output, state) = mlstm_cell(timestep_X[:,timestep,:], state)
    #         outputs.append(cell_output)

    # outputs = outputs[-1]

    return outputs

#######################
# **Get train&test data

# combine old data and save, shape=[data_num, input_width*num_channels+num_labels]=[-1,2706]
#all_data = read_data()
all_data = write_data()

# label raw data, shape=[data_num, input_width*num_channels+num_labels]=[-1,2706]
labeled_data = label_rawData(all_data)
DATA_NUM = labeled_data.shape[0]

# seperate X, Y
# X: shape=[data_num, input_height*input_width*num_channels]
# trainY: shape=[train_num, num_labels]
# testY: shape=[test_num, num_labels]
new_dataset, trainNum, trainY, testY = getXY(labeled_data)

# feature normalize
# new_dataset = feature_normalize(beforenorm_dataset)

print (new_dataset.shape)

# get trainX, testX
# trainX: shape=[train_num, input_height, input_width, num_channels]
# testX: shape=[test_num, input_height, input_width, num_channels]
trainX = new_dataset[:trainNum]
testX = new_dataset[trainNum:]
print (trainX.shape)
print (trainY.shape)
print (testX.shape)
print (testY.shape)


#############
# **Build CNN

X_ = tf.placeholder(tf.float32, shape=[None,input_height*input_width*num_channels], name="input_x")
X = tf.reshape(X_, [-1,input_width,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels], name="input_y")

keep_prob = tf.placeholder(tf.float32)

# depth_conv = apply_depthwise_conv(X,kernel_size,num_channels,depth)
# max_pool = apply_max_pool(depth_conv,20,2)
# depth_conv = apply_depthwise_conv(max_pool,6,depth*num_channels,depth//10)

# depth_conv = X
# for i in range(len(kernel_size)):
#     channels = num_channels
#     if i > 0:
#         for j in range(i):
#             channels = channels * depth[j]
#     depth_conv = apply_depthwise_conv(depth_conv,kernel_size[i],channels,depth[i])

# shape_conv = depth_conv.get_shape().as_list()
# logits = tf.reshape(depth_conv, [-1, shape_conv[1] * shape_conv[2] * shape_conv[3]])

# for i in range(len(dense_size)):
#     logits = tf.layers.dense(inputs=logits, units=dense_size[i], activation=tf.nn.relu)
h_state = RNN_op(X, keep_prob)
#logits = h_state

shp = h_state.get_shape()
flattened_shape = shp[1].value * shp[2].value
logits = tf.reshape(h_state,[-1,flattened_shape])

for i in range(len(dense_size)):
    logits = tf.layers.dense(inputs=logits, units=dense_size[i], activation=tf.nn.relu)
    if i < len(dense_size)-1: 
        logits = tf.layers.dropout(inputs=logits, rate=keep_prob)

W = tf.Variable(tf.truncated_normal([dense_size[-1], num_labels], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[num_labels]), dtype=tf.float32)
Y_pre = tf.nn.softmax(tf.matmul(logits, W) + bias, name = "prediction")
# Y_pre = tf.nn.softmax(logits, name = "prediction")

# h_state is the output of hidden layer
# Weight, Bias, Softmax to predict
# W = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1), dtype=tf.float32)
# bias = tf.Variable(tf.constant(0.1,shape=[num_labels]), dtype=tf.float32)
# Y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias, name = "prediction")

# logits, p = inference_op(X,keep_prob)

# shape_logits = logits.get_shape().as_list()
# Y_pre = tf.nn.softmax(logits, name = "prediction")

# loss = -tf.reduce_sum(Y * tf.log(Y_pre))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

loss = -tf.reduce_mean(Y * tf.log(Y_pre))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(Y_pre,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#############
# **Run Model

# Set up GPU demand
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True

init = tf.global_variables_initializer()
out = tf.identity(Y_pre, name="output")

with tf.Session(config=config) as sess:
    num99 = 0
    sess.run(init)
    for epoch in range(epoch_num):
        batch_X, batch_Y, epoch_start = next_batch(trainX, trainY, batch_size, epoch_start)
        train_accuracy = sess.run(accuracy, feed_dict={
                X_:batch_X, Y: batch_Y, keep_prob:0.5})
        if train_accuracy > 0.99:
            num99 += 1
            if num99 > 10:
                stop_epoch = epoch
                break
        if (epoch+1)%200 == 0:
            # train_accuracy = sess.run(accuracy, feed_dict={
            #     X_:batch_X, Y: batch_Y, keep_prob:0.5})
            #print ("Iter%d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), accuracy))
            print ("Iter%d, Epoch %d, Training Accuracy %g" % ( iters, (epoch+1), train_accuracy))
        sess.run(optimizer, feed_dict={X_: batch_X, Y: batch_Y, keep_prob:0.5})

    print ("Num 99 %g"% num99)
    print ("Epoch Num %g"% stop_epoch)
    test_accuracy = sess.run(accuracy, feed_dict={
        X_: testX, Y: testY, keep_prob:0.5})
    print ("Test Accuracy %g"% test_accuracy)

    if test_accuracy > 0.75:
        frozen_tensors = [out]
        frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, list(map(canonical_name, frozen_tensors)))
        # save ckpt & pb, return pb path
        frozen_graphdef_path = write_model(sess,frozen_graphdef)

        # converter = tf.contrib.lite.TocoConverter.from_frozen_graph(frozen_graphdef_path, ["input_x"], ["prediction"])
        # tflite_model = converter.convert()
        # write_result(tflite_model, test_accuracy)
        write_result(test_accuracy)

        # converter = tf.contrib.lite.TocoConverter.from_frozen_graph(frozen_graphdef, ["input_x"], ["prediction"])
        # tflite_model = converter.convert()
        # write_tflite(tflite_model)
    else:
        print ("Not Fit!")
