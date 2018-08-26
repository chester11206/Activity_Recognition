# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import csv
import sys
import os
from operator import itemgetter
from pathlib import Path
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# About firebase
home = str(Path.home())

cred = credentials.Certificate('firebase-adminsdk.json')
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://total-cascade-210406.firebaseio.com'
})

# About save path
csv_folder_name = os.path.join(home, "Activity_Source/Data")
csv_file_name = "rawData"

model_folder_name = os.path.join(home, "Activity_Source/Model/model")
model_file_name = "ActivityRNN"

model_info_file_name = "model_info.csv"
model_info_type = ["Index", "Test Accuracy", "Batch Size", "Epoch", "Model Layer"]
layer_info = "(lstm + dropout) * 2"

# About ML
# Feature Num #28
input_size = 6
# Time Series Num #28
timestep_size = 450
# Hidden Layer Feature Num 
hidden_size = 256
# LSTM Layer Num
layer_num = 2
# Y Class Num #10
class_num = 6
class_type = ["Biking", "In Vehicle", "Running", "Still", "Tilting", "Walking", "Features"]

lr = 1e-3
epoch_start = 0
epoch_num = 500
iters = 0
_batch_size = 128
perm_X = np.empty
perm_Y = np.empty

# Training Data / Raw Data
train_p = 0.8


################
# **Function

# Data Process
def get_firebase():

    # connect firebase
    root = db.reference()
    values = root.child('SensorDataSet').get()
    data = pd.DataFrame(values).T
    
    data_num = data.shape[0] - (data.shape[0] % timestep_size)
    npdata = np.array(data.values)
    npdata = npdata[:data_num,:]

    return npdata

def label_rawData(npdata):

    new_npdata = []
    if npdata.size != 0:
        activity_base = npdata[0, :class_num]
        idx_base = 0
        for i in activity_base:
            if int(i) == 1:
                break
            else:
                idx_base+=1

        for i in range(0, npdata.shape[0]):
            activity_temp = npdata[i, :class_num]
            idx_temp = 0
            for act in activity_temp:
                if int(act) == 1:
                    break
                else:
                    idx_temp+=1

            if (i+1)%timestep_size == 0:
                temp_data_last = npdata[i+1-timestep_size:i+1, class_num:-1].flatten()
                temp_data_last = np.hstack((activity_temp, temp_data_last))
                new_npdata.append(temp_data_last)

            if idx_temp != idx_base:
                if i%timestep_size != 0:
                    temp_data_last = npdata[i-timestep_size:i, class_num:-1].flatten()
                    temp_data_last = np.hstack((activity_base, temp_data_last))
                    new_npdata.append(temp_data_last)
                if i+timestep_size <= data_num:
                    temp_data_next = npdata[i:i+timestep_size, class_num:-1].flatten()
                    temp_data_next = np.hstack((activity_temp, temp_data_next))
                    new_npdata.append(temp_data_next)

                idx_base = idx_temp
                activity_base = activity_temp

    new_npdata = np.array(new_npdata)
    return new_npdata

def write_data(raw_data):
    global csv_folder_name
    global csv_file_name

    # make new csv folder
    if not os.path.exists(csv_folder_name):
        os.makedirs(csv_folder_name)

    csv_path = os.path.join(csv_folder_name, csv_file_name)
    i = 0
    while os.path.exists(csv_path + str(i) + ".csv"):
        i += 1

    #write raw data to csv
    if i != 0:
        csv_last_path = csv_path + str(i-1) + ".csv"
        csv_path = csv_path + str(i) + ".csv"

        # combine with old data and save to new file
        with open(csv_last_path, 'rt') as infile:
            with open(csv_path, 'wt') as outfile:
                writer = csv.writer(outfile)
                reader = csv.reader(infile)
                writer.writerow(next(reader))
                for row in reader:
                    writer.writerow(row)
                for item in raw_data:
                    writer.writerow(item)
    else:
        csv_path = csv_path + str(i) + ".csv"
        csv_file = open(csv_path,"w")
        csv_w = csv.writer(csv_file)
        csv_w.writerow(class_type)
        for item in raw_data:
            csv_w.writerow(item)
        csv_file.close()

    # read data from new file
    new_data = list(csv.reader(open(csv_path,'r')))
    new_data = np.array(new_data[1:]).astype(float)

    # clean firebase
    # root = db.reference()
    # root.child('SensorDataSet').delete()

    return new_data

# Run Batch
def next_batch(X_train, Y_train, num, start):
    global perm_X
    global perm_Y
    global iters
    if start == 0:
        perm = np.random.permutation(X_train.shape[0])
        perm_X = X_train[perm, :]
        perm_Y = Y_train[perm, :]
        batch_X = perm_X[start:start + num, :]
        batch_Y = perm_Y[start:start + num, :]
        start += num
    elif start <= X_train.shape[0] - num:
        batch_X = perm_X[start:start + num, :]
        batch_Y = perm_Y[start:start + num, :]
        start += num
    else:
        rest_num = X_train.shape[0] - start
        new_part_num = num - rest_num
        batch_X = np.vstack((perm_X[start:, :], perm_X[:new_part_num, :]))
        batch_Y = np.vstack((perm_Y[start:, :], perm_Y[:new_part_num, :]))
        perm = np.random.permutation(X_train.shape[0])
        perm_X = X_train[perm, :]
        perm_Y = Y_train[perm, :]
        start = 0
        iters += 1

    return batch_X, batch_Y, start

def canonical_name(x):
    return x.name.split(":")[0]


# Save Result
def write_model(sess, frozen_graphdef):
    global model_folder_name
    global model_file_name

    # **Models
    # make new models folder
    i = 0
    while os.path.exists(model_folder_name + str(i)):
        i += 1
    model_path = model_folder_name + str(i)
    os.makedirs(model_path)

    model_path = os.path.join(model_path, model_file_name)
    ckpt_path = model_path + str(i) + ".ckpt"

    # save model
    # ckpt
    saver = tf.train.Saver()
    saver.save(sess, ckpt_path)
    # pb
    # tf.train.write_graph(frozen_graphdef, model_folder_name + str(i),
    #                  model_file_name + str(i) + '.pb', as_text=False)
    tf.train.write_graph(sess.graph_def, model_folder_name + str(i),
                     model_file_name + str(i) + '.pbtxt')

    pb_path = model_path + str(i) + '.pb'
    pbtxt_path = model_path + str(i) + '.pbtxt'
    opt_pb_path = model_path + "opt" + str(i) + ".pb"

    freeze_graph.freeze_graph(input_graph = pbtxt_path,  input_saver = "",
             input_binary = False, input_checkpoint = ckpt_path, output_node_names = "prediction",
             restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0",
             output_graph = pb_path, clear_devices = True, initializer_nodes = "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(pb_path, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            frozen_graphdef,
            ["input_x"], 
            ["prediction"],
            tf.float32.as_datatype_enum)

    f = tf.gfile.FastGFile(opt_pb_path, "w")
    f.write(output_graph_def.SerializeToString())

    return pb_path

def write_result(tflite_model, test_accuracy):
    global model_folder_name
    global model_file_name
    global model_info_file_name
    global _batch_size
    global epoch_num
    global layer_info

    # **Models
    # make new models folder
    i = 0
    while os.path.exists(model_folder_name + str(i)):
        i += 1
    model_path = model_folder_name + str(i-1)

    model_path = os.path.join(model_path, model_file_name)
    tflite_path = model_path + str(i-1) + ".tflite"

    # save model
    # tflite
    open(tflite_path, "wb").write(tflite_model)

    # **Write model information to csv
    minfo = [i, test_accuracy, _batch_size, epoch_num, layer_info]
    if not os.path.exists(model_info_file_name):
        model_info_file = open(model_info_file_name, "w")
        model_info_w = csv.writer(model_info_file)
        model_info_w.writerow(model_info_type)
        model_info_w.writerow(minfo)
    else:
        model_info_file = open(model_info_file_name, "a")
        model_info_w = csv.writer(model_info_file)
        model_info_w.writerow(minfo)


#######################
# **Get train&test data

# Read raw data, shape=[rawData_num, num_channels+class_num+1]=[-1,13]
raw_data = get_firebase()

# label raw data, shape=[data_num, timestep_size*num_channels+class_num]=[-1,2706]
labeled_data = label_rawData(raw_data)

# combine old data and save, shape=[data_num, timestep_size*num_channels+class_num]=[-1,2706]
all_dataset = write_data(labeled_data)

# Read raw data
# raw_data = connect_firebase()
# new_dataset = write_data(raw_data)
permutation = np.random.permutation(all_dataset.shape[0])
new_dataset = all_dataset[permutation, :]


# Get train, test data

dataNum = new_dataset.shape[0]
trainNum = int(dataNum*train_p)
trainX = new_dataset[:trainNum, class_num:]
trainY = new_dataset[:trainNum, :class_num].astype(int)
testX = new_dataset[trainNum:, class_num:]
testY = new_dataset[trainNum:,:class_num].astype(int)

print (trainX.shape)
print (trainY.shape)
print (testX.shape)
print (testY.shape)
print (trainX)
print (trainY)
print (testX)
print (testY)

# set training x, y placeholder
X_ = tf.placeholder(tf.float32, [None, timestep_size*input_size], name="input_x")
Y_ = tf.placeholder(tf.float32, [None, class_num], name="input_y")

X_train = tf.placeholder(tf.float32, [_batch_size, timestep_size*input_size], name="input_train_x")
Y_train = tf.placeholder(tf.float32, [_batch_size, class_num], name="input_train_y")

# set testing x, y placeholder
X_test = tf.placeholder(tf.float32,[None, input_size*timestep_size], name='input_test_x')
Y_test = tf.placeholder(tf.float32,[None, class_num], name='input_test_y')
keep_prob = tf.placeholder(tf.float32, [])
print (X_train.shape)

# Training Data: [_batch_size, timestep_size*input_size] ==> [_batch_size, timestep_size, input_size]
# Testing Data: [None, timestep_size*input_size] ==> [None, timestep_size, input_size]
# Build RNN LSTM layer
####################################################################

# **Step 1: Input Shape = (batch_size, timestep_size, input_size)
#X = tf.placeholder(tf.float32, [None, timestep_size, input_size])
X = tf.reshape(X_, [-1, timestep_size, input_size])

# **Step 2: Run MultiRNN with ((lstm + dropout) * 2)
mlstm_cell = []
for i in range(layer_num):
    #lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True, name='basic_lstm_cell')
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
    mlstm_cell.append(lstm_cell)
mlstm_cell = tf.contrib.rnn.MultiRNNCell(mlstm_cell,state_is_tuple=True)

# **Step3: Initiate state with zero
init_state = mlstm_cell.zero_state(tf.shape(X)[0], dtype=tf.float32)

# **Step4: Calculate in timeStep
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # Variable "state" store the LSTM state
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        outputs.append(cell_output)
h_state = outputs[-1]

# h_state is the output of hidden layer
# Weight, Bias, Softmax to predict
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
Y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias, name = "prediction")


# Loss function and accuracy
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

train_correct_prediction = tf.equal(tf.argmax(Y_pre,1), tf.argmax(Y_,1))
train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, "float"))

test_correct_prediction = tf.equal(tf.argmax(Y_pre,1), tf.argmax(Y_,1))
test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, "float"))

####################################################################

# Set up GPU demand
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True

init = tf.global_variables_initializer()
out = tf.identity(Y_pre, name="output")
#saver = tf.train.Saver()

print ("Start...")
with tf.Session(config=config) as sess:
    sess.run(init)
    for i in range(epoch_num):
        batch_X, batch_Y, epoch_start = next_batch(trainX, trainY, _batch_size, epoch_start)
        # print (batch_X.shape)
        # print (batch_Y.shape)
        # print (epoch_start)
        # batch = mnist.train.next_batch(_batch_size)
        # reshape_trainX = np.array(batch[0]).reshape(-1, timestep_size, input_size)
        # reshape_testX = np.array(mnist.test.images).reshape(-1, timestep_size, input_size)
        reshape_trainX = np.array(batch_X).reshape(-1, timestep_size, input_size)
        reshape_testX = np.array(testX).reshape(-1, timestep_size, input_size)
        if (i+1)%200 == 0:
            accuracy = sess.run(train_accuracy, feed_dict={
                X_:batch_X, Y_: batch_Y})
            #print ("Iter%d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), accuracy))
            print ("Iter%d, Epoch %d, Training Accuracy %g" % ( iters, (i+1), accuracy))
        sess.run(train_op, feed_dict={X_: batch_X, Y_: batch_Y})

    # Testing data accuracy
    # print ("Test Accuracy %g"% sess.run(test_accuracy, feed_dict={
    #     X_test: mnist.test.images, Y_test: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0], X: reshape_testX}))
    test_accuracy = sess.run(test_accuracy, feed_dict={
        X_: testX, Y_: testY})
    print ("Test Accuracy %g"% test_accuracy)

    #saver.save(sess, "model/rnn.ckpt")

    frozen_tensors = [out]
    out_tensors = [out]

    frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, list(map(canonical_name, frozen_tensors)))
    # save ckpt & pb, return pb path
    frozen_graphdef_path = write_model(sess, frozen_graphdef)

    # tf.train.write_graph(frozen_graphdef, "model",
    #                  'rnn.pb', as_text=False)
    #toco_convert
    #tflite_model = tf.contrib.lite.TocoConverter(frozen_graphdef, [X_train], out_tensors, allow_custom_ops=True)

    converter = tf.contrib.lite.TocoConverter.from_frozen_graph(frozen_graphdef_path, ["input_x"], ["prediction"])
    converter.allow_custom_ops=True
    tflite_model = converter.convert()

    #open("writer_model.tflite", "wb").write(tflite_model)

    write_result(tflite_model, test_accuracy)