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

# About save path
home = str(Path.home())
csv_folder_name = os.path.join(home, "Activity_Source/Data")
csv_file_name = "rawData"
data_file_name = "data.csv"

model_folder_name = os.path.join(home, "Activity_Source/Model/model")
model_file_name = "ActivityCNN"

model_info_file_name = os.path.join(home, "Activity_Source/Model/model_info.csv")
model_info_type = ["Index", "Test Accuracy", "Input Width", "Batch Size", "Epoch", "Kernel Size", "Depth", "Dense Size", "Model Layer"]
layer_info = "(depthConv + BatchNormarlization)*2 + (Dense + BatchNormarlization)*3, no normalize"
class_type = ["Biking", "In Vehicle", "Running", "Still", "Tilting", "Walking"]

# About ML
input_height = 1
input_width = 400
num_channels = 6
num_labels = 6

kernel_size = [30,20]
depth = [16,8]
dense_size = [256,256,num_labels]
num_hidden = 256
# LSTM Layer Num
layer_num = 2

learning_rate = 0.0001
epoch_num = 2500
epoch_start = 0
iters = 0
batch_size =128
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

    data_path = os.path.join(csv_folder_name, data_file_name)
    new_data = list(csv.reader(open(data_path,'r')))
    new_data = np.array(new_data[1:]).astype(float)

    temp1 = np.array(new_data[:,1])
    temp2 = np.array(new_data[:,2])
    new_data[:,1] = temp2
    new_data[:,2] = temp1

    act = new_data[:,0].astype(int)
    one_hot = np.zeros((act.size, len(class_type)))
    one_hot[np.arange(act.size), act] = 1

    new_data = np.hstack((one_hot, new_data[:,1:]))
    print (new_data.shape)

    return new_data

def write_data():
    global csv_folder_name
    global csv_file_name
    global data_file_name

    data_path = os.path.join(csv_folder_name, data_file_name)
    if os.path.exists(data_path):
        # read new data
        new_data = list(csv.reader(open(data_path,'r')))
        new_data = np.array(new_data[1:]).astype(float)

        act = new_data[:,0].astype(int)
        one_hot = np.zeros((act.size, len(class_type)))
        one_hot[np.arange(act.size), act] = 1

        new_data = np.hstack((one_hot, new_data[:,1:]))
        print (new_data.shape)

        # get last data path
        csv_path = os.path.join(csv_folder_name, csv_file_name)
        i = 0
        while os.path.exists(csv_path + str(i) + ".csv"):
            i += 1

        #write new data to csv
        if i != 0:
            csv_last_path = csv_path + str(i-1) + ".csv"
            csv_path = csv_path + str(i) + ".csv"

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
            csv_path = csv_path + str(i) + ".csv"
            os.rename(data_path, csv_path)
    else:
        csv_path = os.path.join(csv_folder_name, csv_file_name)
        i = 0
        while os.path.exists(csv_path + str(i) + ".csv"):
            i += 1

        csv_last_path = csv_path + str(i-1) + ".csv"
        csv_path = csv_path + str(i) + ".csv"

        # combine new and last data, save to new file
        with open(csv_last_path, 'rt') as infile:
            with open(csv_path, 'wt') as outfile:
                writer = csv.writer(outfile)
                reader = csv.reader(infile)
                writer.writerow(next(reader))
                for row in reader:
                    writer.writerow(row)

    # read data from new file
    all_data = list(csv.reader(open(csv_path,'r')))
    all_data = np.array(all_data[1:]).astype(float)

    print (all_data.shape)
    print (all_data[:20,:20])
    return all_data

def label_rawData(npdata):

    # move acceZ
    #npdata = np.delete(npdata,8,1)
    new_npdata = []
    if npdata.size != 0:
        activity_base = npdata[0, :num_labels]
        idx_base = 0
        for i in activity_base:
            if int(i) == 1:
                break
            else:
                idx_base+=1

        data_num = npdata.shape[0]
        same_count = 0
        for i in range(0, data_num):
            activity_temp = npdata[i, :num_labels]
            idx_temp = 0
            for act in activity_temp:
                if int(act) == 1:
                    break
                else:
                    idx_temp+=1
            if idx_temp == idx_base:
                same_count += 1;
                if same_count == input_width:
                    temp_data_last = npdata[i+1-input_width:i+1, num_labels:-1].flatten()
                    temp_data_last = np.hstack((activity_temp, temp_data_last))
                    new_npdata.append(temp_data_last)
                    same_count = 0;
            else:
                # if i > input_width:
                #     temp_data_last = npdata[i-input_width:i, num_labels:-1].flatten()
                #     temp_data_last = np.hstack((activity_base, temp_data_last))
                #     new_npdata.append(temp_data_last)
                if i+input_width <= data_num:
                    temp_data_next = npdata[i:i+input_width, num_labels:-1].flatten()
                    temp_data_next = np.hstack((activity_temp, temp_data_next))
                    new_npdata.append(temp_data_next)

                idx_base = idx_temp
                activity_base = activity_temp
                same_count = 0;

            # if (i+1)%input_width == 0:
            #     temp_data_last = npdata[i+1-input_width:i+1, num_labels:-1].flatten()
            #     temp_data_last = np.hstack((activity_temp, temp_data_last))
            #     new_npdata.append(temp_data_last)

            # if idx_temp != idx_base:
            #     if i%input_width != 0:
            #         if i > input_width:
            #             temp_data_last = npdata[i-input_width:i, num_labels:-1].flatten()
            #             temp_data_last = np.hstack((activity_base, temp_data_last))
            #             new_npdata.append(temp_data_last)
            #         if i+input_width <= data_num:
            #             temp_data_next = npdata[i:i+input_width, num_labels:-1].flatten()
            #             temp_data_next = np.hstack((activity_temp, temp_data_next))
            #             new_npdata.append(temp_data_next)

            #     idx_base = idx_temp
            #     activity_base = activity_temp

    new_npdata = np.array(new_npdata)
    print (new_npdata.shape)
    return new_npdata

def getXY(all_dataset):

    dataNum = all_dataset.shape[0]
    permutation = np.random.permutation(all_dataset.shape[0])
    all_dataset = all_dataset[permutation, :]

    trainNum = int(dataNum*train_p)
    print (class_type)
    print (np.sum(all_dataset[:,:num_labels], axis=0))
    trainY = all_dataset[:trainNum, :num_labels].astype(int)
    testY = all_dataset[trainNum:,:num_labels].astype(int)
    allX = all_dataset[:, num_labels:]

    # new_dataset = []
    # for i in range(0, num_channels):
    #     temp = np.array(allX[:,range(i,allX.shape[1],num_channels)])
    #     if i == 0:
    #         new_dataset = temp
    #     else:
    #         new_dataset = np.hstack((new_dataset, temp))

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
    global batch_size
    global epoch_num
    global layer_info

    # **Models
    # make new models folder
    i = 0
    while os.path.exists(model_folder_name + str(i)):
        i += 1
    # model_path = model_folder_name + str(i-1)

    # model_path = os.path.join(model_path, model_file_name)
    # tflite_path = model_path + str(i-1) + ".tflite"

    # save model
    # tflite
    # open(tflite_path, "wb").write(tflite_model)

    # **Write model information to csv
    minfo = [i-1, test_accuracy, input_width, batch_size, epoch_num, kernel_size, depth, dense_size, layer_info]
    if not os.path.exists(model_info_file_name):
        model_info_file = open(model_info_file_name, "w")
        model_info_w = csv.writer(model_info_file)
        model_info_w.writerow(model_info_type)
        model_info_w.writerow(minfo)
    else:
        model_info_file = open(model_info_file_name, "a")
        model_info_w = csv.writer(model_info_file)
        model_info_w.writerow(minfo)

def write_tfite(tflite_modely):
    global model_folder_name
    global model_file_name

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

# ML layer
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)
    
def apply_depthwise_conv(x,kernel_size,num_channels,depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])

    # filter = weigths, shape=[filter_height, filter_width, in_channels, channel_multiplier]
    #                  =[1, kernel_size, num_channels, depth]
    # output_channels = in_channels * channel_multiplier
    output_conv = tf.nn.depthwise_conv2d(x,weights, [1, 1, 1, 1], padding='VALID')

    return tf.nn.relu(tf.add(output_conv,biases))
    
def apply_max_pool(x,kernel_size,stride_size):
    return tf.layers.max_pooling2d(inputs=x, pool_size=[1, kernel_size], strides=stride_size)


def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    # 输入的通道数
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = weight_variable([kh,kw,n_in,n_out])
        #kernel = tf.get_variable("w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val , trainable=True , name='b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        p += [kernel,biases]
        return activation

# 定义全连接层
def fc_op(input_op,name,n_out,p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = weight_variable([n_in,n_out])
        #kernel = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
        # tf.nn.relu_layer()用来对输入变量input_op与kernel做乘法并且加上偏置b
        activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        p += [kernel,biases]
        return activation

# 定义最大池化层
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)

#定义网络结构
def inference_op(input_op,keep_prob):
    p = []
    conv1_1 = conv_op(input_op,name='conv1_1',kh=1,kw=3,n_out=64,dh=1,dw=1,p=p)
    conv1_2 = conv_op(conv1_1,name='conv1_2',kh=1,kw=3,n_out=64,dh=1,dw=1,p=p)
    pool1 = mpool_op(conv1_2,name='pool1',kh=1,kw=3,dw=2,dh=2)

    conv2_1 = conv_op(pool1,name='conv2_1',kh=1,kw=3,n_out=128,dh=1,dw=1,p=p)
    conv2_2 = conv_op(conv2_1,name='conv2_2',kh=1,kw=3,n_out=128,dh=1,dw=1,p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=1, kw=3, dw=2, dh=2)

    conv3_1 = conv_op(pool2, name='conv3_1', kh=1, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=1, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=1, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=1, kw=3, dw=2, dh=2)

    # conv4_1 = conv_op(pool3, name='conv4_1', kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv4_2 = conv_op(conv4_1, name='conv4_2', kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv4_3 = conv_op(conv4_2, name='conv4_3', kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    # pool4 = mpool_op(conv4_3, name='pool4', kh=1, kw=3, dw=2, dh=2)

    # conv5_1 = conv_op(pool4, name='conv5_1', kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv5_2 = conv_op(conv5_1, name='conv5_2', kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv5_3 = conv_op(conv5_2, name='conv5_3', kh=1, kw=3, n_out=512, dh=1, dw=1, p=p)
    # pool5 = mpool_op(conv5_3, name='pool5', kh=1, kw=3, dw=2, dh=2)

    shp = pool3.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool3,[-1,flattened_shape],name="resh1")

    fc6 = fc_op(resh1,name="fc6",n_out=1024,p=p)
    fc6_drop = tf.nn.dropout(fc6,keep_prob,name='fc6_drop')
    fc7 = fc_op(fc6_drop,name="fc7",n_out=1024,p=p)
    fc7_drop = tf.nn.dropout(fc7,keep_prob,name="fc7_drop")
    fc8 = fc_op(fc7_drop,name="fc8",n_out=512,p=p)

    return fc8,p

def RNN_op(input_op, keep_prob):
    mlstm_cell = []
    for i in range(layer_num):
        #lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, forget_bias=1.0, state_is_tuple=True, name='basic_lstm_cell')
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
        mlstm_cell.append(lstm_cell)
    mlstm_cell = tf.contrib.rnn.MultiRNNCell(mlstm_cell,state_is_tuple=True)

    # **Step3: Initiate state with zero
    init_state = mlstm_cell.zero_state(tf.shape(X)[0], dtype=tf.float32)

    # **Step4: Calculate in timeStep
    outputs = list()
    state = init_state
    with tf.variable_scope('RNN'):
        input_shape = input_op.get_shape().as_list()
        print (input_shape)
        for timestep in range(batch_size):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # Variable "state" store the LSTM state
            timestep_X = tf.reshape(input_op[timestep], [-1,input_height,input_width,num_channels])
            timestep_cell, p = inference_op(timestep_X,keep_prob)

            (cell_output, state) = mlstm_cell(timestep_cell, state)
            outputs.append(cell_output)

    h_state = outputs[-1]

    return h_state

#######################
# **Get train&test data

# combine old data and save, shape=[data_num, input_width*num_channels+num_labels]=[-1,2706]
all_data = read_data()
#all_data = write_data()

# label raw data, shape=[data_num, input_width*num_channels+num_labels]=[-1,2706]
labeled_data = label_rawData(all_data)

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
X = tf.reshape(X_, [-1,input_height,input_width,num_channels])
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

# h_state is the output of hidden layer
# Weight, Bias, Softmax to predict
W = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[num_labels]), dtype=tf.float32)
Y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias, name = "prediction")

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
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True

init = tf.global_variables_initializer()
out = tf.identity(Y_pre, name="output")

with tf.Session(config=config) as sess:
    sess.run(init)
    for epoch in range(epoch_num):
        batch_X, batch_Y, epoch_start = next_batch(trainX, trainY, batch_size, epoch_start)
        if (epoch+1)%200 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                X_:batch_X, Y: batch_Y})
            #print ("Iter%d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), accuracy))
            print ("Iter%d, Epoch %d, Training Accuracy %g" % ( iters, (epoch+1), train_accuracy))
        sess.run(optimizer, feed_dict={X_: batch_X, Y: batch_Y})

    test_accuracy = sess.run(accuracy, feed_dict={
        X_: testX, Y: testY})
    print ("Test Accuracy %g"% test_accuracy)

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
