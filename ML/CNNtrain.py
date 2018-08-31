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
csv_folder_name = os.path.join(home, "Activity_Source/Datas")
csv_file_name = "rawData"

model_folder_name = os.path.join(home, "Activity_Source/Models/model")
model_file_name = "ActivityCNN"

model_info_file_name = os.path.join(home, "Activity_Source/Models/model_info.csv")
model_info_type = ["Index", "Test Accuracy", "Input Width", "Batch Size", "Epoch", "Kernel Size", "Depth", "Dense Size", "Model Layer"]
layer_info = "(depthConv + BatchNormarlization)*2 + (Dense + BatchNormarlization)*6"
class_type = ["Biking", "In Vehicle", "Running", "Still", "Tilting", "Walking", "Features"]

# About ML
input_height = 1
input_width = 250
num_channels = 6
num_labels = 6

kernel_size = [30,20]
depth = [16,8]
dense_size = [256,256,num_labels]
num_hidden = 256

learning_rate = 0.0001
epoch_num = 2500
epoch_start = 0
iters = 0
batch_size = 64
perm_X = np.empty
perm_Y = np.empty

# Training Data / Raw Data
train_p = 0.9


################
# **Function

# Data Process
def get_firebase():

    # connect firebase
    root = db.reference()
    values = root.child('SensorDataSet').get()
    data = pd.DataFrame(values).T
    
    data_num = data.shape[0] - (data.shape[0] % input_width)
    npdata = np.array(data.values)
    npdata = npdata[:data_num,:]

    return npdata

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

    return new_data

def label_rawData(npdata):

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
        for i in range(0, data_num):
            activity_temp = npdata[i, :num_labels]
            idx_temp = 0
            for act in activity_temp:
                if int(act) == 1:
                    break
                else:
                    idx_temp+=1

            if (i+1)%input_width == 0:
                temp_data_last = npdata[i+1-input_width:i+1, num_labels:-1].flatten()
                temp_data_last = np.hstack((activity_temp, temp_data_last))
                new_npdata.append(temp_data_last)

            if idx_temp != idx_base:
                if i%input_width != 0:
                    temp_data_last = npdata[i-input_width:i, num_labels:-1].flatten()
                    temp_data_last = np.hstack((activity_base, temp_data_last))
                    new_npdata.append(temp_data_last)
                if i+input_width <= data_num:
                    temp_data_next = npdata[i:i+input_width, num_labels:-1].flatten()
                    temp_data_next = np.hstack((activity_temp, temp_data_next))
                    new_npdata.append(temp_data_next)

                idx_base = idx_temp
                activity_base = activity_temp

    new_npdata = np.array(new_npdata)
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

# clean firebase
def remove_firebase():
    root = db.reference()
    root.child('SensorDataSet').delete()

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
    global model_info_file_name
    global batch_size
    global epoch_num
    global layer_info

    # **Write model information to csv
    i = 0
    while os.path.exists(model_folder_name + str(i)):
        i += 1

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


#######################
# **Get train&test data

# Read raw data, shape=[rawData_num, num_channels+num_labels+1]=[-1,13]
raw_data = get_firebase()

# combine old data and save, shape=[data_num, input_width*num_channels+num_labels]=[-1,2706]
all_data = write_data(raw_data)

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

remove_firebase()


#############
# **Build CNN

X_ = tf.placeholder(tf.float32, shape=[None,input_height*input_width*num_channels], name="input_x")
Y = tf.placeholder(tf.float32, shape=[None,num_labels], name="input_y")

X = tf.reshape(X_, [-1,input_height,input_width,num_channels])

# depth_conv = apply_depthwise_conv(X,kernel_size,num_channels,depth)
# max_pool = apply_max_pool(depth_conv,20,2)
# depth_conv = apply_depthwise_conv(max_pool,6,depth*num_channels,depth//10)

beta= tf.get_variable("beta",shape=[],initializer=tf.constant_initializer(0.0))
gamma=tf.get_variable("gamma",shape=[],initializer=tf.constant_initializer(1.0))


depth_conv = X
for i in range(len(kernel_size)):
    channels = num_channels
    if i > 0:
        for j in range(i):
            channels = channels * depth[j]
    depth_conv = apply_depthwise_conv(depth_conv,kernel_size[i],channels,depth[i])

    axes=[d for d in range(len(depth_conv.get_shape()))]
    x_mean,x_variance=tf.nn.moments(depth_conv,axes)
    depth_conv=tf.nn.batch_normalization(depth_conv,x_mean,x_variance,beta,gamma,1e-10,"bn")

    #depth_conv=max_pool = apply_max_pool(depth_conv,20,2)

shape_conv = depth_conv.get_shape().as_list()
logits = tf.reshape(depth_conv, [-1, shape_conv[1] * shape_conv[2] * shape_conv[3]])

for i in range(len(dense_size)):
    logits = tf.layers.dense(inputs=logits, units=dense_size[i], activation=tf.nn.relu)

    axes=[d for d in range(len(logits.get_shape()))]
    x_mean,x_variance=tf.nn.moments(logits,axes)
    logits=tf.nn.batch_normalization(logits,x_mean,x_variance,beta,gamma,1e-10,"bn")

shape_logits = logits.get_shape().as_list()
# dense1 = tf.layers.dense(inputs=depth_conv, units=1024, activation=tf.nn.relu)
# dense2= tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu)
# logits= tf.layers.dense(inputs=dense2, units=256, activation=None)

# shape = logits.get_shape().as_list()
# c_flat = tf.reshape(logits, [-1, shape[1] * shape[2] * shape[3]])

# f_weights_l1 = weight_variable([shape[1] * shape[2] * shape[3], num_hidden])
# f_biases_l1 = bias_variable([num_hidden])
# f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))

# out_weights = weight_variable([num_hidden, num_labels])
# out_biases = bias_variable([num_labels])
# Y_pre = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases, name = "prediction")
Y_pre = tf.nn.softmax(logits, name = "prediction")

loss = -tf.reduce_sum(Y * tf.log(Y_pre))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

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
    write_result(test_accuracy)

    # converter = tf.contrib.lite.TocoConverter.from_frozen_graph(frozen_graphdef, ["input_x"], ["prediction"])
    # tflite_model = converter.convert()
    # write_tflite(tflite_model)
