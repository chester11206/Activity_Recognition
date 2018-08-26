import pyrebase
import pandas as pd
import numpy as np
import csv
import sys
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from operator import itemgetter
from pathlib import Path

# About firebase
home = str(Path.home())

cred = credentials.Certificate('firebase-adminsdk.json')
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://total-cascade-210406.firebaseio.com'
})

# About save path
csv_folder_name = os.path.join(home, "Activity_Source/Data")
csv_file_name = "rawData"


input_size = 6
timestep_size = 450
class_num = 6
class_type = ["Biking", "In Vehicle", "Running", "Still", "Tilting", "Walking", "Features"]

train_p = 0.8


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

def load_data(datafile):
    read_data = list(csv.reader(open(datafile,'r')))

    data = []
    
    for item in read_data[1:]:
        data.append([i for i in item])
    data = np.array(data)
    print (data.shape)
    
    return data


# readdata = load_data("rawData1.csv")
# print (readdata.shape)

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