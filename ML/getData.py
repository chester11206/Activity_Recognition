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

cred = credentials.Certificate('firebase-adminsdk.json')
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://total-cascade-210406.firebaseio.com'
})

configfb = {"apiKey": "AIzaSyBkvh_XeJKp1v5XiRSiQEOsfiG5tfV7d9Y",
"authDomain": "total-cascade-210406.firebaseapp.com",
"databaseURL": "https://total-cascade-210406.firebaseio.com",  
"storageBucket": "total-cascade-210406.appspot.com"
}
timestep_size = 450

class_num = 6
class_type = ["Biking", "In Vehicle", "Running", "Still", "Tilting", "Walking", "Features"]

train_p = 0.8

csv_folder_name = "data"
csv_file_name = "rawData"

def connect_firebase_admin():
    root = db.reference()
    values = root.child('SensorDataSet').get()
    data = pd.DataFrame(values).T
    data_index = [[i.split()[0], int(i.split()[1])] for i in list(data)[class_num:]]
    
    sorted_data_index = sorted(data_index, key=itemgetter(1))
    sorted_data_index = [i[0] + " " + str(i[1]) for i in sorted_data_index]
    sorted_data_index = list(data)[:class_num] + sorted_data_index
    
    data = data.reindex(columns=sorted_data_index)
    print (data)

    npdata = np.array(data.values)
    print (npdata.shape)
    print (npdata[:,0:10])

    return npdata

def connect_firebase_pyrebase():
# add a way to encrypt those, I'm a starter myself and don't know how
    username = "chester11206@gmail.com"
    password = "abcd1234"

    firebase = pyrebase.initialize_app(configfb)
    auth = firebase.auth() 

    user = auth.sign_in_with_email_and_password(username, password)

    #user['idToken']
    # At pyrebase's git the author said the token expires every 1 hour, so it's needed to refresh it
    user = auth.refresh(user['refreshToken'])

    #set database
    db = firebase.database()
    values = db.child('SensorDataSet').get()
    data = pd.DataFrame(values.val())
    #print (dataX[0:6].T)
    npdata = np.array(data.values).T

    print (npdata.shape)
    print (npdata[0:5,:])

    # new_dataX = []
    # for i in range(0, npdataX.shape[0] , timestep_size):
    #     if i+timestep_size < npdataX.shape[0]-1 :
    #         new_dataX.append(npdataX[i:i+timestep_size, :].flatten())
    # new_dataX = np.array(new_dataX)

    # # dataNum = new_dataX.shape[0]
    # # new_dataY = npdataY[0:dataNum,:]

    # dataNum = npdataY.shape[0]
    # new_dataY = npdataY
    # new_dataX = new_dataX[0:dataNum,:]
    # print (new_dataX.shape)
    # print (new_dataY.shape)

    # raw_data = np.hstack((new_dataY, new_dataX))
    # print (raw_data.shape)
    # print (raw_data[0:5,:])

    # trainNum = int(dataNum*train_p)
    # trainX = new_dataX[0:trainNum,:]
    # trainY = new_dataY[0:trainNum,:]
    # testX = new_dataX[trainNum:dataNum,:]
    # testY = new_dataY[trainNum:dataNum,:]

    # print (trainX.shape)
    # print (trainY.shape)
    # print (testX.shape)
    # print (testY.shape)

    return npdata

def writetrain(raw_data):
    global csv_folder_name
    global csv_file_name

    # **Raw Data
    # make new csv folder
    if not os.path.exists(csv_folder_name):
        os.mkdir(csv_folder_name)

    csv_path = os.path.join(csv_folder_name, csv_file_name)
    i = 0
    while os.path.exists(csv_path + str(i) + ".csv"):
        i += 1

    #write raw data to csv
    if i != 0:
        csv_last_path = csv_path + str(0) + ".csv"
        csv_path = csv_path + str(i) + ".csv"
        # write raw data to csv
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

    all_data = list(csv.reader(open(csv_path,'r')))
    all_data = np.array(all_data[1:]).astype(float)
    print (all_data.shape)

    return all_data


raw_data = connect_firebase_admin()
all_data = writetrain(raw_data)

permutation = np.random.permutation(all_data.shape[0])
new_dataset = all_data[permutation, :]
print (all_data.shape[0])
print (new_dataset.shape[0])

dataNum = new_dataset.shape[0]
trainNum = int(dataNum*train_p)
trainX = new_dataset[:trainNum, class_num:]
trainY = new_dataset[:trainNum, :class_num].astype(int)
testX = new_dataset[trainNum:, class_num:]
testY = new_dataset[trainNum:,:class_num].astype(int)

# print (trainX.shape)
# print (trainY.shape)
# print (testX.shape)
# print (testY.shape)
# print (trainX)
# print (trainY)
# print (testX)
# print (testY)

#and now the hard/ easy part that took me a while to figure out:
# notice the value inside the .child, it should be the parent name with all the cats keys

# adding all to a dataframe you'll need to use the .val()  