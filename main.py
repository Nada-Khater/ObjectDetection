import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.applications.vgg16 import VGG16
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#reading images from pc and converting array to numpy array
path = 'E:\DataSet'
data,label = [],[]
for root, dirs, files in os.walk(path):
    key = os.path.basename(root)
    for file in files:
        full_file_path = os.path.join(root,file)
        img = cv2.imread(full_file_path)
        img = cv2.resize(img,(256,256))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        data.append(img)
        label.append(key)

data = np.array(data)
label = np.array(label)

#spliting dataset into train and test datasets
data_train,data_test,label_train,label_test = train_test_split(data,label,test_size=0.3)

#Encode labels from string to int
le = preprocessing.LabelEncoder()
labelEnc_test = le.fit_transform(label_test)
le = preprocessing.LabelEncoder()
labelEnc_train = le.fit_transform(label_train)

#normalize images
data_train,data_test = data_train/255.0 , data_test/255.0

#getting features from images
model = VGG16(weights='imagenet',include_top=False,input_shape=(256,256,3))

data_train_feature = model.predict(data_train)

data_train_features = data_train_feature.reshape(data_train_feature.shape[0], -1)
data_train_rf = data_train_features

#Random Forest
rfc = RandomForestClassifier(n_estimators=100,random_state=42)
rfc.fit(data_train_rf,labelEnc_train)

data_test_feature = model.predict(data_test)

data_test_features = data_test_feature.reshape(data_test_feature.shape[0], -1)

predRF = rfc.predict(data_test_features)
predRF = le.inverse_transform(predRF)

print("Accuracy = ",metrics.accuracy_score(label_test,predRF))

idx = np.random.randint(0,data_test.shape[0])
pic = data_test[idx]
Ipic = np.expand_dims(pic,axis=0)
IFpic = model.predict(Ipic)
IFpic1 = IFpic.reshape(IFpic.shape[0],-1)
predRF = rfc.predict(IFpic1)[0]
predRF = le.inverse_transform([predRF])
print("The Prediction For This Image =",predRF)
print("The Actual Name For This Image =",label_test[idx])
fig,ax = plt.subplots()
ax.imshow(pic)
rect = patches.Rectangle((20, 30),200,200,linewidth=2,edgecolor='r',facecolor='none')
image = cv2.putText(pic,''.join(predRF),(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),thickness=2)
ax.add_patch(rect)
plt.imshow(pic)