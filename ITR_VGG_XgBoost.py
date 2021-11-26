#class0-airport
#class1-bakery
#class2-bar
#class3-bedroom
#class4-casino
#class5-inside_subway
#class6-kitchen
#class7-livingroom
#class8-restaurant
#class9-subway
#class10-warehouse
########################################################

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image


from keras.models import Model,Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import matplotlib.image as mpimg

dataset_path = 'C:/Users/aryan/.spyder-py3/indoorCVPR_09'
print (os.listdir(dataset_path))
SIZE = 128
train_images = []
train_labels = []
for directory_path in glob.glob("C:/Users/aryan/.spyder-py3/indoorCVPR_09/train/*"):
  label = directory_path.split("\\")[-1]
  print(label)
  for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
    print(img_path)
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img = cv2.resize(img,(SIZE,SIZE))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    train_images.append(img)
    train_labels.append(label)


train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = []
for directory_path in glob.glob("C:/Users/aryan/.spyder-py3/indoorCVPR_09/test/*"):
  fruit_label = directory_path.split("\\")[-1]
  for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
    print(img_path)
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img = cv2.resize(img,(SIZE,SIZE))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    test_images.append(img)
    test_labels.append(fruit_label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train,y_train,x_test,y_test = train_images,train_labels_encoded,test_images,test_labels_encoded

x_train = (x_train/255) 
x_test = (x_test/255)
print(x_train)
print(x_test)

VGG_model = VGG19(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3),)

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0


#Now, let us use features from convolutional network for RF
feature_extractor=VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_training = features #This is our X input to RF

#RANDOM FOREST
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data


#XGBOOST
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_for_training, y_train) #For sklearn no one hot encoding

#Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

#Now predict using the trained RF model. 
prediction = model.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction)
#print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction = model.predict(input_img_features)[0] 
prediction = le.inverse_transform([prediction])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", test_labels[n])

























