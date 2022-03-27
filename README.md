# Comparision of Indoor Scene Classification Models

## Introduction

Humans for many years have yearned to understand the intricacies which evade them, and
continuous learning has helped them achieve many problems which we thought were
unsolvable. However, humans fall behind in a vital part of learning i.e., repetition. The
reasons for this have long been studied and have always concluded that we are slow in
repetitive tasks, but this is not the case for computers. Computers excel in repetitive tasks
which have helped humans achieve the solution to many problems. This comes in full light in
the field of Machine Learning. Machine Learning is the ability of computers to learn and
repeat without being explicitly programmed. It is a novel field that has applications in the
field of medicine, physics, chemistry, mathematics, and computer science. This project
mainly targets the comparison of different supervised learning classification models on image
data, and which one performs the best. This project is based on the paper titled ‘Indoor Scene
Recognition’ CVPR ’09 which tries to give us an understanding of how indoor scene
classification works but has not gone into testing the many models which improve its
accuracy.

### Objectives

This project helped me achieve the following objectives: \
a) Understanding different concepts and fundamentals of machine learning. \
b) Getting acquainted with the different terminologies of statistics. \
c) Gaining and applying mathematical knowledge into a field that has vast
   applications of it and working on computer vision tools. \
d) Understanding the working and use of different machine learning and deep
   learning algorithms. \
e) Implementing various supervised learning ML algorithms on a particular dataset and measuring its
   performance. \
f) Learning how to represent data by plotting graphs, histograms, etc. using the
   matplotlib library in python. \
g) Using NumPy arrays in python to make algorithms work fast and efficiently. \
h) To get a better understanding of the uses of Ensemble learning techniques on
   image datasets in Machine Learning. \
i) Perform a comparative study on different types of models used. 

### Weekly Breakup 

My weekly breakup of activities for this mini-project can be summarized in the following
manner: \
Week 1 \
a) Learning about CNN and VGG \
b) Learning to use OpenCV, matplotlib, seaborn \
c) Implementing simple algorithms on the CIFAR-10 dataset \
d) Reading papers on VGG as a feature extractor \
Week 2 \
a) Building a CNN model on MNIST dataset \
b) Implementing hyperparameter tuning \
c) Applying data augmentation techniques and optimizers \
d) Paper Reading \
Week 3 \
a) Collecting and cleaning the dataset \
b) Learning to implement Principal Component Analysis \
c) Implementing Xgboost and Random Forest \
Week 4 \
a) Paper Reading and Implementation \
b) Project Report 

## Dataset

To train the models for image classification we need to collect image data. Any Image
classification task requires many corrupt-free images belonging to a particular class. I have
taken images from across the web using selenium as well as using some images from the
original paper. The dataset originally consisted of 67 classes, with each class having an
unbalanced number of images, to prevent issues with overfitting and imbalanced datasets, I
ensured that only 11 classes were selected out of which 100 images were used for training
and 20 images for testing. Due to hardware constraints, the entire dataset couldn't be trained
however they need to be explored further with modern techniques.
Some of the more difficult classes were chosen and tested upon.
The problem with image classification tasks lies in the selection of good training data, for
example, two images belonging to 2 different classes in our eyes could look very similar to
the model.
![image](https://user-images.githubusercontent.com/68948344/160279789-68a5490e-624f-47ea-bba0-12e3d692395a.png)


The images were separated into different classes using one-hot encoding to label them as
integers from their respective categorical integer data types. It is represented by ‘1’ and ‘0’’s
indicating true and false. One-hot encoding is a process by which categorical variables are
converted into a form that could be provided to ML algorithms to do a better prediction.
The main difference between Label Encoder and One-hot is that one-hot generates Boolean
values for each category and only one of these categories can take the value 1 for each
sample. \
Hence, the term one-hot encoding.

![image](https://user-images.githubusercontent.com/68948344/160279833-3019ce69-1f2f-4220-9163-0a4831fafce5.png)


The dataset also went through preprocessing techniques to ensure a quality dataset that may
not result in a high number of true negatives or false positives. All the images were resized to
128x128 and converted to grayscale. Finally, the images were split into their respective
training and testing groups, they were also normalized to ensure that the values remained
between 0 and 1.


![image](https://user-images.githubusercontent.com/68948344/160279853-1c229ca1-9e66-461c-aa80-b7a8fd5b5e5b.png)




