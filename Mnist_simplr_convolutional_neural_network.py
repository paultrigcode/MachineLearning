# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 20:22:26 2018

@author: PaulTrig
"""

#in this tut we will implement a simple convolutional neural network in tensor flow which has a 
#classification accuracy of about 99%
#Convolutional neural network works by moving small filters across the input image....
#This means that the filter are reused for  recognizing patterns through out the entire input image

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
print(tf.__version__)
#Next,we define the configuration of the neural network
#The configuration is defined here for convenience so that you can easily find and change this number
#and rerun the notebook
#convolutfion layer 1
filter_size1=5#convolution filters are 5X5 pixels
num_filters1=16#There are 16 of this filters
#convolution layer 2
filter_size2=5#convolutfion filters are 5X5 pixels
num_filters2=36# there are 36 of this filters
#fully connected layer
fc_size=128#Number of Neurons in fully connected layer


from tensorflow.examples.tutorials.mnist import input_data
data=input_data.read_data_sets("C:/Users/PaulTrig/MNIST_data",one_hot=True)
data.test.cls=np.argmax(data.test.labels,axis=1)
#data dimensions
img_size=28#we know that mnist image are 28 pixels in each dimensions
#images are stored in one dimensional array of this length
img_size_flat=img_size*img_size
#Tuple with height and width of images used to reshape arrays
img_shape=(img_size,img_size)
#Number of colour channels for the images: 1 channel for gray_scale
num_channels=1
#Number of classes,one for each of 10 digits
num_classes=10


def plot_images(images,cls_true,cls_pred=None):
    assert len(images)==len(cls_true)==9
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape),cmap='binary')
        if cls_pred is None:
            xlabel="True: {0}".format(cls_true[i])
        else:
            xlabel="True:{0},pred:{1}".format(cls_true[i],cls_pred[i])
            
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
images=data.test.images[0:9]
cls_true=data.test.cls[0:9]
plot_images(images=images,cls_true=cls_true)


#The entire purpose of tesnsorflow is to have a so called computational graph that
#can be executed much more efficiently than if the same calculations were to be performed
#directly in python...Tensorflow can be more eficient than numpy because tensorflow
#knows the entire computation graph that must br executed, while numpy knows the computation of a 
#single mathematical operation ata time
#Tensorflow can also automatically calculate the gradients that are needed to optimize
#the variable of the graph so as to make the model perform better...This is becuase the
#graph is a  combination of simple mathematical expressions so the gradient of the entire graph
#can be calculated using the chain rule of derivatives
#Tensorflow can even take advantage of multicore cpus as well as GPUs...and Google has even
# built special chips just for Tensorflow which are called Tpus(tensor Processimg unis)
#and are even faster than Gpus
#A tensorflow graph consists of the following parts which will be detailed below
#1 placeholder variable used ford inputting data to the graph
#2 Variables that are going to be optimized so as to make the convolutional netwsork perfordm better
#3Thes mathematical formulas for the convolutional network
#4 A cost measure that can be used to guide the optimization of the variable
#4 An optimization method which updates the variables



#Helper Funcyion for creating new variables
#function for creating new Tensorflow variables and initializing them with random values
#Note that the initialization is not actually done at this point,it is merely being
#defined in the tensor graph

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def new_biases(legnth):
    return tf.Variable(tf.constant(0.05,shape=[legnth]))

#Heper function for creating a new convolution layer
#This function creates a new convolutional layer in the computational graph for Tensorflow
#Nothing is actually calculated here,we are just adding the mathematical formulas to the tensor graph
#It is assumed that the input is a 4dim tensor  with the following dimensions
#1 Image number
#2 X_axis of each image'
#3 Y_axis of each image'
#4 Channels of each image
#The output is another 4dim tensor with the following dimensions:
    #1 image number same as input
    #2 X_axis  of each image
    #3 y_axis of each image'
    #4channel produced by each of the convolution layer
    
def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):
    shape=[filter_size,filter_size,num_input_channels,num_filters]
    weights=new_weights(shape=shape)
    biases=new_biases(length=num_filters)
    layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
    layer+=biases
    if use_pooling:
        layer=tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    layer=tf.nn.relu(layer)
    return layer,weights


def flatten_layer(layer):
    layer_shape=layer.get_shape()
    num_features=np.array(layer_shape[1:4],dtype=int).prod()
    layer_flat=tf.reshape(layer,[-1,num_features])
    return layer_flat,num_features


def new_fc_layer(input,num_inputs,num_outputs,use_relu=True):
    weights=new_weights(shape=[num_inputs,num_outputs])
    biases=new_biases(length=num_outputs)
    layer=tf.matmul(inputs,weights)+biases
    if use_relu:
        layer=tf.nn.relu(layer)
        
    return layer

x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
x_image=tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,10],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)

layer_conv1,weights_conv1=new_conv_layer(input=x_image,num_input_channels=num_channels,
                                         filter_size=filter_size1,
                                         num_filters=num_filters1,
                                         use_pooling=true)


layer_conv2,weights_conv2=new_conv_layer(input=layer_conv1,num_input_channels=num_channels,
                                         filter_size=filter_size2,
                                         num_filters=num_filters2,
                                         use_pooling=true)


layer_fc2=new_fc_layer(input=layer_fc1,num_inputs=fc_size,
                                         num_ouputs=num_classes,
                                         use_relu=false)

    





    






    




