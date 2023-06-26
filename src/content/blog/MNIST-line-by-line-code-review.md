---
author: Sonia Milanoi
pubDatetime: 2023-04-23T16:02:00Z
title: MNIST:A Line-By-Line Code Review
postSlug: MNIST-code-review
featured: true
draft: false
tags:
  - article
ogImage: ""
description:
  A line by line explainer of a machine learning model using the popular MNIST dataset
---

This post explains how to create a simple Machine Learning model using the MNIST dataset. It covers the process of loading the dataset, reshaping it, normalizing it, one-hot encoding it, and creating a model using the Sequential API. It also explains the use of Conv2D layers, max pooling, flattening, and Dense layers. Finally, it covers the use of SGD as the optimizer and categorical cross-entropy as the loss function.


MNIST is one of the simplest Machine Learning models you'll see.

Most people have heard about MNIST, but only a few can explain how this code works.

Let's go together line by line and understand what's happening here: 


Let's start by loading the MNIST dataset.

It contains 70,000 28x28 images showing handwritten digits.

The function returns the dataset split into train and test sets. 



x_train and x_test contain our train and test images.

y_train and y_test contain the target values: a number between 0 and 9 indicating the digit shown in the corresponding image.

We have 60,000 images to train the model and 10,000 to test it. 

When dealing with images, we need a tensor with 4 dimensions: batch size, width, height, and color channels.

x_train is (60000, 28, 28). We must reshape it to add the missing dimension ("1" because these images are grayscale.) 



Each pixel goes from 0 to 255. Neural networks work much better with smaller values.

Here we normalize pixels by dividing them by 255. That way, each pixel will go from 0 to 1. 



Target values go from 0 to 9 (the value of each digit.)

This line one-hot encodes these values.

For example, this will transform a value like 5 in an array of zeros with a single 1 corresponding to the fifth position:

[0, 0, 0, 0, 0, 1, 0, 0, 0, 0] 



Let's now define our model.

There are several ways to create a model in Keras. This one is called the "Sequential API."

Our model will be a sequence of layers we will define individually. 



A lot is going on with this first line.

First, we define our model's input shape: a 28x28x1 tensor (width, height, channels.)

This is exactly the shape we have in our train dataset. 



Then we define our first layer: a Conv2D layer with 32 filters and a 3x3 kernel.

This layer will generate 32 different representations using the training images. 



We must also define the activation function used for this layer: ReLU.

You'll see ReLU everywhere. It's a popular activation function.

It will allow us to solve non-linear problems, like recognizing handwritten digits. 



After our Conv2D layer, we have a max pooling operation.

The goal of this layer is to downsample the amount of information collected by the convolutional layer.

We want to throw away unimportant details and retain what truly matters. 



We are now going to flatten the output. We want everything in a continuous list of values.

That's what the Flatten layer does. It will give us a flat tensor. 



Finally, we have a couple of Dense layers.

Notice how the output layer has a size of 10, one for each of our possible digit values, and a softmax activation.

The softmax ensures we get a probability distribution indicating the most likely digit in the image. 



After creating our model, we compile it.

I'm using Stochastic Gradient Descent (SGD) as the optimizer.

The loss is categorical cross-entropy because this is a multi-class classification problem.

We want to record the accuracy as the model trains. 



Finally, we fit the model. This starts the training process.

A couple of notes:

• I'm using a batch size of 32 images.
• I'm running 10 total epochs.

When fit() is done, we'll have a fully trained model! 



Let's now test the model.

This gets a random image from the test set and displays it.

Notice that we want the image to come from the test set, containing data the model didn't see during training. 



We can't forget to reshape and normalize the image as we did before. 



Finally, I predict the value of the image.

Remember that the result is a one-hot-encoded vector. That's why I take the argmax value (the position with the highest probability), and that's the result. 



If you have any questions about this code, reply to this thread and I'll help you understand what's happening here.
