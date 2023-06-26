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

MNIST is one of the simplest Machine Learning models you'll see, especially as a beginner in machine learning. Most people have heard about MNIST, but only a few can explain how this code works. 

This post explains how to create a simple Machine Learning model using the MNIST dataset. It covers the process of loading the dataset, reshaping it, normalizing it, one-hot encoding it, and creating a model using the Sequential API. It also explains the use of Conv2D layers, max pooling, flattening, and Dense layers. Finally, it covers the use of SGD as the optimizer and categorical cross-entropy as the loss function.

```python
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_train = x_train.astype("float32") / 255.0

y_train = to_categorical(y_train)

model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(x_train, y_train, epochs=10, batch_size=32)

image = random.choice(x_test)

plt.imshow(image, cmap=plt.get_cmap("gray"))
plt.show()

image = (image.reshape((1, 28, 28, 1))).astype("float32") / 255.0

digit = np.argmax(model.predict(image)[0], axis=-1)
print("Prediction:", digit)

```

## The Break Down

Let's go together line by line and understand what's happening in this code. I’ll skip the import section because I believe if you’re implementing any kind of machine learning model you already know what the import function does.

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

This line loads the MNIST dataset. The MNIST dataset is a dataset which consists of handwritten digits and their corresponding labels. The data is split into training and testing sets, assigned to `(x_train, y_train)` and `(x_test, y_test)`, respectively which will be used to train the model and then test the model’s performance. The `x_train` and `x_test` contain our train and test images while the `y_train` and `y_test` contain a number between 0 and 9 indicating the digit shown in the corresponding image.

```python
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_train = x_train.astype("float32") / 255.0

```

These lines reshape the training images to have dimensions `(number of samples, 28, 28, 1)`, where 28x28 is the image size and 1 represents the number of channels (grayscale). The reshaping is necessary to match the expected input shape of the model. The images are also converted to the float32 data type and normalized by dividing them by 255.0, scaling the pixel values to a range between 0 and 1. This normalization aids in faster convergence during training.

<aside>
💡 
  **Convergence** in this context refers to the process of gradually reaching an optimal state where the model's parameters are adjusted to minimize the loss function and achieve stable performance. It signifies that the model has learned from the training data and can make reliable predictions

</aside>

```python
y_train = to_categorical(y_train)

```

This line performs one-hot encoding on the training labels (`y_train`). It converts the labels into a categorical format, where each label is represented as a binary vector. This encoding is suitable for multi-class classification tasks like digit recognition.

<aside>
💡 

  **One-hot encoding** is a technique used to convert categorical data into binary format. It converts categorical variables into a numerical representation that allows algorithms to effectively process and understand the data. For example: Suppose we have a categorical variable "Color" with three categories: Red, Green, and Blue. In one-hot encoding, we create three binary features, one for each category. If we have a data point with the color "Red," the one-hot encoded representation would be [1, 0, 0]. Here, the first element represents "Red" and is set to 1, while the other elements (representing "Green" and "Blue") are set to 0. Similarly, for "Green," the one-hot encoding would be [0, 1, 0], and for "Blue," it would be [0, 0, 1]. Each feature represents whether the observation belongs to a particular category or not.

</aside>

```python
model = Sequential([...])

```

Here, a sequential model is created using the Keras `Sequential` class. The ellipsis (`[...]`) represents the layers added to the model. In this implementation, the model consists of ***convolutional layers, pooling layers*,** and ***dense layers***, which form a convolutional neural network (CNN). 

<aside>
💡 
  
  **Hyperparameters** are settings or configuration choices that are set before training the model. They are not learned from the data but are specified by the user. Examples of hyperparameters include the learning rate, the number of layers in the model, the number of neurons in each layer, etc. Choosing appropriate hyperparameters is crucial for the model's performance.

</aside>

Let’s get into the Sequential Model. There are several ways to create a model in Keras. This one is called the "Sequential API."

```python
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(10, activation="softmax"),
])

```

1. The `model` is initialized as a `Sequential` model. The `Sequential` class allows you to create a linear stack of layers.
2. `Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1))` creates a convolutional layer with 32 filters of size 3x3. This layer applies convolutional operations to the input image. The `"relu"` activation function is used to introduce non-linearity. The `input_shape` parameter defines the shape of the input image, which is (28, 28, 1) for grayscale images.
3. `MaxPooling2D((2, 2))` adds a max pooling layer with a pool size of 2x2. This layer downsamples the input, reducing the spatial dimensions, and helps to extract the most important features while reducing computational complexity.
4. `Flatten()` adds a flatten layer. This layer reshapes the multidimensional output of the previous layer into a one-dimensional vector. It prepares the data for the subsequent fully connected layers.
5. `Dense(100, activation="relu")` adds a fully connected layer with 100 neurons and applies the "relu" activation function. This layer captures high-level features from the flattened input.
6. `Dense(10, activation="softmax")` adds the final fully connected layer with 10 neurons, corresponding to the 10 possible digit classes (0 to 9). The "softmax" activation function is used to produce probability distributions over the classes, indicating the likelihood of each class.

When implementing a model, the specific architecture design, including the number of layers and their configurations, is a trade-off between model complexity and performance. These design decisions are made based on prior knowledge, experimentation, and empirical observations. It's important to strike a balance between model complexity and performance.

The design decisions and trade-offs in this model architecture are discussed below:

1. **Convolutional Layers**: Convolutional layers are effective in capturing spatial patterns and local dependencies in images. However, increasing the number of convolutional layers or filters can lead to increased model complexity, which may require more computational resources and longer training times. There's a trade-off between the model's capacity to learn complex features and the computational constraints.
2. **Pooling Layers**: Max pooling layers downsample the spatial dimensions, reducing the model's sensitivity to small local variations. However, aggressive pooling can lead to information loss and decreased spatial resolution. Choosing an appropriate pool size is crucial to balance retaining important information and reducing the model's complexity.
3. **Flatten Layer**: The flatten layer is necessary to transition from the convolutional layers to the fully connected layers. It converts the multi-dimensional feature maps into a flat vector. There are no significant trade-offs associated with this layer since it doesn't introduce additional model complexity.
4. **Fully Connected Layers**: The dense layers after flattening extract high-level representations and perform classification based on these representations. The number of neurons in these layers influences the model's capacity to learn complex patterns. However, increasing the number of neurons increases the model's parameter count and computational requirements, which can lead to overfitting if not balanced properly.
5. **Activation Functions**: The "relu" activation function is commonly used in convolutional and dense layers to introduce non-linearity, enabling the model to learn complex relationships. Choosing the appropriate activation function depends on the specific problem and the desired properties of the model.

If you haven’t noticed already, I love using analogies to explain complex concepts in clear and simple English. Analogies are intuitive and make it easier to gain deep understanding. Incase you haven’t fully grasped what the jargon above means, here are some analogies that should make the model architecture more accessible.

<aside>
💡 
  
  **What is:**

**A Convolutional Layer?**

Imagine you have a set of filters (like highlighter pens) that you slide over an image. Each filter looks for specific patterns (like edges or textures) in different parts of the image. The output of this process is a set of filtered images that highlight different features of the original image.

**A Max Pooling Layer?**

Think of it as a way to summarize information. Imagine dividing the filtered image into small squares and only keeping the brightest pixel from each square. By doing this, you reduce the size of the image and focus on the most important features while discarding some less important details.

**A Flatten Layer?** 

Picture taking a stack of papers and laying them out in a single line. The flatten layer does something similar with the filtered images, transforming them from their multidimensional shape into a simple, long vector. This prepares the data for further analysis.

**A Fully Connected Layer?**

 Visualize a network of interconnected nodes, like a social network. Each node receives input from all the nodes in the previous layer and performs calculations based on those inputs. These calculations help the model understand complex relationships between different features in the data.

**An Activation Function?** 

Analogous to a decision-making process, an activation function determines if a neuron "fires" or not based on the input it receives. In the case of "relu" (Rectified Linear Unit), if the input is positive, it "fires" and passes the information to the next layer; otherwise, it doesn't do anything.

**A Neuron?**

In a neural network, neurons are the basic computational units. They receive input, perform calculations using weighted connections, apply an activation function, and produce an output. Neurons are organized in layers, and they collectively learn to extract and represent patterns and features from the input data.

**A Parameter?**

Think of parameters as knobs that the model adjusts during training to make accurate predictions. These knobs control how the model learns from the data. For example, the number of filters or neurons, the size of the filters are all parameters.

**Overfitting?**

Imagine trying to memorize a specific set of answers to a quiz without understanding the underlying concepts. In machine learning, overfitting happens when the model becomes too specialized in the training data and performs poorly on new, unseen data. It's important to balance the model's ability to learn from the data without memorizing it - this is a foundational idea in machine learning.

</aside>

Now back to MNIST:

```python
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

```

An optimizer (`SGD`) is instantiated with a learning rate of 0.01 and a momentum value of 0.9. The learning rate controls the step size during weight updates, while momentum helps accelerate convergence by accumulating past gradients. The choice of optimizer and its hyperparameters is a trade-off between training speed and convergence accuracy. The model is compiled with the optimizer, specifying `"categorical_crossentropy"` as the loss function for multi-class classification and `"accuracy"` as the metric to evaluate during training.

<aside>
💡 
  
  **SGD (Stochastic Gradient Descent)** is an optimization algorithm used during training to update the model's parameters based on the computed gradients of the loss function. It iteratively adjusts the parameters to minimize the loss and find the optimal values. Think of training a model as trying to find the lowest point in a hilly landscape. SGD is like a hiker who takes small steps downhill, adjusting their position based on the slope of the terrain. The goal is to reach the bottom of the hill, which represents the optimal solution. By repeatedly taking steps in the direction that leads to a steeper descent, the hiker gradually converges to the lowest point. Similarly, SGD adjusts the model's parameters step by step, guided by the gradients of the loss function, in order to reach the optimal solution and minimize the loss.

</aside>

```python
history = model.fit(x_train, y_train, epochs=10, batch_size=32)

```

This line trains the model using the training data (`x_train` and `y_train`). The model is trained for 10 epochs, which represents the number of times the entire training dataset is passed through the model. The batch size of 32 indicates that the model's weights are updated after processing 32 samples at a time. The number of epochs and batch size are design choices that balance computational efficiency and model convergence.

```python
image = random.choice(x_test)
plt.imshow(image, cmap=plt.get_cmap("gray"))
plt.show()

```

A random image from the testing data (`x_test`) is selected and displayed using Matplotlib. The image is shown in grayscale, indicating that it is a grayscale image.

```python
image = (image.reshape((1, 28, 28, 1))).astype("float32") / 255.0

```

We can't forget to reshape and normalize the image as we did before. The selected image is reshaped to match the input shape of the model, with dimensions `(1, 28, 28, 1)`. This reshaping ensures that the image has the correct dimensions for the model's input. Additionally, the image is converted to the float32 data type and normalized by dividing it by 255.0 to ensure consistency with the preprocessing done during training.

```python
digit = np.argmax(model.predict(image)[0], axis=-1)
print("Prediction:", digit)

```

The model predicts the digit in the preprocessed image using `model.predict(image)`. The `predict` method returns the predicted probability distribution over the classes. The `argmax` function is then used to find the index with the highest probability, which corresponds to the predicted digit. Finally, the predicted digit is printed to the console.

And that’s it! We’re done. If you have any questions about this post, shoot me an email @ himilanoi@gmail.com 

Shout out to @svpino on Twitter for this idea. It’s been fun expounding on his code!
