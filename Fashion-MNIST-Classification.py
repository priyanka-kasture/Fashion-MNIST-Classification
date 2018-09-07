'''
Fashion-MNIST Classification

Author: Priyanka Kasture | pkasture2010@gmail.com
Language: Python 3.6
Dataset: Fashion MNIST Database
Environment: Spyder (Python 3.6)
Important Dependancies: Tensorflow, Keras, Numpy, Matplotlib
'''

# Importing Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras

# Other Dependancies
import numpy as np
import matplotlib.pyplot as plt

# Our Dataset - Fashion MNIST (https://github.com/zalandoresearch/fashion-mnist)
fashion_mnist = keras.datasets.fashion_mnist

''' 
Each example in the Fashion MNIST dataset is a 28x28 grayscale image.
The images are associated with a label from 10 classes.
Classes: t-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.
'''

'''
Loading the Data, and splitting it into training and testing sets.
The train_images and train_labels arrays are the training set — that is, the data, the model uses to learn.
The model is tested against the test set, the test_images, and test_labels arrays.
'''
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Shapes of the Training Set
print("Training set (images) shape: {shape}".format(shape=train_images.shape))
print("Training set (labels) shape: {shape}".format(shape=train_labels.shape))

'''
Shows that there are 60,000 images in the training set, with each image represented as 28 x 28 pixels.
Likewise, there are 60,000 labels in the training set. Each label is an integer between 0 and 9.
'''

# Shapes of the Testing Set
print("Test set (images) shape: {shape}".format(shape=test_images.shape))
print("Test set (labels) shape: {shape}".format(shape=test_labels.shape))

'''
Shows that there are 10,000 images in the testing set, with each image represented as 28 x 28 pixels.
Likewise, there are 10,000 labels in the testing set. Each label is an integer between 0 and 9.
'''


'''
Each image is mapped to a single label.
Since the class names are not included with the dataset, we store them here, to use later, when plotting the images.
'''
Categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''
Label 	Class
0 	T-shirt/top
1 	Trouser
2 	Pullover
3 	Dress
4 	Coat
5 	Sandal
6 	Shirt
7 	Sneaker
8 	Bag
9 	Ankle boot
'''

''' Visualizing a random training image. '''
plt.figure()
plt.imshow(train_images[7])
plt.colorbar()
plt.grid(True)

'''
If you inspect this image, you will see that the pixel values fall in the range of 0 to 255.
We will therefore scale these values to a range of 0 to 1 before feeding to the neural network model.
For this, the image components are to be typecasted to 'float' type, and are to be divided by 255.
'''
train_images = train_images/255.0
test_images = test_images/255.0

'''
For verification that our image data has been correctly mapped to a label, we will
display the first 25 images from the training set and display the class name below each image.
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(Categories[train_labels[i]])

    
'''
Now, we will build the model and set up the layers of the Neural Network.

The 1st layer: tf.keras.layers.Flatten
It transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.

The 2nd and 3rd layers: tf.keras.layers.Dense
They are densely-connected, or fully-connected, neural layers. 
The first Dense layer has 128 nodes (or neurons). 
The second (and last) layer is a 10-node softmax layer — this returns an array of 10 probability scores that sum to 1. 
Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
    

'''
Now we will be compiling the Model.

Optimizer — This is how the model is updated, based on the data it sees and its loss function.

Loss function — This measures how accurate the model is during training. 
We want to minimize this function to "steer" the model in the right direction.

Metrics — Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

'''
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


''' 
To start training, we call the model.fit() method.
We want the model to "fit" to the training data.
'''
model.fit(train_images, train_labels, epochs=10)
# Training Accuracy is between 90% - 92%

''' Now, we will compare how the model performs on the test dataset. '''
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# Testing Accuracy is between 86% - 89%
'''
The accuracy on the test dataset is a little less than the accuracy on the training dataset. 
That means the model is overfitting.
'''

''' 
Now, we will be feeding test images.
The neural network is expected to correctly predict the corresponding label.
'''
predictions = model.predict(test_images)

''' Checking the prediction for the 0th (1st) test image. '''
predictions[0]

''' 
The output is an array of values (probabilities) that sum up to 1. 
The first value in the array is ~ 2.59e-11.
It implies that there is 2.59e-11 probability that the object in the test image belongs to 1st Category.
The actual label/category of the test image is the one with the hightest probability.
Using the argmax function we find out the maximum value (probability) in the array.
'''
np.argmax(predictions[0])
''' 
The predicted output is 9.
That means, our model has predicted that this particular test image belongs to 9th Category.
'''
'''
To verify whether the object in the test image really belongs to 9th Category, we check it's test label.
'''
test_labels[0]
'''
The output is 9.
That is ====> Predicted Outut of the Model = Actual Output Stored in the Testing Dataset.
This implies that our model has correctly predicted the category of the object in the test image.
'''

''' To plot the image, the predicted labels, and the actual labels, two functions are written as follows. '''
# Plotting the Image, the predicted label, the actual label
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(Categories[predicted_label],
                                100*np.max(predictions_array),
                                Categories[true_label]),
                                color=color)
# Plotting the Graph
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

''' 
Now we plot the first X test images, their predicted label, and the true label.
Correct predictions are plotted in blue, incorrect predictions in red.
The model is accurate to a good extent, and therefore, more of blue will be seen.
'''
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
 
    
"""
Here, we predicted the labels of a batch of images.
But, we can also use the trained model to make a prediction about a single image. 
********** IT IS OPTIONAL **********
The code for the same is as follows:

# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), Categories, rotation=45)

np.argmax(predictions_single[0])
"""
