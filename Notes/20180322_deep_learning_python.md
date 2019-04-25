# Deep Learning with Python

## <font color="red">What will you learn?</font>
You will learn:

### Theory [around 30 mins]

- Basics of Machine Learning
- What is Deep Learning
- What is Convolutional Neural Networks(CNN)
- What is supervised and unsupervised learning

### Practice [60+ mins]

- What is Python
- What is Keras library and how to use it
- How to build step by step Convolutional Neural Networks(CNN)
- What are differences in model results

## What is Keras?
<font color="red">Keras is a minimalist Python library for deep learning that can run on top of Theano or TensorFlow</font>. It was developed to make implementing deep learning models as fast and easy as possible for research and development. It runs on Python 2.7 or 3.5 and can seamlessly execute on GPUs and CPUs given the underlying frameworks. It is released under the permissive MIT license.

<b>Keras was developed and maintained by François Chollet, a Google engineer using four guiding principles</b>:
- Modularity: A model can be understood as a sequence or a graph alone. All the concerns of a deep learning model are discrete components that can be combined in arbitrary ways.
- Minimalism: The library provides just enough to achieve an outcome, no frills and maximizing readability.
- Extensibility: New components are intentionally easy to add and use within the framework, intended for researchers to trial and explore new ideas.
- Python: No separate model files with custom file formats. Everything is native Python.

## <font color="red">Install Keras in Anaconda Python</font>
### How to install Keras
Keras is relatively straightforward to install if you already have a working Python and SciPy environment. You can install Keras on Windows with the following commands.

#### Step 1: Install Anaconda (Python 3.6 version) Download
1. For Windows user, please download the <font style="color:red">*.exe</font> file.

<center><img src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/raw/master/anaconda_windows_installation.png"></img></center>

2. For Mac user, please download <font color="red">*.dmg</font> file
<img src="https://courses.edx.org/asset-v1:MITx+6.008.1x+3T2016+type@asset+block/02.PNG"></img>
Then install the dmg file according to the following steps:
<img src="https://unidata.github.io/online-python-training/images/conda.gif"></img>

#### Step 2: Update Anaconda
1. For windows users, please get into <font color="red">CMD</font>, and for mac users please open termminal

<img src="https://www.ibm.com/developerworks/community/blogs/jfp/resource/BLOGS_UPLOADED_IMAGES/anaconda.png"></img>
<img src="http://blog.teamtreehouse.com/wp-content/uploads/2012/09/Screen-Shot-2012-09-25-at-12.57.00-PM.png"></img>

Then, open Anaconda Prompt to type the following command(s)

```bash
conda update conda
conda update --all
```
<center><img src="http://alimanfoo.github.io/assets/2017-05-18-installing-python/capture-prompt-12.PNG"><img></center>


#### Step 2.1: Install CUDA(<font color="red">You can skip this command if you do not have GPU!</font>)
1. Choose your version depending on your Operating System
<center><img src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/raw/master/cuda8_windows7_local_installation.png"></img></center>

2. Choose your version depending on your Operating System. Membership registration is required.
<center><img src="https://github.com/antoniosehk/keras-tensorflow-windows-installation/raw/master/cuDNN_windows_download.png"></img></center>

3. Put your unzipped folder in C drive as follows:
```bash
C:\cudnn-8.0-windows10-x64-v5.1
```

4. Add cuDNN into Environment PATH Tutorial

Add the following path in your Environment. Subjected to changes in your installation path.
```bash
C:\cudnn-8.0-windows10-x64-v5.1\cuda\bin
```
Turn off all the prompts. Open a new Anaconda Prompt to type the following command(s)
```bash
echo %PATH%
```
You shall see that the new Environment PATH is there.

#### Step 3: Create an Anaconda environment with Python=3.5
Open Anaconda Prompt to type the following command(s)
```bash
conda create -n tensorflow python=3.5 numpy scipy matplotlib spyder
```

#### Step 4: Activate the environment
#### Step 5: Install TensorFlow-GPU-1.0.1, if you do not have GPU, you should install Tensorflow-CPU
Open Anaconda Prompt to type the following command(s)
```bash
# If you only want CPU version
pip install --ignore-installed --upgrade tensorflow
# If you have GPU
pip install --ignore-installed --upgrade tensorflow-gpu
```
Open Anaconda Prompt to type the following command(s)
```bash
activate tensorflow
```

#### Step 6: Install Keras
Open Anaconda Prompt to type the following command(s)
```bash
pip install keras
```

#### Step 7: Testing
Open Anaconda Prompt to type the following command(s)
```bash
activate tensorflow
python
```
<center><img src="http://gowrishankarnath.com/wp-content/uploads/2015/10/12.png"></img></center>

## <font color="red">Before we start, let's recall some deep learning key terms</font>
1. <font color="red">Deep learning</font></br>
As defined before, deep learning is the process of applying deep neural network technologies to solve problems. Deep neural networks are neural networks with <b>one hidden layer minimum (see below)</b>. Like data mining, deep learning refers to a process, which employs deep neural network architectures, which are particular types of machine learning algorithms.

2. <font color="red">Artificial Neural Networks (ANNs)</font></br>
The machine learning architecture originally inspired by the <b>biological brain (particularly the neuron)</b> by which deep learning is carried out. Actually, ANNs alone (the non-deep variety) have been around for a very long time, and have been able to solve certain types of problems historically. However, comparatively recently, neural network architectures were devised which included layers of hidden neurons (beyond simply the input and output layers), and this added level of complexity is what enables deep learning, and provides a more powerful set of problem-solving tools.</br>
ANNs actually vary in their architectures quite considerably, and therefore there is no definitive neural network definition. The 2 generally-cited characteristics of all ANNs are the possession of adaptive weight sets, and the capability of approximating non-linear functions of the inputs to neurons.

3. <font color="red">Perceptron</font></br>
  A perceptron is a simple linear binary classifier. Perceptrons take inputs and associated weights (representing relative input importance), and combine them to produce an output, which is then used for classification. Perceptrons have been around a long time, with early implementations dating back to the 1950s, the first of which were involved in early ANN implementations.

4. <font color="red">Multilayer Perceptron (MLP)</font></br>
A multilayer perceptron (MLP) is the implementation of several fully adjacently-connected layers of perceptrons, forming a simple feedforward neural network (see below). This multilayer perceptron has the additional benefit of nonlinear activation functions, which single perceptrons do not possess.

5. <font color="red">Epoch vs Batch Size vs Iterations</font></br>
To find out the difference between these terms you need to know some of the machine learning terms like Gradient Descent to help you better understand. We need terminologies like epochs, batch size, iterations only when the data is too big which happens all the time in machine learning and we can’t pass all the data to the computer at once. So, to overcome this problem we need to divide the data into smaller sizes and give it to our computer one by one and update the weights of the neural networks at the end of every step to fit it to the data given.
  - Epochs. One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
  - Batch Size. Total number of training examples present in a single batch.
  - Iterations. Iterations is the number of batches needed to complete one epoch.

6. <font color="red">Gradient Descent</font></br>
Here is a short summary on Gradient Descent: <i>"It is an iterative optimization algorithm used in machine learning to find the best results (minima of a curve)"</i>.

<b>Gradient</b> means the rate of inclination or declination of a slope. <b>Descent</b> means the instance of descending. The algorithm is <font color="red">iterative</font> means that we need to get the results <b>multiple times</b> to get the most optimal result. The iterative quality of the gradient descent helps a under-fitted graph to make the graph fit optimally to the data. The following examples are the illustration of costs and outputs.
<center><img src="https://cdn-images-1.medium.com/max/1600/1*pwPIG-GWHyaPVMVGG5OhAQ.gif"><img></center>

The Gradient descent has a parameter called <font color="red">learning rate</font>. As you can see above (left), initially the steps are <font color="red">bigger</font> that means the learning rate is higher and as the point goes down the learning rate becomes more smaller by the shorter size of steps. Also,the Cost Function is decreasing or the cost is decreasing. Sometimes you might see people saying that the Loss Function is decreasing or the loss is decreasing, both Cost and Loss represent same thing (btw it is a good thing that our loss/cost is decreasing).

## Let's do it! Cifar-10 Classification using Keras

You can do some cool thing after you understand deep learning, for example, run object recognition algorithm on FPGA ([EagleGo HD](http://www.v3best.com/products/SDVision/EagleGo/EagleGo-HD.html), [Demo Video](http://v.youku.com/v_show/id_XMTcxOTU5MzIwNA==.html?spm=a2h0k.8191407.0.0&from=s1.8-1-1.2))

<center><img src="https://vthumb.ykimg.com/0541040857D2DA436A0A41046C28DDCC"></img></center>

### What is Cifar-10
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data set consists of 60000 32×32 color images in 10 classes (see Table below), with 6000 images per class. There are 50000 training images and 10000 test images.

<center><img src="http://karpathy.github.io/assets/cifar_preview.png"></img></center>

Recognizing photos from the cifar-10 collection is one of the most common problems in the today’s  world of machine learning. <font style="color:red"><b>In this course, we will show you – step by step – how to build multi-layer artificial neural networks that will recognize images from a cifar-10  set with an accuracy of about 80% and visualize it.</b></font>


### Hands-on practice

> <font color="red">Target: Building 4 and 6-layer Convolutional Neural Networks</font>

<i>Keras is an open source neural network Python library which can run on top of other machine learning libraries like TensorFlow, CNTK or Theano. It allows for an easy and fast prototyping, supports convolutional, recurrent neural networks and a combination of the two.</i>

<font color="red">In the following part, we will teach you step-by-step:</font>
- At the beginning we will learn what Keras is, deep learning, what we will learn, and briefly about the cifar-10 collection.
- Then step by step, we will build a 4 and 6 layer neural network along with its visualization, resulting in % accuracy of classification with graphical interpretation.
- Finally, we will see the results and compare the two networks in terms of the accuracy and speed of training for each epoch.

##### Step 0: Open <font color="red">Jupyter Notebook</font>
```
jupyter notebook
```

##### Step 1: First of all, we will be defining all of the classes and functions we will need
```python
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras import backend as K

# Loading the CIFAR-10 datasets
from keras.datasets import cifar10
```
As a good practice suggests, we need to declare our variables:
- batch_size – the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you’ll need
- num_classes – number of cifar-10 data set classes
- one epoch – one forward pass and one backward pass of all the training examples

```Python
# Declare variables
batch_size = 32
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
num_classes = 10 #
epochs = 100 # repeat 100 times
```

##### Step 2: Next, we can load the CIFAR-10 data set.
```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train - training data(images), y_train - labels(digits)
```

Print figure with 10 random images from cifar dataset
```python
# Print figure with 10 random images from each

fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:]==i)[0]
    features_idx = x_train[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = features_idx[img_num,::]
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()
```

Running the code create a 5×2 plot of images and show examples from each class.
<center><img src="https://blog.plon.io/wp-content/uploads/2017/08/cifar.png"></img></center>

The pixel values are in the range of 0 to 255 for each of the <b>red, green and blue</b> channels. It’s good practice to work with normalized data. Because the input values are well understood, <b>we can easily normalize to the range 0 to 1 by dividing each value by the maximum observation which is 255</b>.

##### Step 3: Data pre-processing.
Note, the data is loaded as integers, so we must cast it to floating point values in order to perform the division.
```python
# Convert and pre-processing

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255
```

The output variables are defined as a vector of integers from 0 to 1 for each class.

##### Step 4: Let’s start by defining a simple CNN model.
We will use a model with four convolutional layers followed by max pooling and a flattening out of the network to fully connected layers to make predictions:
- Convolutional input layer, 32 feature maps with a size of 3×3, a rectifier activation function
- Convolutional input layer, 32 feature maps with a size of 3×3, a rectifier activation function
- Max Pool layer with size 2×2
- Dropout set to 25%
- Convolutional input layer, 64 feature maps with a size of 3×3, a rectifier activation function
- Convolutional input layer, 64 feature maps with a size of 3×3, a rectifier activation function
- Max Pool layer with size 2×2
- Dropout set to 25%
- Flatten layer
- Fully connected layer with 512 units and a rectifier activation function
- Dropout set to 50%
- Fully connected output layer with 10 units and a softmax activation function

A logarithmic loss function is used with the stochastic gradient descent optimization algorithm configured with a large momentum and weight decay start with a learning rate of 0.1.


##### Step 5: Then we can fit this model with 100 epochs and a batch size of 32.
```python
def base_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9 nesterov=True)

    # Train model

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

cnn_n = base_model()
cnn_n.summary()

# Fit model

cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),shuffle=True)
```

The second variant for 6-layer model (<font color="red">compare it with the 4-layer model</font>):
- Convolutional input layer, 32 feature maps with a size of 3×3, a rectifier activation function
- Dropout set to 20%
- Convolutional input layer, 32 feature maps with a size of 3×3, a rectifier activation function
- Max Pool layer with size 2×2
- Convolutional input layer, 64 feature maps with a size of 3×3, a rectifier activation function
- Dropout set to 20%
- Convolutional input layer, 64 feature maps with a size of 3×3, a rectifier activation function
- Max Pool layer with size 2×2
- Convolutional input layer, 128 feature maps with a size of 3×3, a rectifier activation function
- Dropout set to 20%
- Convolutional input layer, 128 feature maps with a size of 3×3, a rectifier activation function
- Max Pool layer with size 2×2
- Flatten layer
- Dropout set to 20%
- Fully connected layer with 1024 units and a rectifier activation function and a weight constraint of max norm set to 3
- Dropout set to 20%
- Fully connected output layer with 10 units and a softmax activation function


```python
def base_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
    model.add(Dropout(0.2))

    model.add(Conv2D(32,(3,3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
```

##### Step 6: visualization

In this part, we can visualize model structure. For this problem, we can use a library for Keras for investigating architectures and parameters of sequential models by Piotr Migdał.
```python
# Vizualizing model structure

sequential_model_to_ascii_printout(cnn_n)
```

First variant for 4-layer:

![](https://blog.plon.io/wp-content/uploads/2017/08/4.jpg)

Second variant for 6-layer:

![](https://blog.plon.io/wp-content/uploads/2017/08/6.jpg)

##### Step 7: let's see the training loss and accuracy
After training process, we can see loss and accuracy on plots using the code below:
```python
# Plots for training and testing process: loss and accuracy

plt.figure(0)
plt.plot(cnn.history['acc'],'r')
plt.plot(cnn.history['val_acc'],'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])


plt.figure(1)
plt.plot(cnn.history['loss'],'r')
plt.plot(cnn.history['val_loss'],'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])


plt.show()
```

4-layer:
![](https://blog.plon.io/wp-content/uploads/2017/08/6a.png)
6-layer:
![](https://blog.plon.io/wp-content/uploads/2017/08/4a.png)

```python
scores = cnn_n.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

Running this example prints the classification accuracy and loss on the training and test datasets each epoch.

##### Step 8: visualize the confusion matrix
After that, we can print [<font color="blue">confusion matrix</font>](https://en.wikipedia.org/wiki/Confusion_matrix) for our example with graphical interpretation.

> Confusion matrix – also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning one (in unsupervised learning it is usually called a matching matrix). Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa).

```python
# Confusion matrix result

from sklearn.metrics import classification_report, confusion_matrix
Y_pred = cnn_n.predict(x_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)

for ix in range(10):
    print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(cm)

# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd


df_cm = pd.DataFrame(cm, range(10),
                  range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()
```

4-layer confusion matrix and visualizing:
```python
[[599   5  74  98  55   14  12   9 117  17]
 [ 16 738  12  65   9   26   7   6  40  81]
 [ 31   0 523 168 136   86  33  14   9   0]
 [ 10   1  31 652  90  175  19  15   5   2]
 [  6   0  34 132 717   55  16  31   9   0]
 [  5   1  17 233  53  661  10  15   4   1]
 [  2   1  39 157 105   48 637   3   7   1]
 [  6   0  14  97 103   96   5 637   5   1]
 [ 41   7  28  84  19   18   6   4 783  10]
 [ 25  28   8  77  29   27   5  19  59 723]]
```

<center><img src="https://blog.plon.io/wp-content/uploads/2017/08/cm4.png"></img></center>

6-layer confusion matrix and visualizing:
```python
[[736  11  54  45  30  14  15   9  61  25]
 [ 10 839   6  38   3  13   7   5  22  57]
 [ 47   2 566  96 145  65  51  17   7   4]
 [ 23   6  56 570  97 140  57  29  12  10]
 [ 16   2  52  80 700  55  25  64   3   3]
 [ 10   1  64 211  59 582  24  39   6   4]
 [  4   3  42 114 121  40 650  13   5   8]
 [ 14   1  40  57  69  68  11 723   3  14]
 [ 93  32  26  37  16  15   6   2 752  21]
 [ 34  83   8  42  12  21   6  21  25 748]]
```

<center><img src="https://blog.plon.io/wp-content/uploads/2017/08/cm6.png"></img></center>

Comparison Accuracy [%] between 4-layer and 6-layer CNN
As we can see in the chart below, the best accuracy for 4-layer CNN is for epochs between 20-50. For 6-layer CNN is for epochs between 10-20 epochs.

<center><img src="https://blog.plon.io/wp-content/uploads/2017/08/acc.png"></img></center>

Comparison time of learning process between 4-layer and 6-layer CNN
As we can see in the chart below, the neural network training time is considerably longer for a 6-layer network.

<center><img src="https://blog.plon.io/wp-content/uploads/2017/08/lp.png"></img></center>

## Code
You can run the project in your browser or download it from [GitHub](https://github.com/gujiuxiang/dl_tutorial/tree/master).
