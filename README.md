# MyDeepLearning

## Repo Intro
This repo is to construct a DL library for learning and testing some classic projects that includes:

  1, CNN, RNN model in pure Numpy, with all BP gredient calculation included.
  
  2, Some classic model in tensorflow, include ResNet, SqueezeNet and more.
  
  3, Some classic project script include Image Captioning, Image Fooling, Deep Dream, style transfer and more.
  
  4, Multiple GAN implemented in Tensorflow.
  
  5, Q-Learning implement in Numpy and Gym.
  
## File structure

all files in root content represents a mini project, introduced in the section below. All files in lib has been covered by these mini projects.

    --lib

      -- classifiers
        "all classifier models in both pure numpy or tensorflow"

      -- solvers
         "Trainer for both TF and pure numpy model"

      -- layers
         "layers for pure numpy model"

      -- utils
         "some assistant files"

      -- datasets
         "CIFAR-1O, MNIST dataset"
       
## Mini projects

The projects is listed here in a recommended reading sequence:

### PART I: pure Numpy project

### 1, LinearClassifer.py

It uses lib/model/linear_classifer.py to construct a linear classifer model to classify CIFAR-10 images. It's implemented in pure Numpy.

### 3, FCN.py
It uses lib/model/fc_net.py to construct a fullt connected neural network model to classify CIFAR-10 images. It's implemented in pure Numpy.

### 2, ThreeLayerCNN.py
It uses lib/model/cnn.py to construct a cnn model to classify CIFAR-10 images. It's implemented in pure Numpy.

### 4, ImageCaptioning.py
It uses lib/model/rnn.py to contruct a vanilla rnn or lstm model to construct image captioning for microsoft coco data, the input feature is not from a CNN but from dataset directly. It's implemented in pure Numpy.

Please check this blog for details: 

### PART II: Tensorflow project

### 5, TFCNN.py
It build and test a CNN classifer in tensorflow.

### 6, SaliencyMap.py
It uses lib/model/squeezeNet.py with well trained parameters to show the saliencymap of images in CIFAR-10.

### 7, DeepDream.py
It uses lib/model/squeezeNet.py with well trained parameters to construct deepdream images based on images in CIFAR-10, for details about deepdream, please refer to the official github repo: https://github.com/google/deepdream

### 8, ImageFooling.py
It construct a model that fool well-trained squeezenet to label an image with wrong label.

Please check this blog for details of these 3 prjects above: 

### 9, StyleTransfer.py
It uses lib/model/squeezeNet.py with well trained parameters to extract features from both content and style image to construct a new content image with style image style.

Please check this blog for details: 

### 10. TFGAN.py
It implemented DCGAN and WGAN in tensorflow to generate images based on both MNIST and CIFAR-10.

## How to use

    1, Download the repo to local, a star to the repo is appreciated.
    
    2, make sure Python3 is installled in your local env, a virtual env is recommended.
    
    3, pip3 install -r requirements
    
    4, go to lib/datasets, manually run get_datasets.sh and get_3rd_data.sh to download dataset to local.
    
    5, feel free to run any project under root.

## More to do

I will implemented the following extra more model and project in very recent time.

[] DQN

[] RCNN

[] ResNet

## Special Thanks

This repo is heavily referenced with the assignment of cs231n. Special thanks to the great staff who gives us such a great online course materails.

Course link: http://cs231n.stanford.edu/

## Contact Me

Email: nick_fandingwei@outlook.com

Twitter: https://twitter.com/nick_fandingwei

For Chinese user, zhihu is the fastest way to get response from me: https://www.zhihu.com/people/NickWey

You can also check my tech blog for more: http://nickiwei.github.io/
