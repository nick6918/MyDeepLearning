# MyDeepLearning

## Repo Intro
This repo is to construct a DL library for learning and testing some classic projects that include:

  1, CNN, RNN(LSTM) model in pure Numpy, with all BP gradients calculation included.
  
  2, Some classic model in tensorflow, including ResNet, SqueezeNet and more.
  
  3, Some classic project scripts including Image Captioning, Image Fooling, Deep Dream, style transfer and more.
  
  4, Image Localization, Dectection and Segmentation using Fast(er) R-CNN. (Coming very soon)
  
  5, Multiple GAN(DCGAN, LSGAN, WGAN and more) implementation in Tensorflow.
  
  6, Q-Learning implementation in Numpy and Gym. (Coming very soon)
  
## File structure

All files in root directory represent a mini project, introduced in the section below. All files in lib has been covered by these mini projects.

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

### 1, LinearClassifier.py

It uses lib/model/linear_classifier.py to construct a linear classifer model to classify CIFAR-10 images. It's implemented in pure Numpy.

### 3, FCN.py
It uses lib/model/fc_net.py to construct a fully connected neural network model to classify CIFAR-10 images. It's implemented in pure Numpy.

[FCN Relevant blog(In Chinese)](http://nickiwei.github.io/2017/09/01/CNN%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C%E7%9A%84Python%E5%AE%9E%E7%8E%B0I-FCN%E5%85%A8%E8%BF%9E%E6%8E%A5%E7%BD%91%E7%BB%9C/)

### 2, ThreeLayerCNN.py
It uses lib/model/cnn.py to construct a cnn model to classify CIFAR-10 images. It's implemented in pure Numpy.

[CNN Relevant blog(In Chinese)](http://nickiwei.github.io/2017/09/07/CNN%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C%E7%9A%84Python%E5%AE%9E%E7%8E%B0III-CNN%E5%AE%9E%E7%8E%B0/)

### 4, ImageCaptioning.py
It uses lib/model/rnn.py to contruct a vanilla rnn or lstm model to construct image captioning for microsoft coco data, the input feature is not from a CNN but from dataset directly. It's implemented in pure Numpy.

[Image Captioning Relevant blog(In Chinese)](http://nickiwei.github.io/2017/09/16/RNN,-LSTM%E4%B8%8EImageCaptioning%E5%8E%9F%E7%90%86%E5%8F%8APython%E5%AE%9E%E7%8E%B0/)

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

[Saliency Map/DeepDream/ImageFooling Relevant Blog(In Chinese)](http://nickiwei.github.io/2017/09/19/%E4%BB%8ESaliency-Map%E5%88%B0Gredient-Ascent(%E5%9F%BA%E4%BA%8ETensorFlow%E5%AE%9E%E7%8E%B0)/)

### 9, StyleTransfer.py
It uses lib/model/squeezeNet.py with well trained parameters to extract features from both content and style image to construct a new content image with style image style.

[Style Transfer Relevant Blog(In Chinese)](http://nickiwei.github.io/2017/09/24/Style-Transfer-%E5%9F%BA%E4%BA%8ETensorFlow%E5%AE%9E%E7%8E%B0/)

### 10. TFGAN.py
It implemented DCGAN and WGAN in tensorflow to generate images based on both MNIST and CIFAR-10.

[GAN relevant blog(In Chinese)](http://nickiwei.github.io/2017/09/25/GAN%E7%9A%84%E5%8E%9F%E7%90%86%E5%8F%8ATensorFlow%E5%AE%9E%E7%8E%B0/)

## How to use

    1, Download the repo to local, a star to the repo is appreciated.
    
    2, make sure Python3 is installled in your local env, a virtual env is recommended.
    
    3, pip3 install -r requirements
    
    4, go to lib/datasets, manually run get_datasets.sh and get_3rd_data.sh to download dataset to local.
    
    5, feel free to run any project under root.

## More to do

I will upload the following extra models and projects in very recent time.

[] DQN

[] RCNN Family

[] ResNet

[] Policy Gradient(Reenforcement learning)

[] VAE(Generative model)

## Special Thanks

This repo is heavily referenced with the assignment of Stanford cs231n. Special thanks to the great staff who give us such a great online course materials.

Course link: http://cs231n.stanford.edu/

## Contact Me

Email: nick_fandingwei@outlook.com

Twitter: https://twitter.com/nick_fandingwei

For Chinese user, zhihu is the fastest way to get response from me: https://www.zhihu.com/people/NickWey

You can also check my tech blog for more: http://nickiwei.github.io/

Consider to follow me on Zhihu, Twitter and Github, thanks!
