# Computer Vision
In this Computer Vision Repository I have upload five projects Related to Computer vision which I have done. 
#### The following project is :
1.	Face Recognition with Emotion capture using Open CV, 
2.	Real Time Object Recognition using Single Shot Detection (SSD). 
3.	Image Creation with Deep Convolutional GANs. 
4.	MNIST-Handwritten-Digits-Recognition-Using-Deep-Neural-Network 
5.	CIFAR 10 Object Recognition in Images Using Deep Convolutional Neural Network.
#### My LinkedIn Profile:
https://www.linkedin.com/in/amiralicheema/

#### My Medium Blog
https://medium.com/machine-learning-researcher

These are project which I have done.
## Project 1: Face Recognition with Emotion capture using Open CV
In this project I made face Recognition using open CV Libraries. Basically these face Detection projects detect your face with Green Rectangle eyes with Blue Detection and Red Rectangle detect Smile face which your face make a smile. I use haarcascade_eye for detecting eyes, haarcascade_frontalface_default for face detection and haarcascade_smile for detect smile face.
#### Source data:
https://github.com/opencv/opencv/tree/master/data/haarcascades
#### References:
1.	Rapid Object Detection using a Boosted Cascade of Simple Features Paul Viola Michael Jones viola@merl.com mjones@crl.dec.com Mitsubishi Electric Research Labs Compaq CRL 201 Broadway, 8th FL One Cambridge Center Cambridge, MA 02139 Cambridge, MA 02142
2.	A Discriminative Feature Learning Approach for Deep Face Recognition Yandong WenKaipeng Zhang Zhifeng LiEmail author Yu Qia

## Project 2: Real Time Object Dectection using Single Shot Dectection(SSD)
In this project I made detection system which detects the real time objects. I use Single Shot Detection (SSD) and Open CV in this project. SSD (Single Shot MultiBox Detector) is a popular algorithm in object detection. It's generally faster than Faster RCNN. I use randomly 2 second video and apply our model on it it’s detect the everything (objects) in video with rectangle and specify with name.
#### References:
1.	SSD: Single Shot Multi Box Detector Wei Liu1, Dragomir Anguelov2, Dumitru Erhan3, Christian Szegedy3, Scott Reed4, Cheng-Yang Fu1, Alexander C. Berg1 1UNC Chapel Hill 2Zoox Inc. 3Google Inc. 4University of Michigan, Ann-Arbor 1 wliu@cs.unc.edu, 2 drago@zoox.com, 3 {dumitru,szegedy}@google.com, 4reedscot@umich.edu, 1 {cyfu,aberg}@cs.unc.edu
2.	Object recognition and detection with deep learning for autonomous driving applications Ayşegül Uçar, Yakup Demir, Cüneyt Güzeliş
3.	Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
4.	Robust Real-time Object Detection Paul Viola Michael J. Jones

## Project 3: Image Creation with Deep Convolutional GANs
In this project create a fake images from real image using Deep Convolutional GANs (Generative Adversarial Networks) .For creating new images from real image we use CIFAR-10 dataset. In this project we pytorch library and build a model (Generator & Discriminator) from scratch. During training a model I run only one number of epoch because I am student I can’t have complete source to run this model for maximum  number of epochs because It’ required high level system to run this model which I can’t offered. I run only one number of epochs and achieved more than 80% accuracy.
#### Source:
https://www.cs.toronto.edu/~kriz/cifar.html
#### References:
1.	Generative Adversarial Nets Authors: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,Sherjil Ozair, Aaron Courville, Yoshua Bengio
2.	Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks Alec Radford, Luke Metz, Soumith Chintala
3.	Deep Generative Image Models using a ￼Laplacian Pyramid of Adversarial Networks Authors: Emily L. Denton, Soumith Chintala, arthur szlam, Rob Fergus,
4.	Conditional Generative Adversarial Nets Authors: Mehdi Mirza, Simon Osindero

## Project 4: MNIST: Handwritten Digits Recognition Using Deep Neural Network
Used deep neural network model trained on MNIST dataset to classify images of handwritten digits
1.	Generates automatic predictions for images.
2.	Trained the model on the MNIST (Modified National Institute of Standards and Technology) dataset, has a training set of 60,000 examples, and a test set of 10,000 examples.
3.	Achieved an accuracy of 0.9871 after training for 5 epochs.
4.	Achieved a loss value of 0.0404 after training for 5 epochs.
#### Dependencies
Used Keras with Tensorflow backend for the code. Everything is implemented in the Jupyter notebook which will hopefully make it easier to understand the code.
1.	Tensoflow
2.	Keras
3.	Numpy
4.	Matplotlib
#### References
1.	Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
2.	Official site for MNIST dataset: http://yann.lecun.com/exdb/mnist/
3.	Yann LeCun, Professor, The Courant Institute of Mathematical Sciences, New York University
4.	Corinna Cortes, Research Scientist, Google Labs, New York
5.	Christopher J.C. Burges, Microsoft Research, Redmond

## Project 5: CIFAR-10: Object Recognition in Images Using Deep Convolutional Neural Network (DCNN)
In this project, we used deep convolutional neural network model trained on CIFAR-10 (Canadian Institute For Advanced Research) dataset to recognize multiple objects present in various images.
1.	Recognize 10 different objects which are airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships and trucks.
2.	Trained the model on the CIFAR-10 (Canadian Institute For Advanced Research) dataset. There are 50000 training images and 10000 test images.
3.	Achieved an accuracy of 0.8005 after training for 100 epochs.
4.	Achieved a loss value of 0.5659 after training for 100 epochs.
5.	Achieved validation accuracy of 0.8147 after training for 100 epochs.
6.	Achieved a validation loss value of 0.5372 after training for 100 epochs.
#### Dependencies
Used Keras with Tensorflow backend for the code. CIFAR-10 is a common benchmark in machine learning for image recognition.
Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on GPU.
1.	Tensoflow
2.	Keras
3.	GPU
4.	Matplotlib
#### References
1.	Convolutional Deep Belief Networks on CIFAR-10, Alex Krizhevsky, CS department, University of Toronto
2.	Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, April 8, 2009
3.	Official site for CIFAR-10 dataset: Computer Science department, University of Toronto (https://www.cs.toronto.edu/~kriz/cifar.html)
4.	Exploding, Vainishing Gradient descent / deeplearning.ai — Andrew Ng.








