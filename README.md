# Duckietown Imitation Learning

## Overview

Author: Yang Shaohui (shyang@ethz.ch)

Collaborator: Wang Tianlu ()

Computing Source(GPU) Provider: Zilly Julian (jzilly@ethz.ch)

Course: [Autonomous Mobility on Demand: From Car to Fleet](http://www.vvz.ethz.ch/lerneinheitPre.do?semkez=2017W&lerneinheitId=119019&lang=en)

Project: Deep Learning in [Duckietown](http://book.duckietown.org/master/duckiebook/)

Place: ETH Zurich

Project Duration: Fall 2017, 2017.12.26 - 2018.01.15 to be exact

Brief Introduction: The code in this repo will train a [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) under the framework of [Caffe](http://caffe.berkeleyvision.org/) to substitute the traditional controller on the Duckiebot for lane following.

Extra Module: [Movidius™ Neural Compute Stick](https://movidius.github.io/ncsdk/)

Achievements: [Video](https://www.youtube.com/watch?v=FCP8Ndoxae0) is already online. Report will be soon out.

Similar Project: NVIDIA [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

## Code Organization

deploy.prototxt: file needed to deploy the pre-trained caffe model. The input is 1x1x80x160 and output is a scalar.

duckie.graph: final graph that can be loaded to the Neural Stick.

duckie_model_iter_10000.caffemodel: caffe model file that contains the pre-defined CNN structure and parameters for each layer.

solver.prototxt: specify the hyper parameters used in training. I used a gradient descent optimizer.

test.h5: 1720 testing data points. Each point consists of one image with 80x160x1 and one scalar label.

test.txt: for the train_val.prototxt to read test.h5

train.h5: 4836 training data points. Each point consists of one image with 80x160x1 and one scalar label.

train.txt: for the train_val.prototxt to read train.h5

train_val.prototxt: specify the structure of the CNN and how to initialize the parameters.

demo.py: two demo functinos inside. One runs on computers using the pre-trained caffe model while the other runs on the stick using the pre-trained graph file.

sample_images: 10 real world images.

sample_omega.csv: 10 real outputs.

prediction_omega.csv: 10 prediction outputs which are the results from either function of demo.py.

## About Given Data

The output omega should be postive if the car should turn left and vice versa. The training data concentrates on the left turning ones only but in reality, right turning also works fine.
Actually more work can be done with a better training dataset.

## Installation Guide

All code here are supposed to run in a Linux machine. The demo.py will require [caffe](https://github.com/BVLC/caffe) and [ncsdk](https://movidius.github.io/ncsdk/install.html) installed.

If you have a Duckiebot and want to try out the demo with Duckietown software, try this [link](https://github.com/duckietown/duckuments/blob/devel-super-learning-jan15/docs/atoms_20_setup_and_demo/30_demos/15_imitation_learning.md).

## More

How to train your own caffe model?

  - /path-to-caffe-folder/build/tools/caffe train -solver solver.prototxt -gpu 0 (if you have a gpu)

How to convert a caffe model to a movidius graph file?

  - mvNCCompile deploy.prototxt -w duckie_model_iter_10000.caffemodel -s 12 -o GRAPH_FILE_NAME