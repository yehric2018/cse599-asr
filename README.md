# Automatic Speech Recognition with AudioMNIST
Final project for deep learning

## Abstract
For this project, I trained the a convolutional neural network to classify digits spoken in the AudioMNIST dataset. The result was 45% accuracy after 20 epochs, which could easily be improved by training for more epochs.

## Problem Statement
The goal is to be able to predict words being spoken in real-time. Ideally this would require using a RNN, where variable-length data points can continuously be streamed into the network to produce a prediction. For this project, I will start with a small vocabulary size of 10 words and see if I can create a model that predicts the word being spoken.

## Related work
The paper by Becker et al. that created the AudioMNIST dataset already includes a neural network architecture that feeds the entire raw audio signal into the network at once, using an architecture similar to an image classifier to predict which digit from 0-9 is being spoken. They produced a model that takes in the raw audio signals from the AudioMNIST dataset and predicts the spoken digit with over a 90% accuracy.

You can download the AudioMNIST dataset from the Github repository [here](https://github.com/soerenab/AudioMNIST).

The link to the paper by Becker et al. can be found [here](https://arxiv.org/abs/1807.03418).

## Methodology
### AudioNet1 - Convolutional Neural Network

### AudioNet2 - Recurrent Neural Network

## Experiments/Evaluation

## Results

## Examples

## Video
