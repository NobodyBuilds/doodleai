# DoodleAI 🎨

A CUDA accelerated neural network that recognizes hand-drawn doodles in real time.

DoodleAI is a custom neural network inference application built from scratch using C++ and CUDA. The model was trained using the custom CUDA neural network framework:

https://github.com/NobodyBuilds/cuda_neural_net

The trained model is deployed with GPU accelerated inference for fast doodle classification.

## Features

- Real-time doodle recognition
- CUDA accelerated neural network inference
- Custom neural network implementation from scratch
- Interactive drawing canvas
- 28×28 pixel input processing
- GPU based forward propagation
- Exported model weight loading
- ImGui interface

## Neural Network Architecture

The model uses a fully connected feed-forward neural network:
