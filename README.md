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
Input Layer
784 neurons (28×28 pixels)

    ↓

512 neurons

    ↓

256 neurons

    ↓

128 neurons

    ↓

64 neurons

    ↓

Output Layer
20 classes


## How It Works

1. User draws a doodle on the canvas.
2. The drawing is converted into a 28×28 pixel representation.
3. Pixel values are normalized.
4. CUDA executes the neural network forward pass.
5. The output layer predicts the doodle category.

## Technologies

- C++
- CUDA C++
- CUDA Runtime API
- OpenGL
- GLFW
- Dear ImGui
- Custom Neural Network Engine

## Model Training

The model was trained using:

https://github.com/NobodyBuilds/cuda_neural_net

Training output is exported into:


w2/

├── weights.txt
├── bias.txt
└── labels.txt


These files are loaded during runtime for inference.

## Build

Requirements:

- NVIDIA GPU
- CUDA Toolkit
- C++ compiler
- OpenGL + GLFW dependencies

Run:


compile.bat


## Project Structure


doodleai/

├── model.cu # CUDA neural network implementation
├── compile.bat # Build script
├── w2/ # Model parameters
└── README.md


## About

DoodleAI is an experiment in building a complete deep learning pipeline without relying on high-level frameworks, focusing on understanding neural networks, GPU acceleration, and model deployment.
