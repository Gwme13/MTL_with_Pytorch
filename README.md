# Investigating Multi-Task Learning (MTL) Paradigms with PyTorch

This repository presents two Jupyter notebooks designed to explore and compare various Multi-Task Learning (MTL) strategies against the conventional Single-Task Learning (STL) approach. The primary goal is to demonstrate how MTL can be employed to train models on multiple tasks simultaneously.

## Core Learning Concepts Explored

Throughout both notebooks, the following learning paradigms are implemented and evaluated:

1.  **Single-Task Learning (STL):** This traditional approach involves training separate, independent models for each specific task. STL serves as a baseline for performance comparison.
2.  **Multi-Task Learning (MTL) - Hard Parameter Sharing:** In this common MTL technique, initial layers (or a shared "backbone") of a neural network are utilized by all tasks, while the final layers are task-specific. This method promotes shared feature extraction.
3.  **Multi-Task Learning (MTL) - Soft Parameter Sharing (Cross-Stitch Networks):** This more flexible approach allows task-specific models to be trained somewhat independently but enables them to learn how to share and combine their feature maps at various network levels. This is achieved through "cross-stitch" units, which learn linear combinations of activations from different tasks.

## Overview of the Notebooks

### 1. `MTL_digits.ipynb` - Classification on a Modified MNIST Dataset

This notebook focuses on applying and comparing STL and MTL (Hard Parameter Sharing and Cross-Stitch) techniques using a dataset derived from MNIST.

* **Dataset:** MNIST, adapted for two binary classification tasks:
    * **Task 1:** Determine if a handwritten digit is **even or odd**.
    * **Task 2:** Determine if a handwritten digit is a **multiple of 3**.

    These tasks are considered **correlated** as they operate on the same input domain (handwritten digits) and are likely to benefit from shared low-level feature representations.
* **Base Architecture:** A simple Multi-Layer Perceptron (MLP) architecture is consistently used for all approaches. This choice helps to isolate the impact of different parameter-sharing strategies.
* **Objectives:**
    * To introduce the fundamental concepts of Multi-Task Learning.
    * To compare the performance of STL, Hard Parameter Sharing MTL, and Cross-Stitch MTL on correlated tasks.
    * To observe how Cross-Stitch units learn to mediate information flow between tasks.

### 2. `MTL_CIFAR10.ipynb` - Classification on CIFAR-10 with Uncorrelated Tasks

This notebook extends the investigation of MTL to a more complex dataset, CIFAR-10, and introduces tasks that are intentionally **uncorrelated**. The goal is to analyze the behavior of MTL approaches in more challenging scenarios.

* **Dataset:** CIFAR-10, a dataset of 32x32 color images in 10 classes.
* **Tasks:**
    * **Task 1 (Multi-class Classification):** Classify images into one of the 10 original CIFAR-10 categories (e.g., airplane, automobile, bird).
    * **Task 2 (Rotation Classification):** Determine the rotation angle applied to an image (0°, 90°, 180°, or 270°). This is a self-supervised task that relies on the image's structural properties rather than its primary semantic content.
    The primary objective here is to understand how a single model behaves when learning these distinct tasks jointly.
* **Base Architecture:** AlexNet architectures are employed as the foundation for the models, reflecting the increased complexity of the CIFAR-10 dataset compared to MNIST.
* **Objectives:**
    * To evaluate the effectiveness of Hard Parameter Sharing and Cross-Stitch MTL on a more realistic image dataset.
    * To investigate how MTL approaches manage uncorrelated tasks, where indiscriminate feature sharing might not be optimal and could lead to "negative transfer" (degraded performance).
    * To compare the flexibility of Cross-Stitch Networks with Hard Parameter Sharing in this context.

## General Structure and Implementation

Both notebooks adhere to a similar organizational structure:

1.  **Setup and Configuration:** Importing necessary libraries (PyTorch, Torchvision, NumPy, Matplotlib) and defining global parameters (e.g., computation device, batch size, learning rate, number of epochs).
2.  **Data Loading and Preprocessing:** Preparing the specific datasets (MNIST or CIFAR-10) and creating DataLoaders for training and testing. This includes appropriate data transformations and label definitions for the respective tasks.
3.  **Model Architecture Definition:** Implementing PyTorch classes for:
    * STL models (one per task).
    * MTL model with Hard Parameter Sharing.
    * MTL model with Cross-Stitch Networks (including the Cross-Stitch unit implementation).
4.  **Utility, Training, and Evaluation Functions:** Defining generic training loops (`train_epoch`) and evaluation loops (`evaluate_epoch`), functions for loss calculation (typically CrossEntropyLoss), and accuracy metrics.
5.  **Experiment Execution:** Training and evaluating each learning approach on the defined tasks.
6.  **Results Visualization and Discussion:** Presenting performance metrics (loss and accuracy) through plots and tables, followed by a qualitative discussion of the observed outcomes.

## How to Use

1.  Ensure you have a Python environment with the required libraries installed (primarily `torch`, `torchvision`, `numpy`, `matplotlib`).
2.  Open and run the cells within the Jupyter notebooks (`MTL_digits.ipynb` or `MTL_CIFAR10.ipynb`) in a suitable Jupyter environment.
3.  Users are encouraged to modify configuration parameters (e.g., number of epochs, learning rates) to conduct further experiments.

## Overall Project Conclusions

These notebooks serve as a practical introduction to Multi-Task Learning. They highlight how different parameter-sharing strategies can be implemented and demonstrate their potential advantages and disadvantages depending on the nature of the tasks and data. MTL is a promising field that can lead to more efficient and generalizable models, particularly when tasks exhibit some degree of relatedness. However, the choice of MTL approach and its careful implementation are crucial for avoiding negative transfer and realizing tangible benefits.

---

## License

MIT License

Copyright (c) 2025 [Gemelli Mattia, Nocella Francesco]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.