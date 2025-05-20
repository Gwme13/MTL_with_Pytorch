# Multi-Task Learning (MTL) vs. Single-Task Learning (STL) for Image Classification on CIFAR-10

This notebook explores and compares Multi-Task Learning (MTL) and Single-Task Learning (STL) approaches for image classification using the CIFAR-10 dataset. The project exploits AlexNet-like architectures as the base for neural networks and investigates both hard and soft parameter sharing mechanisms for MTL.

## Table of Contents
1.  [Tasks](#tasks)
2.  [Learning Approaches Explored](#learning-approaches-explored)
3.  [Objectives](#objectives)
4.  [Libraries Used](#libraries-used)
5.  [Setup and Configuration](#setup-and-configuration)
    * [Reproducibility](#reproducibility)
    * [Device Selection](#device-selection)
6.  [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    * [CIFAR-10 Dataset and Task Definitions](#cifar-10-dataset-and-task-definitions)
    * [Data Normalization](#data-normalization)
    * [Data Augmentation and Transformations](#data-augmentation-and-transformations)
    * [Dataset Splits](#dataset-splits)
7.  [Custom Dataset Class](#custom-dataset-class)
8.  [DataLoaders](#dataloaders)
9.  [Model Architectures](#model-architectures)
    * [AlexNet-like Base Architecture (Feature Extractor)](#alexnet-like-base-architecture-feature-extractor)
    * [Hard Parameter Sharing MTL (`AlexNetMTL_HardShare`)](#hard-parameter-sharing-mtl-alexnetmtl_hardshare)
    * [Soft Parameter Sharing MTL (Cross-Stitch Networks - `CrossStitchAlexNetMTL`)](#soft-parameter-sharing-mtl-cross-stitch-networks---crossstitchalexnetmtl)
    * [Single-Task Learning (`AlexNetSTL`)](#single-task-learning-alexnetstl)
10. [Utility Functions](#utility-functions)
11. [Training and Evaluation](#training-and-evaluation)
12. [Results and Discussion](#results-and-discussion)
    * [Training Progress Visualization](#training-progress-visualization)
    * [Final Test Set Performance Summary](#final-test-set-performance-summary)
    * [Interpretation of Results](#interpretation-of-results)
    * [Confusion Matrices Analysis](#confusion-matrices-analysis)
13. [Conclusion](#conclusion)
14. [License](#license)

## Tasks
Two distinct image classification tasks are defined based on the CIFAR-10 dataset:
1.  **Task 1: Original CIFAR-10 Classification (Multi-class)**: Classify images into one of the 10 original CIFAR-10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
2.  **Task 2: Animal vs. Non-Animal Classification (Binary)**: Classify images into "animal" or "non-animal".

## Learning Approaches Explored
The notebook implements and compares three main learning strategies:
* **Hard Parameter Sharing MTL**: A single AlexNet-based network with a shared backbone (feature extractor) and task-specific heads (classifiers) for each task.
* **Soft Parameter Sharing MTL (Cross-Stitch Networks)**: Two separate AlexNet-based networks (one for each task) with *cross-stitch* units. These units allow for learned linear combinations of feature maps at intermediate layers, providing a more flexible way of sharing information between tasks.
* **Single-Task Learning (STL)**: Two separate AlexNet-based networks, one for each task, trained independently with no parameter sharing. This serves as a baseline.

## Objectives
* Implement 2 Multi-Task learning techniques:
  * Hard parameter sharing
  * Soft parameter sharing
* Compare the performance against single-task models.


## Libraries Used
* **PyTorch & Torchvision**: For building and training neural networks, loading the CIFAR-10 dataset, and applying image transformations.
* **NumPy**: For numerical operations.
* **Tqdm**: For displaying progress bars during training and evaluation loops.
* **Matplotlib & Seaborn**: For plotting training results and confusion matrices.
* **Pandas**: For creating and managing DataFrames to save training/validation metrics to CSV files.
* **Scikit-learn**: For calculating evaluation metrics like confusion matrices.
* **Humanize**: For formatting large numbers (e.g., parameter counts) in a human-readable way.
* **Pathlib**: For path manipulations, particularly checking for model file existence.
* **Copy**: For deep copying dataset objects to apply different transformations.

## Setup and Configuration

### Reproducibility
A random seed (`SEED = 1234`) is set for Python's `random`, `NumPy`, and `PyTorch (CPU, CUDA, and MPS)` to ensure that experiments are reproducible. This is crucial for making fair comparisons between different models and training runs. 

For CUDA:
* `torch.backends.cudnn.deterministic = True`
* `torch.backends.cudnn.benchmark = False` 

### Device Selection
Selection of the appropriate computation device available in the order:
1.  CUDA GPU (`torch.device('cuda')`)
2.  Apple Silicon GPU (MPS) (`torch.device('mps')`)
3.  CPU (`torch.device('cpu')`)


## Data Loading and Preprocessing

### CIFAR-10 Dataset and Task Definitions
The CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, is automatically downloaded.
* **Task 1 (Multi-class)** uses the original 10 labels: `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.
* **Task 2 (Binary)** classifies images as "Animal" or "Non-animal".
    * Non-animal classes are defined as: `airplane, automobile, ship, truck` (original indices 0, 1, 8, 9).
    * Animal classes include all other categories (`bird, cat, deer, dog, frog, horse`).

### Data Normalization
The mean and standard deviation of the pixel values are calculated from the training portion of the original CIFAR-10 dataset. These statistics are then used to normalize the image data (scaling pixel values to a standard range).
* Calculated means: `[0.4914, 0.4822, 0.4465]`
* Calculated stds: `[0.2470, 0.2435, 0.2616]`

### Data Augmentation and Transformations
* **Training Transformations (`train_transforms`)**:
    * `RandomRotation(5)`: Randomly rotates images by up to 5 degrees.
    * `RandomHorizontalFlip(0.5)`: Randomly flips images horizontally with a 50% probability.
    * `RandomCrop(32, padding=2)`: Randomly crops images to 32x32 after padding with 2 pixels on each side.
    * `ToTensor()`: Converts PIL images to PyTorch tensors.
    * `Normalize(mean=means, std=stds)`: Normalizes tensor images using the pre-calculated means and standard deviations.
* **Test/Validation Transformations (`test_transforms`)**:
    * `ToTensor()`: Converts PIL images to PyTorch tensors.
    * `Normalize(mean=means, std=stds)`: Normalizes tensor images.
    Data augmentation is applied only to the training set to improve model generalization and reduce overfitting.

### Dataset Splits
The CIFAR-10 training dataset (50,000 images) is split into:
* **Training set**: 90% of the original training data (45,000 images), with `train_transforms` applied.
* **Validation set**: 10% of the original training data (5,000 images), with `test_transforms` applied (no augmentation during validation).
The CIFAR-10 test dataset (10,000 images) is used for final model evaluation, with `test_transforms` applied.

## Custom Dataset Class
A custom PyTorch `Dataset` class, `CustomCIFAR10TasksDataset`, is implemented.
* It wraps a subset of the CIFAR-10 dataset (e.g., train, validation, or test split).
* The `__getitem__` method is overridden to return:
    1.  The transformed image.
    2.  The original 10-class label (for Task 1).
    3.  A binary label (0 for non-animal, 1 for animal) derived from the original label (for Task 2).
This allows a single dataset instance to provide data for both tasks simultaneously, which is essential for MTL.

## DataLoaders
PyTorch `DataLoader` instances are created for each dataset split (training, validation, and test) using `BATCH_SIZE = 128`.
* The `train_loader` shuffles the data at each epoch.
* `valid_loader` and `test_loader` do not shuffle the data.

## Model Architectures
AlexNet-like architectures are adapted for CIFAR-10's 32x32 input images.

### AlexNet-like Base Architecture (Feature Extractor)
The adapted convolutional base, used as a feature extractor in Hard Sharing and STL models, consists of five convolutional layers:
1.  Conv1: `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)`, ReLU, `MaxPool2d(kernel_size=2, stride=2)` -> Output: 64x16x16
2.  Conv2: `nn.Conv2d(64, 192, kernel_size=3, padding=1)`, ReLU, `MaxPool2d(kernel_size=2, stride=2)` -> Output: 192x8x8
3.  Conv3: `nn.Conv2d(192, 384, kernel_size=3, padding=1)`, ReLU
4.  Conv4: `nn.Conv2d(384, 256, kernel_size=3, padding=1)`, ReLU
5.  Conv5: `nn.Conv2d(256, 256, kernel_size=3, padding=1)`, ReLU, `MaxPool2d(kernel_size=2, stride=2)` -> Output: 256x4x4

The **flattened output** size of this feature extractor is `4096` ($256 \times 4 \times 4$). This output is then passed to the classifier base and task-specific heads.

### Hard Parameter Sharing MTL (`AlexNetMTL_HardShare`)
* **Architecture**:
    * Uses the shared AlexNet-like feature extractor.
    * The flattened output of the feature extractor is passed through a shared classifier base:
        * `Dropout(0.5)`
        * `Linear(4096, 1024)`, ReLU
        * `Dropout(0.5)`
        * `Linear(1024, 1024)`, ReLU
    * This shared representation (1024 features) is then fed into two task-specific heads:
        * Task 1 Head: `Linear(1024, num_classes_task1)` (10 output classes for CIFAR-10)
        * Task 2 Head: `Linear(1024, num_outputs_task2)` (1 output for binary animal/non-animal, using sigmoid later)
* **Total Parameters**: 7,507,787 (~7.5M).

### Soft Parameter Sharing MTL (Cross-Stitch Networks - `CrossStitchAlexNetMTL`)
* **Architecture**:
    * Two separate AlexNet-like pathways, one for each task. Each pathway is divided into three blocks similar to the base architecture.
    * **CrossStitchUnit**: This module connects the two pathways. It takes feature maps from corresponding layers of the two task pathways and learns a linear combination:
        * $$\tilde{x}_A^l = \alpha_{AA} \cdot x_A^l + \alpha_{AB} \cdot x_B^l$$
        * $$\tilde{x}_B^l = \alpha_{BA} \cdot x_A^l + \alpha_{BB} \cdot x_B^l$$
        * The $\alpha$ values form a learnable $2 \times 2$ "stitch matrix", are initialized as `[[0.9, 0.1], [0.1, 0.9]]` to encourage initial self-reliance with some learned sharing.
    * Cross-stitch units are applied after Block 1 and Block 2 of each pathway.
    * The outputs from Block 3 of each pathway are flattened and fed into task-specific classifiers (similar to the HardShare model's classifier base and heads, but separate for each task).
* **Total Parameters**: 15,004,307 (~15M).
    * This is nearly double the HardShare model due to the additional parameters from the cross-stitch units and separate pathways.

### Single-Task Learning (`AlexNetSTL`)
* **Architecture**:
    * A separate AlexNet-like model is used for each task.
    * It uses the same feature extractor as the HardShare model.
    * The classifier consists of:
        * `Dropout(0.5)`
        * `Linear(4096, 1024)`, ReLU
        * `Dropout(0.5)`
        * `Linear(1024, 1024)`, ReLU
        * `Linear(1024, num_outputs)` (task-specific output layer)
* **STL_T1 (Task 1)**: `num_outputs = 10`. Parameters: 7,506,762 (~7.5M).
* **STL_T2 (Task 2)**: `num_outputs = 1`. Parameters: 7,497,537 (~7.5M).

## Utility Functions
* `count_parameters(model)`: Calculates the total number of trainable parameters in a PyTorch model.
* Generic Training Epoch Functions (`generic_train_epoch_mtl`, `generic_train_epoch_stl`):
    * Handle one epoch of training.
    * Iterate through the dataloader, move data to device, perform forward pass, calculate loss(es), backpropagate, and update optimizer.
    * Accumulate and return average epoch loss(es) and accuracy(ies).
    * For MTL, total loss is a simple sum of Task 1 loss (`CrossEntropyLoss`) and Task 2 loss (`BCEWithLogitsLoss`).
* Generic Evaluation Epoch Functions (`generic_evaluate_epoch_mtl`, `generic_evaluate_epoch_stl`):
    * Handle one epoch of evaluation (on validation or test set).
    * Similar to training functions but without backpropagation or optimizer steps (`torch.no_grad()`).
    * Return average epoch loss(es) and accuracy(ies).

## Training and Evaluation
For each model (HardShareMTL, CrossStitchMTL, STL_T1, STL_T2):
1.  **Initialization**: Model, loss function(s) (`nn.CrossEntropyLoss` for Task 1, `nn.BCEWithLogitsLoss` for Task 2), and Adam optimizer (`lr=0.0001`) are initialized.
2.  **Training Loop**: Models are trained for `NUM_EPOCHS = 20`.
    * In each epoch, the model is trained on the training set and evaluated on the validation set.
    * The model weights that achieve the best (lowest) validation loss are saved.
    * Training and validation metrics (losses and accuracies for each task) are recorded for each epoch.
3.  **Results Saving**: Epoch-wise metrics are saved to a CSV file for each model.
4.  **Final Testing**: The best saved model (based on validation performance) is loaded and evaluated on the test set. Final test losses and accuracies are reported.

## Results and Discussion

### Training Progress Visualization
Plots are generated for each model, showing:
* **MTL Models**:
    * Total Train vs. Validation Loss
    * Task 1 Train vs. Validation Accuracy
    * Task 2 Train vs. Validation Accuracy
    * Task 1 Train vs. Validation Loss
    * Task 2 Train vs. Validation Loss
* **STL Models**:
    * Train vs. Validation Loss
    * Train vs. Validation Accuracy

These plots help in analyzing training dynamics, convergence, and potential overfitting.

### Final Test Set Performance Summary

| Model                   | Task 1 Accuracy (10-Class) | Task 2 Accuracy (Animal/Non-Animal) |
| :---------------------- | :------------------------- | :---------------------------------- |
| AlexNet HardShareMTL    | 79.22%                     | **96.71%** |
| AlexNet CrossStitchMTL  | 79.85%                     | 95.56%                              |
| AlexNet STL (Task 1)    | **80.56%** | N/A                                 |
| AlexNet STL (Task 2)    | N/A                        | 95.76%                              |

### Interpretation of Results

* **Task 1 (10-Class CIFAR-10 Classification)**:
    * The **STL model dedicated to Task 1 achieved the highest accuracy (80.56%)**. This is often expected, as a model focused solely on one task can optimize its parameters specifically for it.
    * **CrossStitchMTL (79.85%)** performed slightly better than **HardShareMTL (79.22%)** on this primary task. This suggests that the more flexible parameter sharing of Cross-Stitch networks might have been marginally beneficial for the more complex 10-class task.
    * Both MTL approaches were competitive and can be considered effective. The performance gap between MTL and STL models is relatively small, indicating that MTL can be a viable alternative to STL for related tasks.

* **Task 2 (Binary Animal/Non-Animal Classification)**:
    * **HardShareMTL achieved the highest accuracy (96.71%)** for this auxiliary task, outperforming even the dedicated STL model for Task 2 (95.76%). This is a key finding, indicating that the shared representations learned by the HardShareMTL model were highly beneficial for this simpler, related binary task. The main task (Task 1) might have acted as a regularizer or helped the model learn features that are also discriminative for the animal/non-animal distinction.
    * **CrossStitchMTL (95.56%)** performed slightly lower than the other two on this task, though still achieving high accuracy.

* **Overall Comparison**:
    * **STL models** provide strong baselines and achieved the best individual performance on Task 1.
    * **HardShareMTL** demonstrated a good balance. While slightly lower than STL on Task 1, it excelled on Task 2, suggesting effective knowledge transfer from the primary task to the auxiliary task. This highlights a common benefit of MTL where an auxiliary task can improve by leveraging features learned for a more complex primary task.
    * **CrossStitchMTL**, despite its theoretical flexibility and higher parameter count (nearly double that of HardShareMTL), did not consistently outperform the other methods in terms of peak test accuracy on both tasks. It was better than HardShareMTL on Task 1 but slightly worse on Task 2. This indicates that the added complexity of soft parameter sharing might require more extensive tuning or might not always translate to superior performance, especially given the computational overhead. The specific architecture of the cross-stitch units and the layers at which they are placed can significantly impact performance.

The results suggest that MTL, particularly hard parameter sharing, can be an effective strategy, especially for improving the performance of related auxiliary tasks by leveraging shared knowledge.

### Confusion Matrices Analysis
Confusion matrices are generated for each model on the test set for both tasks:
* **Task 1 (10-Class)**: These matrices show the per-class true positives, false positives, true negatives, and false negatives, revealing which classes are often confused with each other (e.g., "cat" vs. "dog").
* **Task 2 (Binary)**: These $2 \times 2$ matrices show the performance in distinguishing "Animal" vs. "Non-Animal" classes.

A detailed look at these matrices would provide insights into specific error patterns for each model and task, complementing the overall accuracy scores. For instance, one might observe if MTL helps in disambiguating specific confusing classes in Task 1 better than STL.

## Conclusion
This notebook provides a thorough implementation and comparative analysis of Single-Task Learning, Hard Parameter Sharing MTL, and Soft Parameter Sharing (Cross-Stitch) MTL for image classification on CIFAR-10.

The key takeaways are:
* STL models generally set strong performance baselines for individual tasks.
* Hard Parameter Sharing MTL can effectively leverage shared representations to benefit related tasks, sometimes even outperforming a dedicated STL model on an auxiliary task (as seen with Task 2). It is also more parameter-efficient than training multiple STL models or complex soft-sharing MTL models.
* Soft Parameter Sharing with Cross-Stitch Networks, while offering more flexibility, showed competitive but not consistently superior performance in this setup. Its effectiveness can be highly dependent on the dataset, task relatedness, and architectural choices, and it comes with increased model complexity and computational cost.

The framework established in this notebook, including data handling, model definitions, generic training functions, and detailed result analysis, serves as a valuable resource for further exploration into multi-task learning.

## License

MIT License

Copyright (c) 2025 Gemelli Mattia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
