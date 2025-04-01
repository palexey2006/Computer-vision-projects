# CNN Models
This directory contains implementations of various Convolutional Neural Network (CNN) architectures, showcasing their strengths, weaknesses, and applications. The models demonstrate different approaches to image classification and feature extraction using PyTorch.

## Directory Structure

## LeNet.py:
Implements LeNet-5, one of the earliest CNN architectures, designed for digit recognition and small-scale image tasks.

Simple_CNN.py:
A basic CNN with a few convolutional layers, suitable for small datasets.

AlexNet.py:
Reproduction of AlexNet, which introduced deep CNNs and revolutionized image classification.

VGG16.py:
Implements VGG16, a deep network with small convolutional filters to improve feature extraction.

GoogleNet.py:
Includes an implementation of GoogleNet (Inception v1), which introduced inception modules for efficient deep learning.

ResNet.py:
Implements ResNet, which uses residual connections to prevent vanishing gradients and enable very deep networks.

# Advantages & Disadvantages of CNN Architectures
**LeNet** is one of the earliest CNN architectures, known for its simplicity and efficiency in processing small images like handwritten digits. It requires minimal computational power, making it an excellent choice for lightweight applications. However, it struggles with complex datasets and is unsuitable for large-scale image classification tasks.

**Simple CNN** models are easy to implement and train quickly, making them ideal for small-scale datasets. However, their limited depth prevents them from capturing complex patterns, which reduces their effectiveness in real-world applications.

**AlexNet** introduced deeper CNN architectures and ReLU activation, significantly improving training speed and feature extraction. It was a breakthrough in deep learning, but its large number of parameters makes it computationally expensive and prone to overfitting on small datasets.

**VGG16** builds on AlexNet by using small convolutional filters and increasing depth, resulting in better feature extraction. It is widely used for transfer learning and image classification. However, it is computationally expensive and has slow inference times, making it challenging to deploy in real-time applications.

**GoogleNet**, also known as Inception v1, introduced inception modules, which significantly reduced the number of parameters while maintaining high accuracy. This makes it more efficient than VGG16 and AlexNet. However, its complex architecture makes it difficult to modify and requires careful tuning.

**ResNet** is the most advanced among these architectures, solving the vanishing gradient problem with residual connections. It enables training very deep networks, leading to superior performance on large-scale datasets. However, it is more computationally demanding and requires more memory than other models.

## Best Model Choice
For simple tasks (e.g., digit recognition) → **LeNet**

For general deep learning applications → **VGG16** (better balance between accuracy and computational cost)

For large-scale deep learning → **ResNet** (best performance for deep networks)
