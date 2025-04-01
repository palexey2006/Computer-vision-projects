# PytorchExtra

The **PytorchExtra** directory contains supplementary resources for enhancing PyTorch-based computer vision workflows. It includes various techniques and implementations aimed at improving model performance, handling dataset challenges, and leveraging pre-trained models for transfer learning. Each section provides practical examples and reusable scripts for integration into machine learning projects.

**Directory Structure**

## Albumentations/

This folder contains scripts that show how I used the Albumentations library for advanced image augmentation. Albumentations is great for applying efficient and flexible transformations that help improve model performance and make it more robust in real-world situations. It’s also faster than torchvision.transforms, which is one of the reasons I decided to include it in my final portfolio. Although I didn’t use very complex transformations, it still shows my skills with data augmentation. Augmented images are located at the butterflies folder and can be compared with original images. I also reused a dataset class, and I’ve explained its implementation clearly below.



## Data_augmentation/

This folder contains scripts showcasing different data augmentation techniques, including flipping, rotation, color jittering, and other transformations. Augmenting datasets helps reduce overfitting and improves model accuracy, especially when working with limited data. For this project, I used the standard library torchvision.transforms and also reused the Custom_datasets class.

I used 10 images of cats and augmented them, saving the results in the Augmented_images folder. You can compare the original and augmented images to see the transformations in action.



## Handling_imbalanced_datasets/

In this folder, I implemented two methods for handling imbalanced datasets.

The first method is **class weighting**, which is simpler but not commonly used. The idea behind it is to adjust the weights of different classes to balance their ratios. For example, in my dataset, I had 50 images of golden retrievers and only 1 image of Swedish Elkhounds. So, I increased the weight of the Swedish Elkhound class from 1 to 50 to match the golden retriever ratio.

The second method is **data augmentation**. I used this technique to balance the dataset by increasing the number of images for the underrepresented class. My balanced dataset now has equal class ratios, though it’s not truly balanced because I reused the same image 50 times. While this approach increases the number of images, it’s not always ideal since it doesn't introduce new data. You can find the augmented images in the folder **Balanced_dog_dataset_using_oversampling**.

## Custom_datasets.py

A script demonstrating how to create a simple custom dataset in PyTorch. It includes dataset class definitions that extend torch.utils.data.Dataset, allowing seamless integration of unique data formats into PyTorch pipelines.

## Transfer_learning.ipynb

A Jupyter Notebook illustrating the implementation of transfer learning techniques in PyTorch. I transfered VGG16 pretrained model, then freezed almost all layers and used only very last layers to train on the CIFAR10 dataset. My accuracy score is not really high though, but I trained the model just for a couple epochs to showcase how to train a transfered and pretrained model.
The notebook covers model fine-tuning, feature extraction, and practical applications of pre-trained models such as ResNet and EfficientNet.

## Usage

Each component in this directory serves as a standalone module that can be integrated into various deep learning projects. The scripts provide structured and efficient solutions to common challenges in computer vision, including data preparation, augmentation, and training improvements.

