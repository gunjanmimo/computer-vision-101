# Computer Vision 101: Building Algorithms from Scratch

![Computer Vision](https://s41256.pcdn.co/wp-content/uploads/2019/04/SLIDER-Appen_image_annotation_05.jpg)

Welcome to the Computer Vision 101 GitHub repository! This repository is a comprehensive collection of computer vision algorithms implemented from scratch. Whether you're a beginner looking to learn the fundamentals of computer vision or an experienced developer interested in diving deeper, this repository has something for you.

The main goal of this project is to provide clear and well-documented implementations of various computer vision algorithms. By going through these implementations, you'll gain a better understanding of the underlying concepts and techniques that power modern computer vision applications.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Algorithms](#algorithms)
    - [Image Classification](#image-classification)
        - [ResNet](#resnet)
        - [VGG16](#vgg16)
    - [Object Detection](#object-detection)
        - [YOLO (You Only Look Once)](#yolo)
        - [SSD (Single Shot MultiBox Detector)](#ssd)
4. [Contributing](#contributing)
5. [License](#license)

## Introduction <a name="introduction"></a>

Computer vision is an exciting field that focuses on enabling computers to interpret visual information from the world. This repository aims to provide a hands-on approach to understanding various computer vision algorithms. Each algorithm is implemented from scratch, using popular libraries such as NumPy, OpenCV, and PyTorch.

## Getting Started <a name="getting-started"></a>

To get started with this repository, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the algorithm of your choice in the [Algorithms](#algorithms) section.
3. Follow the instructions provided in each algorithm's directory to run and experiment with the code.

## Algorithms <a name="algorithms"></a>

### Image Classification <a name="image-classification"></a>

#### ResNet <a name="resnet"></a>

ResNet (Residual Network) is a deep neural network architecture designed for efficient training of very deep networks. It introduces residual blocks that allow networks to be much deeper while mitigating the vanishing gradient problem.

- Implementation: [Link to ResNet Code](/image_classification/resnet)

#### VGG16 <a name="vgg16"></a>

VGG16 is a widely recognized image classification architecture known for its simplicity and effectiveness. It consists of 16 layers and features small convolutional filters with a consistent 3x3 kernel size.

- Implementation: [Link to VGG16 Code](/image_classification/vgg16)

### Object Detection <a name="object-detection"></a>

#### YOLO (You Only Look Once) <a name="yolo"></a>

YOLO is an object detection algorithm that can detect multiple objects in an image with a single forward pass of the neural network. It divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell.

- Implementation: [Link to YOLO Code](/object_detection/yolo)

#### SSD (Single Shot MultiBox Detector) <a name="ssd"></a>

SSD is another popular object detection algorithm that aims to achieve high accuracy and real-time processing. It utilizes multiple convolutional layers of different scales to detect objects at various sizes.

- Implementation: [Link to SSD Code](/object_detection/ssd)

## Contributing <a name="contributing"></a>

Contributions to this repository are welcome! If you have an algorithm you'd like to add or if you find any issues with the existing implementations, feel free to open a pull request. Please make sure to follow the contribution guidelines outlined in the repository.

## License <a name="license"></a>

This project is licensed under the [MIT License](LICENSE).

---

We hope you find this repository informative and educational. Happy coding and exploring the world of computer vision algorithms from scratch! If you have any questions or suggestions, please feel free to open an issue.

*Disclaimer: This repository is intended for educational purposes. While the implementations aim to accurately represent the algorithms, they might not be optimized for production use.*