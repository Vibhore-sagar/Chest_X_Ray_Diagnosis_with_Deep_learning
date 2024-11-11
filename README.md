# Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning

## Project Overview

This project utilizes a DenseNet-based architecture for multi-label classification of medical conditions from chest X-ray images. The aim is to provide an efficient, high-accuracy model for identifying multiple conditions from a single X-ray, helping to streamline diagnostics and assist healthcare providers in quickly identifying critical findings. The project leverages pre-trained DenseNet121 for its ability to capture fine-grained features essential in medical imaging.

### Dataset
The project uses chest x-ray images taken from the public [ChestX-ray8 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC). This dataset contains 108,948 frontal-view X-ray images of 32,717 unique patients. Each image in the data set contains multiple text-mined labels identifying 14 different pathological conditions. These in turn can be used by physicians to diagnose 8 different diseases. For the project we have been working with a ~1000 image subset of the images.

### Project Highlights

1. **Multi-Label Classification**:
   - Designed to predict multiple medical conditions from each X-ray image.
   - Uses a final output layer with sigmoid activation to support multi-label outputs for conditions that may co-occur.

2. **Pre-Trained Model with Fine-Tuning**:
   - Initialized with ImageNet weights to benefit from pre-learned feature extraction.
   - Specific layers are fine-tuned to adapt to the unique characteristics of medical imaging data.

3. **Data Augmentation and Preprocessing**:
   - Extensive data augmentation applied to improve generalization and model robustness.
   - Preprocessing techniques such as resizing, normalization, and rotation are used to mimic real-world variations in X-ray images.

4. **Performance Metrics**:
   - Evaluated on metrics such as accuracy, AUC (Area Under Curve), and F1 score to ensure reliable performance across conditions.
   - Results are carefully monitored to prevent overfitting, especially crucial with the limited labeled data typical in medical imaging.

---

## DenseNet Architecture Overview

DenseNet, or **Densely Connected Convolutional Network**, is a powerful convolutional neural network architecture designed to improve feature reuse and gradient flow. Instead of traditional layer-to-layer connectivity, DenseNet introduces dense connections that link each layer to every previous layer within a dense block. This architecture is efficient in terms of parameter count and allows the model to capture complex details, which is especially useful in medical image analysis.

### Why DenseNet for Medical Imaging?

DenseNet’s densely connected layers and reduced parameter count make it particularly suitable for tasks like medical image classification, where high precision is required. The benefits of using DenseNet for this project include:

- **Enhanced Feature Propagation**: Dense connections improve gradient flow, allowing the network to better capture nuanced details in medical images.
- **Reduced Overfitting**: Fewer parameters make DenseNet less prone to overfitting, an advantage when working with limited data.
- **Compact Design**: DenseNet’s efficient architecture allows for high accuracy without excessive computational resources.

### Implementation: DenseNet121

In this project, DenseNet121 is chosen for its balance between depth and computational efficiency. Key layers include:

- **Dense Blocks** that promote feature sharing across layers.
- **Transition Layers** to manage feature map sizes and reduce complexity.
- **Output Layer** modified with sigmoid activation for multi-label classification, tailored to detect multiple conditions in a single X-ray.

---

## Summary

This project demonstrates the effectiveness of DenseNet121 for multi-label classification in medical imaging. By leveraging dense connectivity, feature reuse, and transfer learning, the model achieves robust results with limited data, offering a promising approach for supporting healthcare professionals with accurate diagnostic assistance. 
