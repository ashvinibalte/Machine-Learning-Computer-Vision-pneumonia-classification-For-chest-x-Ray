#Chest X-ray Images: Pneumonia Classification
#Abstract
Implementing the knowledge gained in ITCS 4152-5152 Computer Vision, the outcome of this project is to identify whether a person has pneumonia based on chest X-ray images. The dataset used is sourced from Kaggle and contains images classified as either having pneumonia or not. The model is built using Python, convolutional neural networks (CNN) along with VGG-16 and ResNet-50 models. This model is implemented on a platform to make it accessible.

#1. Introduction
Pneumonia is a contagious illness affecting the lungs, caused by bacteria, fungi, or viruses. According to the World Health Organization (WHO), pneumonia is a leading cause of death in children under the age of five. Pneumonia causes the pulmonary alveoli to swell with pus and fluid, restricting oxygen absorption and making breathing difficult.

Researchers use computer-aided diagnosis (CAD) to identify abnormalities in medical images. Computer-aided detection (CADe) and diagnosis (CADx) are common CAD divisions. CAD schemes, including computed tomography (CT), magnetic resonance imaging (MRI), and chest X-ray (CXRAY) images, assist in diagnosing lung illnesses. However, access to diagnostic imaging technologies like CT and MRI is limited, particularly in developing nations. CXRAY scans are widely used but require the expertise of trained medical radiologists.

Innovative solutions have been created to support the medical industry. A variety of machine learning (ML) approaches, particularly deep learning (DL) models like convolutional neural networks (CNNs), are employed for feature extraction in computer vision (CV) tasks.

#2. Proposed Tech Demo
Our project aims to develop an AI system for classifying X-ray images to determine if a patient has pneumonia. The dataset from Kaggle contains approximately 5000 images, split into training, validation, and testing sets. We utilize CNN models, VGG-16, and ResNet-50, comparing results from each model.

#4. Dataset
The dataset is provided by the Guangzhou Women and Children’s Medical Center and is accessible on Kaggle. It includes 5856 chest X-ray images (JPEG), categorized into training, validation, and testing sets. The training set contains more images classified as "Pneumonia" with data augmentation techniques applied to balance the dataset.

#5. Architecture
##5.1 CNN Model
Input Layer: Images resized to (150, 150, 3).
Convolution Layer: Five convolutional blocks with filters of sizes 16, 32, 64, 128, and 256.
Batch Normal Layer: Standardizes each mini-batch to stabilize learning.
ReLU: Introduces nonlinearity by setting negative values to zero.
Fully Connected Layer: Classifies convolved features.
Dropout Layer: Prevents overfitting by temporarily "dropping" neurons.
Output Layer: Uses Sigmoid function for binary classification.
Regularization: Implements ReduceLROnPlateau to lower the learning rate and save the model's best weights.
Fine-Tuning: Utilizes keras callbacks and Model Checkpoints.
##5.2 CNN Model Performance
Accuracy: Achieves 66% on test data after 20 epochs with a batch size of 32.
Loss: Training and validation losses stabilize after the initial epochs.
##5.3 VGG-16
Architecture: 16 layers with weights, using convolution layers of 3x3 filters with stride 1 and maxpool layers of 2x2 filters with stride 2.
Results: Achieves classification accuracy of 89.5%-90% on test data.
##5.4 ResNet-50
Architecture: 50 layers, using a "bottleneck" design with 1x1 convolutions for faster training.
Results: Achieves 79% accuracy on test data, lower than VGG-16.
#6. Conclusion
By carefully adjusting layers, regularizations, and hyperparameters, we trained a CNN from scratch and compared its performance to VGG-16 and ResNet-50. Our VGG-16 model generalized best to unknown data, a promising step towards a reliable computer-aided diagnostic tool.
