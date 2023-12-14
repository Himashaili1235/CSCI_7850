Project Title: Pneumonia Detection using Deep Learning Techniques

Description: 
An acute respiratory infection that affects the lungs,pneumonia is a major global health concern. It is still a major cause of death, which highlights how important early detection is to lowering the death rate. Three convolutional neural network (CNN) models were used in this study to identify pneumonia from chest X-ray images: VGG16, VGG19, and ResNet50. Using numerous convolutional layers, the models were trained to classify images into three categories: normal, viral pneumonia, and bacterial pneumonia. Evaluation metrics including F1-Score, recall, precision, and accuracy were used. The results showed that the ResNet50 model performed better than the other models, with an accuracy of 91.65. These results demonstrate CNN models’ effectiveness in correctly identifying pneumonia and point to their potential influence on enhancing diagnostic results.

Domain             : Computer Vision, Machine Learning

Sub-Domain         : Deep Learning, Image Classification

Techniques         : Deep Convolutional Neural Network, ImageNet, VGG16, VGG19, ResNet50

Dataset Details:
The dataset, sourced from Kaggle, encompasses images categorized into three groups: normal, viral pneumonia, and bacterial pneumonia. Specifically, there are 1990 normal images, 1980 bacterial pneumonia images, and 1343 viral pneumonia images within this dataset. Notably, all these images are in grayscale, representing a monochromatic spectrum without the use of color channels.
Dataset Link: https://www.kaggle.com/datasets/subhankarsen/novel-covid19-chestxray-repository

Model Parameters:

Machine Learning Library: Keras
Base Model              : VGG16, VGG19, ResNet50
Optimizers              : Adam
Loss Function           : categorical_crossentropy
