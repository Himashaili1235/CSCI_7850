# Pneumonia Detection using Deep Learning Techniques

## Project Description

Pneumonia is a critical global health concern, emphasizing the need for early detection to reduce mortality rates. This project utilizes three Convolutional Neural Network (CNN) models—VGG16, VGG19, and ResNet50—to identify pneumonia from chest X-ray images. The models are trained to classify images into three categories: normal, viral pneumonia, and bacterial pneumonia. Evaluation metrics such as F1-Score, recall, precision, and accuracy are employed. Results indicate that the ResNet50 model outperformed others with an accuracy of 91.65%, showcasing the effectiveness of CNN models in pneumonia identification.

### Domain
- Computer Vision
- Machine Learning

### Sub-Domain
- Deep Learning
- Image Classification

### Techniques
- Deep Convolutional Neural Network
- ImageNet
- VGG16, VGG19, ResNet50

## Tools / Libraries

### Languages
- Python

### Tools/IDE
- Anaconda Jupyter Notebook

### Libraries
- Keras
- TensorFlow
- ImageNet

## Dataset Details

The dataset, sourced from Kaggle, comprises grayscale chest X-ray images categorized into normal, viral pneumonia, and bacterial pneumonia. With 1990 normal, 1980 bacterial pneumonia, and 1343 viral pneumonia images, various image processing techniques, including Gaussian blur, median blur, Sobel filter, and HSV color space transformation, enhance feature visibility for accurate diagnosis.

**Dataset Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/subhankarsen/novel-covid19-chestxray-repository)

## Model Parameters

- Machine Learning Library: Keras
- Base Models: VGG16, VGG19, ResNet50
- Optimizers: Adam
- Loss Function: categorical_crossentropy

## Training Parameters

- Batch Size: 32
- Number of Epochs: 30
- Training Time: 3 Hours for all three models

## Implementation Steps

### Step 1: Download Anaconda

- Visit [Anaconda website](https://www.anaconda.com/download/)
- Download the version compatible with your Windows system (64-bit or 32-bit)

### Step 2: Run the Installer

- Double-click on the downloaded Anaconda installer file
- Follow on-screen instructions
- Choose to add Anaconda to your system PATH for ease of use

### Step 3: Completing the Installation

- Optionally, install Microsoft VSCode
- Optionally, learn more about Anaconda Cloud and Conda
- Click "Finish" to complete the installation

### Step 4: Access Anaconda Prompt

- Open Start menu
- Type "Anaconda Prompt" in the search bar
- Click on "Anaconda Prompt" in the search results

### Step 5: Test Anaconda Installation

- In Anaconda Prompt, test if Anaconda is installed: `conda --version`
- Test if Python is installed: `python --version`

### Python Environment Setup

1. `conda create -n pneumonia_detection python=3.7`: Create a virtual environment
2. `conda activate pneumonia_detection`: Activate the virtual environment
3. `pip install ipykernel`: Install ipykernel for Jupyter notebooks
4. `python -m ipykernel install --user --name pneumonia_detection --display-name "pneumonia_detection"`: Register environment as a Jupyter kernel
5. Install required libraries:
   - `pip install opencv-python==4.6.0.66`
   - `pip install matplotlib==3.4.3`
   - `pip install seaborn==0.11.2`
   - `pip install tensorflow==2.5.0`

### Project Execution

- Open Anaconda Prompt
- Type `Jupyter Notebook`
- Execute `preprocess_dataset.ipynb` to generate a new dataset
- Execute `ResNet50.ipynb`, `vgg_16.ipynb`, and `vgg_19.ipynb` for model training
- Obtain outputs from each cell for final results
