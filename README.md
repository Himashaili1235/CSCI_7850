Project Title: Pneumonia Detection using Deep Learning Techniques

Description: 
An acute respiratory infection that affects the lungs,pneumonia is a major global health concern. It is still a major cause of death, which highlights how important early detection is to lowering the death rate. Three convolutional neural network (CNN) models were used in this study to identify pneumonia from chest X-ray images: VGG16, VGG19, and ResNet50. Using numerous convolutional layers, the models were trained to classify images into three categories: normal, viral pneumonia, and bacterial pneumonia. Evaluation metrics including F1-Score, recall, precision, and accuracy were used. The results showed that the ResNet50 model performed better than the other models, with an accuracy of 91.65. These results demonstrate CNN models’ effectiveness in correctly identifying pneumonia and point to their potential influence on enhancing diagnostic results.

Domain             : Computer Vision, Machine Learning

Sub-Domain         : Deep Learning, Image Classification

Techniques         : Deep Convolutional Neural Network, ImageNet, VGG16, VGG19, ResNet50

Tools / Libraries:

Languages               : Python

Tools/IDE               : Anaconda Jupyter Notebook

Libraries               : Keras, TensorFlow, ImageNet


Dataset Details:
The dataset, sourced from Kaggle, encompasses images categorized into three groups: normal, viral pneumonia, and bacterial pneumonia. Specifically, there are 1990 normal images, 1980 bacterial pneumonia images, and 1343 viral pneumonia images within this dataset. Notably, all these images are in grayscale, representing a monochromatic spectrum without the use of color channels. In this image processing pipeline, various filters are applied to enhance the features of a medical image depicting lung opacity. The process begins with the application of Gaussian blur to reduce noise, followed by median blur to further smooth the image. The subsequent step involves the use of a Sobel filter to highlight edges, contributing to the extraction of meaningful structures. Additionally, the image is transformed into the HSV color space after median blur, providing a different perspective for analysis. This comprehensive approach aims to improve the visibility of relevant details in medical imaging for accurate diagnosis and interpretation.
Dataset Link: https://www.kaggle.com/datasets/subhankarsen/novel-covid19-chestxray-repository

Model Parameters-
Machine Learning Library: Keras;
Base Model              : VGG16, VGG19, ResNet50;
Optimizers              : Adam;
Loss Function           : categorical_crossentropy

Training Parameters: 
Batch Size              - 32;
Number of Epochs        - 30;
Training Time           - 3 Hours for all three models

Below are the steps to implement the project:

Step 1: Download Anaconda
1.	Visit the Anaconda website(https://www.anaconda.com/download/)
2.	Click on the "Download" button for the version compatible with your Windows system (64-bit or 32-bit).
3.	The downloaded file is usually an executable (.exe) installer.

Step 2: Run the Installer
1.	Double-click on the downloaded Anaconda installer file.
2.	Follow the on-screen instructions in the Anaconda installer.
3.	During the installation process, you'll be asked whether to add Anaconda to your system PATH. It's recommended to select this option as it makes it easier to use Anaconda from the command line.

Step 3: Completing the Installation
1.	Once the installation is complete, you may be asked whether to install Microsoft VSCode. You can choose based on your preference.
2.	Optionally, you can leave the box checked to "Learn more about Anaconda Cloud and Conda." This will open a web page with more information about Anaconda.
3.	Click "Finish" to complete the installation.

Step 4: Access Anaconda Prompt
1.	Open the Start menu on your Windows system.
2.	In the search bar, type "Anaconda Prompt."
3.	Click on "Anaconda Prompt" in the search results.

Step 5: Test Anaconda Installation
1.	In the Anaconda Prompt, you can test if Anaconda is installed by typing:
   
conda --version

Press Enter, and it should display the installed Conda version.
3.	You can also test if Python is installed by typing:
   
python –version

Press Enter, and it should display the installed Python version.
Now you have successfully installed Anaconda and accessed the Anaconda Prompt. From here, you can create and manage Python environments, install packages, and run Python scripts.

Python environment for a project related to Pneumonia Detection. 
Let's break it down:
1.	conda create -n pneumonia_detection python=3.7: This command creates a new virtual environment named "pneumonia_detection" using Conda, a package manager. It specifies Python version 3.7 for this environment.
2.	conda activate pneumonia_detection: Activates the virtual environment named "pneumonia_detection." This ensures that any subsequent installations or commands are isolated within this environment.
3.	pip install ipykernel: Installs the "ipykernel" package using pip, a Python package installer. This package is likely needed for working with Jupyter notebooks.
4.	python -m ipykernel install --user --name pneumonia_detection --display-name "pneumonia_detection": This registers the virtual environment "pneumonia_detection" as a kernel for Jupyter notebooks. It allows Jupyter to use this environment.
5.	pip install opencv-python==4.6.0.66: Installs a specific version (4.6.0.66) of the OpenCV library, which is often used for computer vision tasks.
6.	pip install matplotlib==3.4.3: Installs a specific version (3.4.3) of Matplotlib, a popular plotting library for Python.
7.	pip install seaborn==0.11.2: Installs a specific version (0.11.2) of Seaborn, a statistical data visualization library based on Matplotlib.
8.	pip install tensorflow==2.5.0: Installs a specific version (2.5.0) of TensorFlow, a machine learning framework.

Once all the necessary libraries are installed open Anaconda Promt from start menu and type Jupyter Notebook to run the Python scripts.
Initially excute the preprocess_dataset.ipynb file and generate new dataset of images for three categories. Eventually, excute the ResNet50.ipynb, vgg_16.ipynb and vgg_19.ipynb files in jupter notebook and Finally obtain the outputs from each cell.


