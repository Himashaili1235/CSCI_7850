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
The dataset, sourced from Kaggle, encompasses images categorized into three groups: normal, viral pneumonia, and bacterial pneumonia. Specifically, there are 1990 normal images, 1980 bacterial pneumonia images, and 1343 viral pneumonia images within this dataset. Notably, all these images are in grayscale, representing a monochromatic spectrum without the use of color channels.
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
