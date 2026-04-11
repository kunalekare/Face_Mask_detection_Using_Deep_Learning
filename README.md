# 😷 MaskNet: Face Mask Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Google Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)

# 🎥 Video Link of Face Mask Detection using Deep Learning.

👉 [Click here to view project files and videos](https://drive.google.com/drive/folders/1St8qte5yzXTzy08rK7i-0tchLCwp5rGw?usp=sharing)

# 🎥 Video Link of Guided Project video

👉 [Click here to view project files and videos](https://drive.google.com/drive/folders/1CAX_nPGiJ13-EnqBSA5_KPilmcDwVUqd?usp=sharing)

# Datasets Link:
[Datasets](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

## 📌 Overview
**MaskNet** is a deep learning computer vision project that classifies whether a person in an image is wearing a face mask or not. Built entirely in Google Colab, this project utilizes **Transfer Learning** via the **MobileNetV2** architecture to achieve high accuracy with minimal computational overhead. 

## 🧠 Problem Statement
- **Type:** Image Classification (Binary)
- **Objective:** To build a robust convolutional neural network (CNN) capable of identifying if a subject is wearing a protective face mask (`with_mask`) or not (`without_mask`).

## 📊 Dataset
The model is trained on the popular **Face Mask Dataset** curated by Omkar Gurav. 
- **Source:** [Kaggle - Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Classes:** 2 (`with_mask`, `without_mask`)
- **Data Access:** Dynamically downloaded directly into the Colab environment using `kagglehub`.

## 🛠️ Technology Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Pre-trained Model:** MobileNetV2 (ImageNet weights)
* **Data Manipulation/Visualization:** NumPy, Matplotlib
* **Environment:** Google Colab

## ⚙️ Model Architecture & Methodology
To train the model efficiently, we leveraged **Transfer Learning**:
1. **Base Model:** `MobileNetV2` (The top classification layers were excluded, and base layers were frozen).
2. **Custom Head:** - `AveragePooling2D` (7x7)
   - `Flatten`
   - `Dense` (128 neurons, ReLU activation)
   - `Dropout` (0.5 to prevent overfitting)
   - `Dense` (2 neurons, Softmax activation for binary output)
3. **Image Preprocessing:** Images are resized to `224x224` pixels and normalized. We use `ImageDataGenerator` to perform real-time data augmentation (rotation, zooming, flipping, etc.) to make the model more robust.
4. **Optimizer:** Adam (Learning rate = 1e-4)
5. **Loss Function:** Categorical Crossentropy

## 🚀 How to Run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. **Enable GPU:** Go to `Runtime > Change runtime type` and set the Hardware Accelerator to **T4 GPU**.
4. Copy the code provided in the project files block-by-block.
5. **Run the cells sequentially**:
   - Step 1: Install dependencies (`kagglehub`) and download data.
   - Step 2: Set up data generators for training and validation.
   - Step 3: Build and compile the MobileNetV2 model.
   - Step 4: Train the model and save it as `masknet_model.h5`.
   - Step 5: Visualize the accuracy and loss curves.
   - Step 6: Test the model on your own uploaded images.

## 📈 Results
After training, the model outputs an accuracy and loss plot. Thanks to MobileNetV2, the model typically converges to a high validation accuracy (usually >95%) within just 10-15 epochs. 

The saved model (`masknet_model.h5`) can be easily exported and integrated into web apps (using Flask/Streamlit) or real-time OpenCV scripts.

## 🔮 Future Scope
- **Real-Time Detection:** Integrate OpenCV to capture webcam feeds and detect masks in real-time.
- **Multiple Faces:** Use an object detection algorithm (like YOLO or MTCNN) to detect multiple faces in a single frame before passing them to MaskNet for classification.
- **Web App Deployment:** Wrap the model in a Streamlit interface for easy user interaction.

---
*Created as a deep learning learning project. Feel free to fork and improve!*
