\# 📷 Object Detection - 10 Classes



!\[Python](https://img.shields.io/badge/Python-3.12-blue)

!\[TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)

!\[Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)

!\[License](https://img.shields.io/badge/License-MIT-green)



\---



\# Project Overview



This project is an end-to-end Deep Learning Image Classification system that classifies images into \*\*10 different object categories\*\* using a Fine-Tuned MobileNetV2 model.



The project demonstrates the complete Machine Learning lifecycle, from data preprocessing and exploratory analysis to model training, evaluation, hyperparameter tuning, deployment, and documentation.



\---



\# Business Problem



Manual identification of objects from images is time-consuming and prone to human error.



The goal of this project is to automate image classification using a lightweight, accurate, and deployment-ready deep learning model.



\---



\# Dataset



\- Dataset Size: \*\*1.46 GB\*\*

\- Number of Classes: \*\*10\*\*

\- Total Images: \*\*13,394\*\*



\### Classes



\- Tench

\- Springer

\- Cassette Player

\- Chain Saw

\- Church

\- French Horn

\- Garbage Truck

\- Gas Pump

\- Golf Ball

\- Parachute



\---



\# Project Lifecycle



\- Business Understanding

\- Data Collection

\- Exploratory Data Analysis

\- Data Preprocessing

\- Feature Engineering

\- Machine Learning Models

\- Deep Learning Models

\- Hyperparameter Tuning

\- Model Comparison

\- Deployment

\- Documentation



\---



\# Models Implemented



\### Machine Learning



\- Linear SVM

\- Random Forest



\### Deep Learning



\- CNN

\- MobileNetV2

\- ResNet50

\- EfficientNetB0



\---



\# Final Results



| Model | Accuracy |

|--------|----------|

| MobileNetV2 (Fine-Tuned) | \*\*98.22%\*\* |

| CNN | 63.44% |

| ResNet50 | 58.34% |

| Random Forest | 39.11% |

| Linear SVM | 32.48% |

| EfficientNetB0 | 18% |



\---



\# Best Model



🏆 \*\*Fine-Tuned MobileNetV2\*\*



\### Performance



\- Accuracy: \*\*98.22%\*\*

\- Precision: \*\*98.28%\*\*

\- Recall: \*\*98.22%\*\*

\- F1 Score: \*\*98.23%\*\*

\- Top-3 Accuracy: \*\*99.62%\*\*



\---



\# Technologies Used



\- Python

\- TensorFlow

\- Keras

\- MobileNetV2

\- NumPy

\- Pandas

\- Matplotlib

\- Pillow

\- Streamlit

\- Google Colab

\- GitHub



\---



\# Project Structure



```text

Object\_Detection\_10\_Classes

│

├── app

│   └── app.py

│

├── model

│   └── Best\_MobileNetV2\_FineTuned.keras

│

├── metadata

│   ├── label\_dictionary.json

│   ├── DeepLearning\_ClassNames.json

│   └── Label\_Mapping.csv

│

├── assets

├── docs

├── screenshots

├── sample\_images

├── requirements.txt

├── README.md

└── .gitignore

```



\---



\# Installation



Clone the repository



```bash

git clone https://github.com/sundar66kalyan/Object\_Detection\_10\_Classes.git

```



Navigate to the project



```bash

cd Object\_Detection\_10\_Classes

```



Install dependencies



```bash

pip install -r requirements.txt

```



Run the application



```bash

streamlit run app/app.py

```



\---



\# Streamlit Application



Upload an image.



The application will



\- Preprocess the image

\- Predict the class

\- Display confidence score

\- Show Top-3 predictions



\---



\# Features



\- Transfer Learning

\- Fine-Tuned MobileNetV2

\- Real-Time Image Classification

\- Professional Streamlit Dashboard

\- Top-3 Predictions

\- Confidence Scores

\- Lightweight Deployment



\---



\# Challenges Faced



\- Large dataset handling

\- Long SVM training time

\- Dataset imbalance verification

\- TensorFlow dataset optimization

\- Google Colab memory limitations

\- Model comparison across multiple architectures

\- Hyperparameter tuning



\---



\# Future Improvements



\- YOLO Object Detection

\- FastAPI Deployment

\- Docker Support

\- TensorFlow Lite

\- ONNX Export

\- Mobile Deployment



\---



\# Author



\*\*Kalyana Sundar\*\*



AI / Machine Learning Engineer



GitHub:



https://github.com/sundar66kalyan



\---



\# License



This project is released under the MIT License.



\---



⭐ If you found this project useful, please consider giving it a star on GitHub.

