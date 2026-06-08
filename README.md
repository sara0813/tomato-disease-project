# Tomato Disease Classification and Monitoring System

CNN-based tomato leaf disease classification project using deep learning models and external dataset evaluation.

This project classifies tomato leaf images into healthy and disease classes using CNN-based deep learning models.  
The final goal is to build a simple monitoring system that shows the predicted disease name, disease status, class information, and recommended response through a web dashboard.

---

## Project Overview

Tomato diseases can reduce crop productivity and quality.  
Since many tomato diseases appear visually on leaves, image-based deep learning models can be used to classify disease types automatically.

This project focuses on:

- Tomato leaf disease classification
- CNN-based model training and comparison
- Class imbalance analysis and mitigation
- External dataset evaluation
- Web dashboard prototype using Streamlit
- Future robustness test using image corruption

---

## Main Features

- Classifies tomato leaf images into 10 classes
- Compares multiple CNN-based models
- Evaluates model performance using accuracy, precision, recall, F1-score, and confusion matrix
- Tests generalization performance using external datasets
- Provides a Streamlit-based dashboard prototype
- Prepares for future image corruption robustness evaluation

---

## Dataset

### Main Dataset

- PlantVillage tomato leaf dataset
- 10 tomato classes
- RGB leaf images
- Used for training, validation, and testing

### External Evaluation Datasets

- Taiwan tomato leaves dataset
- Bangladesh tomato leaf dataset

These external datasets are used to evaluate whether the model trained on PlantVillage can generalize to images from different environments.

---

## Classes

The project uses the following 10 tomato classes:

```text
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___Septoria_leaf_spot
Tomato___Spider_mites Two-spotted_spider_mite
Tomato___Target_Spot
Tomato___Tomato_Yellow_Leaf_Curl_Virus
Tomato___Tomato_mosaic_virus
Tomato___healthy
```
---

## Models

The following CNN-based models are implemented and compared in this project:

- Baseline CNN
- DenseNet121
- EfficientNetB0
- EfficientNetB0 + Class Weight
- MobileNetV2

EfficientNetB0 showed the best performance in the first local CPU experiment, so additional experiments such as class weight, augmentation, and fine-tuning are mainly focused on EfficientNetB0.

---

## Project Structure

```text
tomato-disease-project/
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── corrupted_test/
│
├── models/
│
├── results/
│   └── summary/
│
├── src/
│   ├── data_prep/
│   ├── evaluate/
│   ├── summary/
│   ├── train/
│   ├── visualize/
│   ├── class_info.py
│   ├── config.py
│   └── models.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

> Note: `data/`, `models/`, and most files in `results/` are excluded from GitHub because they can be large.  
> Only summary result files are uploaded for documentation and presentation purposes.

---

## Environment

This project was developed using:

- Python 3.11
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Streamlit
- VS Code
- Windows local environment

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sara0813/tomato-disease-project.git
cd tomato-disease-project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

For Windows PowerShell:

```bash
.venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

After downloading the dataset, place the raw PlantVillage tomato dataset under:

```text
data/raw/plantvillage/
```

Then split the dataset into train, validation, and test sets:

```bash
python src/data_prep/split_dataset.py
```

Check the processed dataset:

```bash
python src/data_prep/check_processed_dataset.py
```

Analyze class imbalance:

```bash
python src/data_prep/analyze_class_imbalance.py
```

---

## Training

Train the Baseline CNN model:

```bash
python src/train/train_baseline.py
```

Train the DenseNet121 model:

```bash
python src/train/train_densenet.py
```

Train the EfficientNetB0 model:

```bash
python src/train/train_efficientnet.py
```

Train the EfficientNetB0 model with class weights:

```bash
python src/train/train_efficientnet_classweight.py
```

Train the MobileNetV2 model:

```bash
python src/train/train_mobilenet.py
```

---

## Evaluation

Evaluate each trained model on the PlantVillage test set:

```bash
python src/evaluate/evaluate_baseline.py
python src/evaluate/evaluate_densenet.py
python src/evaluate/evaluate_efficientnet.py
python src/evaluate/evaluate_efficientnet_classweight.py
python src/evaluate/evaluate_mobilenet.py
```

Evaluate external datasets:

```bash
python src/evaluate/evaluate_external.py taiwan efficientnetb0
python src/evaluate/evaluate_external.py taiwan efficientnetb0_classweight
python src/evaluate/evaluate_external.py bangladesh_bbox efficientnetb0
python src/evaluate/evaluate_external.py bangladesh_bbox mobilenetv2
```

---

## Result Summary

Generate summary tables for presentation and report writing:

```bash
python src/summary/make_model_comparison_summary.py
python src/summary/make_classwise_f1_summary.py
python src/summary/make_external_test_summary.py
```

Summary files are saved under:

```text
results/summary/
```

---

## Current Experiment Status

Completed:

- PlantVillage dataset preprocessing
- Baseline CNN training and evaluation
- DenseNet121 training and evaluation
- EfficientNetB0 training and evaluation
- EfficientNetB0 + Class Weight experiment
- MobileNetV2 training and evaluation
- Taiwan external dataset evaluation
- Bangladesh bbox-based external dataset preparation and evaluation
- Model comparison summary generation

Planned:

- Underrepresented class augmentation experiment
- EfficientNetB0 fine-tuning
- Image corruption robustness test
- Full Streamlit dashboard integration with trained model inference

---

## Dashboard

A Streamlit dashboard prototype is included in:

```text
app/streamlit_app.py
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

Current dashboard status:

- Image upload UI is implemented
- Disease class information is prepared
- Model inference connection will be completed in the next development stage

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

External dataset evaluation is also used to check whether the trained model can generalize beyond the clean PlantVillage dataset.

---

## Future Work

The next development steps are:

1. Improve performance on weak classes such as Early Blight and Target Spot
2. Apply augmentation to underrepresented classes
3. Fine-tune EfficientNetB0 using GPU
4. Perform image corruption tests for robustness evaluation
5. Connect the trained model to the Streamlit dashboard
6. Display disease name, status, confidence score, symptoms, and recommended response on the dashboard

---

## Project Purpose

This project is developed as a capstone design project for tomato disease classification and monitoring.

The goal is not only to achieve high accuracy on clean images, but also to evaluate model robustness and prepare a practical web-based monitoring system.
