# Tomato Disease Classification and Monitoring System

## Project Goal

This project aims to classify tomato leaf images into healthy and disease categories using CNN-based deep learning models.

## Main Tasks

- Tomato leaf disease classification
- Baseline CNN implementation
- DenseNet121 transfer learning
- Dataset comparison
- Image corruption robustness test
- Streamlit-based monitoring dashboard

## Dataset

### Main Dataset

- PlantVillage tomato leaf dataset
- 10 classes
- RGB color images

### Additional Datasets

- Taiwan tomato leaves dataset
- Bangladesh tomato leaf dataset
- AI Hub crop disease image dataset

## Models

- Baseline CNN
- DenseNet121

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Environment

- Python 3.11
- TensorFlow
- Streamlit
- VSCode


python src\data_preprocessing.py
python src\split_dataset.py
python src\prepare_taiwan_external_test.py
python src\convert_bangladesh_bbox_crop.py

python src\train_efficientnet.py
python src\evaluate_efficientnet.py
python src\evaluate_taiwan_external.py efficientnetb0
python src\evaluate_bangladesh_bbox_external.py efficientnetb0