# 1st CPU Experiment Summary

## 1. Experiment Setting
- Dataset: PlantVillage tomato subset
- Classes: 10
- Split: Train / Validation / Test
- Environment: Local CPU
- Epochs: 3

## 2. Internal Test Result
- Best model: EfficientNetB0
- Test Accuracy: 0.8621
- Macro F1: 0.8131
- Weighted F1: 0.8588

## 3. Class Imbalance Result
- EfficientNetB0 + Class Weight accuracy decreased compared to EfficientNetB0.
- However, Early blight F1-score improved from 0.5209 to 0.6564.
- Class Weight helped weak class performance but reduced overall stability.

## 4. External Test Result
- Taiwan External best model: Baseline CNN, Accuracy 0.3089
- Bangladesh BBox External best model: EfficientNetB0, Accuracy 0.1809
- External performance dropped significantly compared to PlantVillage internal test.
- Most models showed prediction bias toward Tomato___Late_blight.
- This suggests domain shift between PlantVillage and real/external datasets.

## 5. Next GPU Experiment Plan
- EfficientNetB0 longer training
- EfficientNetB0 fine-tuning
- Underrepresented class augmentation
- External test re-evaluation
- Image corruption robustness test