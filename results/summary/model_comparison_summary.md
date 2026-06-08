# Model Comparison Summary

| Rank | Model | Experiment Type | Epoch | Test Loss | Test Accuracy | Macro F1 | Weighted F1 | Decision |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | EfficientNetB0 | Transfer Learning | 3 | 0.4414 | 0.8621 | 0.8131 | 0.8588 | Best model candidate |
| 2 | MobileNetV2 | Lightweight Transfer Learning | 3 | 0.4718 | 0.8442 | 0.7667 | 0.8345 | Lightweight model candidate |
| 3 | Baseline CNN | Custom CNN | 3 | 0.4886 | 0.8332 | 0.7749 | 0.8238 | Baseline model |
| 4 | EfficientNetB0 + Class Weight | Imbalance Handling | 3 | - | 0.8230 | 0.7935 | 0.8280 | Useful for imbalance analysis |
| 5 | DenseNet121 | Transfer Learning | 3 | 0.5909 | 0.8058 | 0.7433 | 0.8076 | Needs improvement |
