# External Test Summary

| Dataset | Rank | Model | Accuracy | Weighted F1 | Overlap Macro F1 | Overlap Weighted F1 | Dominant Prediction | Zero Recall |
|---|---:|---|---:|---:|---:|---:|---|---:|
| Taiwan External | 1 | Baseline CNN | 0.3089 | 0.2109 | 0.2197 | 0.2109 | Tomato___Late_blight (242) | 1/3 |
| Taiwan External | 2 | MobileNetV2 | 0.2930 | 0.1987 | 0.2068 | 0.1987 | Tomato___Late_blight (247) | 1/3 |
| Taiwan External | 3 | DenseNet121 | 0.2898 | 0.1984 | 0.2077 | 0.1984 | Tomato___Late_blight (228) | 1/3 |
| Taiwan External | 4 | EfficientNetB0 | 0.2516 | 0.1633 | 0.1737 | 0.1633 | Tomato___Late_blight (212) | 1/3 |
| Taiwan External | 5 | EfficientNetB0 + Class Weight | 0.2325 | 0.1825 | 0.1942 | 0.1825 | Tomato___Late_blight (157) | 1/3 |
| Bangladesh BBox External | 1 | EfficientNetB0 | 0.1809 | 0.0764 | 0.0983 | 0.0764 | Tomato___Late_blight (837) | 3/6 |
| Bangladesh BBox External | 2 | MobileNetV2 | 0.1295 | 0.0520 | 0.0632 | 0.0520 | Tomato___Late_blight (755) | 4/6 |
| Bangladesh BBox External | 3 | EfficientNetB0 + Class Weight | 0.1267 | 0.0654 | 0.0846 | 0.0655 | Tomato___Late_blight (640) | 3/6 |
| Bangladesh BBox External | 4 | DenseNet121 | 0.1102 | 0.0382 | 0.0387 | 0.0382 | Tomato___Late_blight (855) | 5/6 |
| Bangladesh BBox External | 5 | Baseline CNN | 0.0946 | 0.0368 | 0.0411 | 0.0368 | Tomato___Late_blight (795) | 3/6 |
