# Alzheimer's Disease Classification using Neural ODE and ResNet-18

A hybrid deep learning and ODE-based framework for Alzheimer's 
disease stage classification and progression forecasting using 
the OASIS-2 longitudinal MRI dataset.

## Overview

This project combines:
- **ResNet-18** (transfer learning) for MRI classification
- **Logistic growth ODE** for disease progression modelling
- **Grad-CAM** and **SHAP** for explainability
- **Sequential DL→ODE pipeline** for personalised 2-year CDR forecasts

## Classes
- Non-demented (CDR = 0)
- Converted (CDR 0 → 0.5)
- Demented (CDR ≥ 1)

## Dataset
OASIS-2 Longitudinal MRI Database  
373 sessions · 150 subjects · ages 60–96  
Access at: https://www.oasis-brains.org/

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Non-demented | 0.63 | 0.65 | 0.64 |
| Converted | 0.00 | 0.00 | 0.00 |
| Demented | 0.21 | 0.33 | 0.26 |
| Weighted avg | 0.42 | 0.46 | 0.43 |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/train.py
python src/evaluate.py
```

## Requirements
See requirements.txt

## Citation
If you use this work please cite:
Muhammad Faisal Abrar (2026). Alzheimer's Disease Classification 
using Neural ODE and ResNet-18. GitHub.
