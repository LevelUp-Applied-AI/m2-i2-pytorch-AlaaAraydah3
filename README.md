[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YUvA8hIt)
# Integration 2 — PyTorch: Housing Price Prediction

**Module 2 — Programming for AI & Data Science**

See the [Module 2 Integration Task Guide](https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-2/learner/integration-guide) for full instructions.

---

## Quick Reference

**File to complete:** `train.py`

**Install PyTorch before running:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Branch:** `integration-2/pytorch`

**Submit:** PR URL → TalentLMS Unit 8 text field

# --------------------------------------

# PyTorch Housing Price Prediction

# Model Prediction

Target: price_jod 
Input features:
area_sqm — apartment area in square meters
bedrooms — number of bedrooms
floor
age_years 
distance_to_center_km

# Training Configuration
Model: Linear(5 → 32) → ReLU → Linear(32 → 1)
Loss function: Mean Squared Error (MSELoss)
Optimizer
Learning rate: 0.01
Epochs: 100

# Training Outout
Loss decreased slowly over epochs (from 1.95e9 → 1.94e9)
Final loss value: 1,944,846,208.0
Predictions saved in predictions.csv
Observation
Loss decreased gradually but slowly
This could be due to the small dataset size or limited changes per epoch
Standardization of features helped improve the stability of gradient updates