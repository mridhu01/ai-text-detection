# Notebooks

This folder contains all Jupyter notebooks used in the project.

## Notebooks Overview

### 1. Data Collection (`1_data_collection.ipynb`)
- Dataset preparation and cleaning
- Train/test split
- Data exploration

### 2. Perplexity Detection (`2_perplexity_detection.ipynb`)
- GPT-2 perplexity baseline
- Threshold optimization
- Results: 79.1% accuracy

### 3. Statistical Detection (`3_statistical_detection.ipynb`)
- Feature extraction (20+ features)
- Random Forest classifier
- Results: 93.8% accuracy

### 4. Classifier Training (`4_classifier_training.ipynb`)
- Fine-tuning DistilBERT/RoBERTa
- Hyperparameter tuning
- Model evaluation

### 5. Classifier Detector (`5_classifier_detector.ipynb`)
- Final classifier evaluation
- Results: 99.5% accuracy
- Performance analysis

### 6. Adversarial Robustness (`6_adversarial_robustness.ipynb`)
- Testing against 4 attack types
- Robustness analysis
- Results: 98.5% under attack

### 7. Ensemble Method (`7_ensemble_method.ipynb`)
- Combining all three detectors
- Ensemble strategies
- Results: 100% test accuracy

### 8. Final Demo (`8_final_demo.ipynb`)
- Gradio web interface
- Final documentation
- Project summary

## How to Run

1. Open any notebook in Google Colab
2. Upload required data files
3. Run cells sequentially
4. Results will be saved to `results/` folder

## Requirements

See `requirements.txt` in the root directory.
```
