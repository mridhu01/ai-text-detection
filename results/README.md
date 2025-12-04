# Results

This folder contains all evaluation results and visualizations from the project.

## Key Visualizations

### Performance Comparison
- `method_comparison.png` - Comparison of all detection methods
- `ensemble_comparison.png` - Ensemble strategy analysis

### Model Evaluation
- `confusion_matrix_classifier.png` - Confusion matrix for best model
- `roc_curve_classifier.png` - ROC curve for classifier

### Adversarial Robustness
- `adversarial_robustness_analysis.png` - Performance under attacks

### Feature Analysis
- `feature_importance.png` - Most important features
- `dataset_visualization.png` - Dataset distribution

## Performance Metrics

### Final Results
- **Classifier**: 99.5% accuracy, 100% precision, 99.2% recall
- **Statistical**: 93.8% accuracy
- **Perplexity**: 79.1% accuracy
- **Ensemble**: 100% accuracy on test set

### Adversarial Robustness
- **Classifier**: 98.5% (only 1.5% drop)
- **Statistical**: 81.0% (17.5% drop)
- **Perplexity**: 50.5% (32.5% drop)

## Reports

- `classifier_results.json` - Detailed classifier metrics
- `statistical_detector_results.json` - Statistical method results
- `perplexity_detector_results.json` - Perplexity baseline results
- `adversarial_robustness_results.json` - Attack testing results
