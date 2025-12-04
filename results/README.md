# Results

This folder contains evaluation results, visualizations, and performance metrics.

## Key Files

### Performance Reports
- `final_project_report.txt` - Comprehensive project analysis
- `final_metrics_comparison.csv` - Performance metrics table
- `project_summary.json` - Machine-readable project summary

### Visualizations
- `method_comparison.png` - All methods performance comparison
- `confusion_matrix_classifier.png` - Best model confusion matrix
- `confusion_matrix_best_ensemble.png` - Ensemble confusion matrix
- `ensemble_comparison.png` - Ensemble strategies analysis
- `roc_curve_classifier.png` - ROC curve for classifier
- `adversarial_robustness_analysis.png` - Performance under attacks
- `feature_importance.png` - Most important statistical features
- `dataset_visualization.png` - Dataset distribution analysis

## Key Results

### Detection Performance
| Method | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Perplexity | 79.1% | 78.6% | 86.6% | 0.824 |
| Statistical | 93.8% | 96.5% | 92.4% | 0.944 |
| **Classifier** | **99.5%** | **100%** | **99.2%** | **0.996** |
| Ensemble | 100% | 100% | 99.2% | 1.000 |

### Adversarial Robustness
- Classifier: 98.5% (only 1.5% drop under attack)
- Statistical: 81.0% (17.5% drop)
- Perplexity: 50.5% (32.5% drop)

## 📁 File Descriptions

All visualizations are high-resolution PNG files suitable for presentations and publications.
