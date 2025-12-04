#  AI Text Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.5%25-success.svg)]()

> A state-of-the-art AI text detection system achieving 99.5% accuracy with exceptional robustness against adversarial attacks.

---

##  Table of Contents
- [Key Achievements](#-key-achievements)
- [Methods Implemented](#-methods-implemented)
- [Performance Metrics](#-performance-metrics)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Adversarial Robustness](#Ô∏è-adversarial-robustness)
- [Results](#-results)
- [Limitations](#Ô∏è-limitations)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

##  Key Achievements

| Metric | Score |
|--------|-------|
| **Detection Accuracy** | 99.5% |
| **Precision** | 100% (Zero false positives) |
| **Recall** | 99.2% |
| **Adversarial Robustness** | 98.5% (Only 1.5% drop under attack) |
| **Ensemble Accuracy** | 100% on test set |

---

##  Methods Implemented

### 1. **Perplexity-Based Detection** 
- **Accuracy:** 79.1%
- Uses GPT-2 perplexity scores
- Baseline method

### 2. **Statistical Feature Analysis**
- **Accuracy:** 93.8%
- Analyzes 20+ linguistic features
- Lexical diversity, sentence structure, POS tags

### 3. **Deep Learning Classifier** ‚≠ê **BEST**
- **Accuracy:** 99.5%
- Fine-tuned DistilBERT/RoBERTa
- Zero false positives

### 4. **Smart Ensemble**
- **Accuracy:** 100% (test set)
- Combines all three methods
- Maximum robustness

---

##  Performance Metrics

| Method | Accuracy | Precision | Recall | F1 Score | Robustness |
|--------|----------|-----------|--------|----------|------------|
| Perplexity | 79.1% | 78.6% | 86.6% | 0.824 | 50.5% |
| Statistical | 93.8% | 96.5% | 92.4% | 0.944 | 81.0% |
| **Classifier** | **99.5%** | **100%** | **99.2%** | **0.996** | **98.5%** |
| Ensemble | 100% | 100% | 99.2% | 1.000 | - |

---

##  Quick Start

### Option 1: Web Demo (Gradio)
```bash
python demo.py
```
Then open the URL in your browser.

### Option 2: Python API
```python
from detector import detect_ai_text

text = "Your text here..."
result = detect_ai_text(text)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/ai-text-detection.git
cd ai-text-detection
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK data (optional)
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

---

##  Usage

### Using Individual Methods

#### Classifier (Recommended - 99.5% accuracy)
```python
from detector import classifier_detect

text = "Sample text to analyze..."
prediction, confidence = classifier_detect(text)
print(f"{prediction} with {confidence:.2%} confidence")
```

#### Statistical Method
```python
from detector import statistical_detect

text = "Sample text to analyze..."
prediction, confidence = statistical_detect(text)
print(f"{prediction} with {confidence:.2%} confidence")
```

#### Perplexity Method
```python
from detector import perplexity_detect

text = "Sample text to analyze..."
prediction, perplexity, confidence = perplexity_detect(text)
print(f"{prediction} (Perplexity: {perplexity:.2f})")
```

#### Ensemble (All methods combined)
```python
from detector import ensemble_detect

text = "Sample text to analyze..."
prediction, confidence = ensemble_detect(text)
print(f"{prediction} with {confidence:.2%} confidence")
```

---

##  Project Structure
```
ai-text-detection/
‚îÇ
‚îú‚îÄ‚îÄ data/
‚îÇ   ‚îî‚îÄ‚îÄ complete_dataset.csv          # Full dataset (1,055 samples)
‚îÇ
‚îú‚îÄ‚îÄ models/
‚îÇ   ‚îú‚îÄ‚îÄ classifier_final/             # Best model (99.5% accuracy)
‚îÇ   ‚îÇ   ‚îú‚îÄ‚îÄ config.json
‚îÇ   ‚îÇ   ‚îú‚îÄ‚îÄ model.safetensors
‚îÇ   ‚îÇ   ‚îî‚îÄ‚îÄ tokenizer files
‚îÇ   ‚îî‚îÄ‚îÄ statistical_detector.pkl      # Statistical model
‚îÇ
‚îú‚îÄ‚îÄ results/
‚îÇ   ‚îú‚îÄ‚îÄ final_project_report.txt      # Comprehensive report
‚îÇ   ‚îú‚îÄ‚îÄ final_metrics_comparison.csv  # Performance metrics
‚îÇ   ‚îú‚îÄ‚îÄ final_project_summary.png     # Visualization dashboard
‚îÇ   ‚îî‚îÄ‚îÄ [other result files]
‚îÇ
‚îú‚îÄ‚îÄ detector.py                        # Main detection functions
‚îú‚îÄ‚îÄ demo.py                           # Gradio web demo
‚îú‚îÄ‚îÄ requirements.txt                  # Python dependencies
‚îú‚îÄ‚îÄ README.md                         # This file
‚îî‚îÄ‚îÄ LICENSE                           # License file
```

---

##  Technical Details

### Dataset
- **Total Samples:** 1,055
- **Human Samples:** 422 (training) + 92 (test)
- **AI Samples:** 422 (training) + 119 (test)
- **Train/Test Split:** 80/20
- **Sources:** Multiple (essays, articles, stories, etc.)

### Model Architecture
- **Base Model:** DistilBERT / RoBERTa
- **Fine-tuning:** Binary classification head
- **Max Sequence Length:** 256 tokens
- **Batch Size:** 16
- **Learning Rate:** 2e-5
- **Epochs:** 3-5

### Features (Statistical Method)
- Character count, word count, sentence count
- Average word/sentence length
- Lexical diversity & unique word ratio
- Punctuation ratios
- Stopword & transition word ratios
- Personal pronoun usage
- POS tag distributions
- And more...

---

##  Adversarial Robustness

We tested the system against 4 types of adversarial attacks:

| Attack Type | Description | Impact on Classifier |
|-------------|-------------|---------------------|
| **Paraphrasing** | Rewording sentences | Minimal (98.5%) |
| **Synonym Replacement** | Replacing words with synonyms | Minimal (98.5%) |
| **Character-level Typos** | Introducing spelling errors | Minimal (98.5%) |
| **Sentence Reordering** | Shuffling sentence order | Minimal (98.5%) |

**Key Finding:** Only **1.5% performance degradation** (vs. 32.5% for baseline methods)

### Attack Transfer Rate
- **<40%** transfer rate between methods
- Attacks optimized for one detector don't fool others
- **0/50** adversarial samples fooled ALL detectors simultaneously
- Strong **defense-in-depth** with ensemble approach

---

##  Results

### Confusion Matrix (Best Model - Classifier)

|  | Predicted Human | Predicted AI |
|---|-----------------|--------------|
| **Actual Human** | 92 (TN) | 0 (FP) |
| **Actual AI** | 1 (FN) | 118 (TP) |

### Key Metrics
- **True Positives:** 118/119 AI texts detected
- **True Negatives:** 92/92 human texts correctly identified
- **False Positives:** 0 (Perfect precision!)
- **False Negatives:** 1 (99.2% recall)

---

##  Limitations

1. **Language:** Tested on English text only
2. **Dataset Size:** 1,055 samples (larger validation recommended)
3. **Domain:** May require retraining for specialized domains (legal, medical, code)
4. **Long Documents:** Performance on documents >2000 words not fully validated
5. **Newer Models:** Primarily tested on GPT-2/3-era AI text
6. **Hybrid Content:** Human-AI collaborative writing may be challenging

---

##  Future Work

### Short-term
- [ ] Expand dataset to 10,000+ samples
- [ ] Test on GPT-4, Claude, Gemini outputs
- [ ] Add support for longer documents (>2000 words)
- [ ] Create REST API for production deployment

### Medium-term
- [ ] Multilingual support (Spanish, French, Chinese, etc.)
- [ ] Domain-specific models (academic, creative, technical)
- [ ] Real-time detection browser extension
- [ ] Integration with LMS platforms (Canvas, Moodle, Blackboard)

### Long-term
- [ ] Mobile application (iOS/Android)
- [ ] Hybrid human-AI content detection
- [ ] Confidence calibration improvements
- [ ] Explainable AI features (highlight suspicious passages)

---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  Contact

**Mridhula Senthilkumar** - mridhu01@umd.edu

Project Link: [https://github.com/mridhu01/ai-text-detection](https://github.com/mridhu01/ai-text-detection)

---

##  Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) - For the transformer models
- [OpenAI GPT-2](https://openai.com/research/gpt-2) - For perplexity baseline
- [Gradio](https://gradio.app/) - For the web demo interface
- [scikit-learn](https://scikit-learn.org/) - For machine learning utilities

---


<div align="center">

Made by Mridhula Senthilkumar

</div>

<img width="462" height="690" alt="image" src="https://github.com/user-attachments/assets/7dcacd32-6e4a-4fc9-a46b-b58644cef15a" />
