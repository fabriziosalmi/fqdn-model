# FQDN Classifier: Detecting Malicious Domain Names

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://requires.io/github/fabriziosalmi/fqdn-model/requirements)

This project implements a machine learning-based classifier to identify malicious Fully Qualified Domain Names (FQDNs). It leverages extensive feature engineering, TF-IDF vectorization, and a variety of machine learning models, including ensemble methods, to achieve high accuracy in distinguishing between benign and malicious domains. Its generated output can be used by another model to improve detection accuracy by using a set of net tools.

[![asciicast](https://asciinema.org/a/jUVHCTLr1fBSyRInNHbZ7lbFB.svg)](https://asciinema.org/a/jUVHCTLr1fBSyRInNHbZ7lbFB)

## Table of Contents

*   [Demo](#demo)
*   [Features](#features)
*   [Requirements](#requirements)
*   [Installation](#installation)
*   [Usage](#usage)
    *   [Data Format](#data-format)
    *   [Basic Training and Evaluation](#basic-training-and-evaluation)
    *   [Prediction](#prediction)
    *   [Command-Line Options](#command-line-options)
    *   [Example with Options](#example-with-options)
*   [Output](#output)
*   [Model Selection](#model-selection)
*   [Performance Considerations](#performance-considerations)
*   [Contributing](#contributing)
*   [License](#license)
*   [Acknowledgments](#acknowledgments)

## Demo

```
python train.py blacklist.txt whitelist.txt
```

```
╭────────────── Training Summary ───────────────╮
│            Training Configuration             │
│ ┌───────────────────────────┬───────────────┐ │
│ │ Model                     │ random_forest │ │
│ │ Engineered Features Count │ 132           │ │
│ │ Training Samples          │ 21000         │ │
│ │ Test Samples              │ 9000          │ │
│ │ Scaling                   │ False         │ │
│ │ Quantile Transform        │ False         │ │
│ │ SMOTE                     │ False         │ │
│ │ N-gram Range              │ 2-4           │ │
│ └───────────────────────────┴───────────────┘ │
╰───────────────────────────────────────────────╯
Best parameters for random_forest: {'model__n_estimators': 200, 'model__max_depth': None}
Best model, vectorizer, and feature names saved to best_model_20250305_104814.pkl
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric            ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Accuracy          │ 0.9056 │
│ Precision         │ 0.9767 │
│ Recall            │ 0.8793 │
│ F1-Score          │ 0.9255 │
│ AUC               │ 0.9630 │
│ Log Loss          │ 0.2203 │
│ Average Precision │ 0.9842 │
│ Brier Score       │ 0.0681 │
│ MCC               │ 0.8058 │
└───────────────────┴────────┘

Confusion Matrix:
[[2874  126]
 [ 724 5276]]

Classification Report:
              precision    recall  f1-score   support

           0       0.80      0.96      0.87      3000
           1       0.98      0.88      0.93      6000

    accuracy                           0.91      9000
   macro avg       0.89      0.92      0.90      9000
weighted avg       0.92      0.91      0.91      9000


Model Settings:
  Model: random_forest
  Best Hyperparameters: {'model__n_estimators': 200, 'model__max_depth': None}
  Scaling: False
  Quantile Transform: False
  Vectorizer: TfidfVectorizer
    N-gram Range: (2, 4)
    Analyzer: char

Top 10 Feature Importances:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Feature                              ┃ Importance ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ num__subdomain_ratio                 │     0.0241 │
│ num__consonant_count_in_subdomain    │     0.0237 │
│ num__dot_position_variance           │     0.0224 │
│ num__subdomain_length                │     0.0222 │
│ num__subdomain_entropy               │     0.0198 │
│ num__special_to_alphanum_transitions │     0.0184 │
│ num__token_count                     │     0.0179 │
│ num__num_dots                        │     0.0173 │
│ num__num_non_alphanumeric            │     0.0172 │
│ num__num_subdomains                  │     0.0165 │
└──────────────────────────────────────┴────────────┘
```

## Features

*   **Extensive Feature Engineering:**  Extracts a comprehensive set of features designed to capture various aspects of FQDN structure and composition.  These features are critical for the model's ability to distinguish between benign and malicious domains. Includes:
    *   **Lexical Features:**
        *   Length-based features (FQDN length, subdomain length, etc.)
        *   Character counts (digits, hyphens, dots, etc.)
        *   Entropy (Shannon entropy of FQDN, SLD, subdomain)
        *   Ratios (digit ratio, vowel/consonant ratio, etc.)
        *   Boolean flags (presence of hyphen, starting/ending with digit, etc.)
        *   TLD, SLD, and subdomain extraction using `tldextract`
        *   Longest consecutive character sequences (digits, consonants, vowels)
        *   Tokenization (splitting by '.') and analysis of tokens
        *   Character transitions (vowel to consonant, digit to letter, etc.)
        *   N-grams (bigrams and trigrams) and their statistics
        *   Character repetition and frequency analysis
        *   Presence of keywords (login, account, secure, etc.)
        *   Unicode character detection
        *   Punycode detection
    *   **Domain-Specific Features:**
        *   Edit distances (Levenshtein distance) to common TLDs and SLDs
        *   TLD maliciousness score (if `tlds.csv` is provided)
        *   Subdomain analysis (presence of "www", vowel/consonant characteristics)
        *   Information relative to the SLD, like counts of vowels, consonant and special characters
        *   First/Last char type
    *   **Statistical Features:**
        *   Variance and entropy of token lengths and character positions
        *   Density of vowels and consonants
    * ... and many more.
*   **Multiple Machine Learning Models:** Supports a variety of models for flexibility and experimentation:
    *   Random Forest (`RandomForestClassifier`)
    *   Gradient Boosting (`GradientBoostingClassifier`)
    *   Logistic Regression (`LogisticRegression`)
    *   Support Vector Machine (SVM) (`SVC`)
    *   Naive Bayes (`GaussianNB`)
    *   AdaBoost (`AdaBoostClassifier`)
    *   Voting Classifier (ensemble of the above models using `VotingClassifier`)
*   **Automated Hyperparameter Tuning:** Employs `RandomizedSearchCV` for efficient exploration of hyperparameter space and model optimization, to give you the BEST model.
*   **Comprehensive Data Preprocessing:**
    *   Handles missing values (imputation).
    *   Optional feature scaling using `StandardScaler`.
    *   Optional quantile transformation using `QuantileTransformer` for non-normal data.
    *   TF-IDF vectorization of FQDN strings with n-gram support using `TfidfVectorizer`.
    *   Optional L1/L2 regularization for TF-IDF to prevent overfitting.
    *   Optional SMOTE (Synthetic Minority Oversampling Technique) for handling imbalanced datasets using `imblearn`.
*   **Iterative Feature Selection:**  Includes an option to perform iterative feature selection to reduce dimensionality, improve generalization, and enhance model interpretability.
*   **Detailed Model Evaluation:**  Provides a comprehensive suite of evaluation metrics to assess model performance:
    *   Accuracy, Precision, Recall, F1-score
    *   AUC (Area Under the ROC Curve)
    *   Log Loss (Cross-Entropy Loss)
    *   Confusion Matrix
    *   Classification Report (includes precision, recall, F1-score for each class)
    *   Average Precision
    *   Brier Score
    *   Matthews Correlation Coefficient (MCC)
*   **Visualizations for Insights:** Generates and saves informative visualizations:
    *   Confusion Matrix plot (`confusion_matrix.png`)
    *   ROC Curve plot (`roc_curve.png`)
    *   Precision-Recall Curve plot (`precision_recall_curve.png`)
    *   Feature Importance plot (if the model supports it) (`feature_importance.png`)
*   **Model Persistence:**  Saves the trained model, TF-IDF vectorizer, and feature names to a `.pkl` file using `joblib` for easy loading and reuse.  Filenames include timestamps for version control.
*   **User-Friendly Command-Line Interface:** Uses `argparse` to provide a flexible and easy-to-use command-line interface for training, evaluation, and prediction.
*   **Informative Output with Rich:** Leverages the `rich` library for formatted console output, progress bars, tables, and syntax highlighting to improve readability and user experience.
*   **Robust Error Handling:** Implements error handling for file operations, data loading, and unexpected data formats.

## Requirements

*   Python 3.8+
*   pandas
*   scikit-learn
*   numpy
*   tldextract
*   urllib3
*   rich
*   joblib
*   imblearn
*   matplotlib
*   seaborn

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/fabriziosalmi/fqdn-model.git
    cd fqdn-model
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Format

The script expects two text files as input:

*   **Blacklist:**  A list of malicious FQDNs, one FQDN per line.
*   **Whitelist:** A list of benign FQDNs, one FQDN per line.

Example:

```
# blacklist.txt
malicious.example.com
phishing.attack.net
...
```

```
# whitelist.txt
google.com
example.org
legitimate.website.com
...
```

### Basic Training and Evaluation

To train and evaluate a model using the default settings, run:

```bash
python train.py blacklist.txt whitelist.txt
```

### Prediction

To predict the class (benign or malicious) of a single FQDN, use the `-p` or `--predict` option:

```bash
python train.py blacklist.txt whitelist.txt -p example.com
```

### Command-Line Options

```
usage: train.py [-h] [-p PREDICT] [-ts TEST_SIZE] [-rs RANDOM_STATE] [-m {random_forest,gradient_boosting,logistic_regression,svm,naive_bayes,adaboost,ensemble}] [-ng NGRAM_RANGE NGRAM_RANGE]
                [--skip_errors] [--scale] [--quantile_transform] [-o OUTPUT_DIR] [-s SAVE_MODEL] [--smote] [--l1] [--l2] [--feature_selection]
                [--num_iterations NUM_ITERATIONS] [--max_jobs MAX_JOBS] [--rf_n_estimators RF_N_ESTIMATORS] [--rf_max_depth RF_MAX_DEPTH]
                [--gb_n_estimators GB_N_ESTIMATORS] [--gb_learning_rate GB_LEARNING_RATE] [--lr_C LR_C] [--svm_C SVM_C] [--svm_kernel SVM_KERNEL]
                [--ab_n_estimators AB_N_ESTIMATORS] [--ab_learning_rate AB_LEARNING_RATE]
                blacklist whitelist

FQDN Classifier

positional arguments:
  blacklist             Path to the blacklist file
  whitelist             Path to the whitelist file

options:
  -h, --help            show this help message and exit
  -p PREDICT, --predict PREDICT
                        FQDN to predict
  -ts TEST_SIZE, --test_size TEST_SIZE
                        Test size (default: 0.3)
  -rs RANDOM_STATE, --random_state RANDOM_STATE
                        Random state (default: 42)
  -m {random_forest,gradient_boosting,logistic_regression,svm,naive_bayes,adaboost,ensemble}, --model {random_forest,gradient_boosting,logistic_regression,svm,naive_bayes,adaboost,ensemble}
                        Model to use (default: random_forest) Available models: random_forest gradient_boosting logistic_regression svm naive_bayes adaboost
                        ensemble (VotingClassifier)
  -ng NGRAM_RANGE NGRAM_RANGE, --ngram_range NGRAM_RANGE NGRAM_RANGE
                        N-gram range (default: 2 4)
  --skip_errors         Skip malformed lines
  --scale               Scale features
  --quantile_transform  Apply quantile transformation
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory (default: results)
  -s SAVE_MODEL, --save_model SAVE_MODEL
                        Save the best model to a file (default: model/best_model.pkl)
  --smote               Use SMOTE for oversampling
  --l1                  Use L1 regularization for text features
  --l2                  Use L2 regularization for text features
  --feature_selection   Perform iterative feature selection
  --num_iterations NUM_ITERATIONS
                        Number of iterations for feature selection
  --max_jobs MAX_JOBS   Max number of parallel jobs to run (default: 4)
  --rf_n_estimators RF_N_ESTIMATORS
                        RF: Number of estimators
  --rf_max_depth RF_MAX_DEPTH
                        RF: Max depth
  --gb_n_estimators GB_N_ESTIMATORS
                        GB: Number of estimators
  --gb_learning_rate GB_LEARNING_RATE
                        GB: Learning rate
  --lr_C LR_C           LR: Inverse regularization
  --svm_C SVM_C         SVM: Regularization
  --svm_kernel SVM_KERNEL
                        SVM: Kernel ('linear', 'rbf', 'poly', 'sigmoid')
  --ab_n_estimators AB_N_ESTIMATORS
                        AB: Number of estimators
  --ab_learning_rate AB_LEARNING_RATE
                        AB: Learning Rate
```

### Example with Options

*   Train a Gradient Boosting model with SMOTE, scaling, and save the model:

    ```bash
    python train.py blacklist.txt whitelist.txt -m gradient_boosting --smote --scale -s my_gb_model.pkl
    ```

*   Train a Voting Classifier ensemble:

    ```bash
    python train.py blacklist.txt whitelist.txt -m ensemble
    ```

*   Train using iterative feature selection:

    ```bash
    python train.py blacklist.txt whitelist.txt --feature_selection --num_iterations 10
    ```

*   Use L2 regularization for TF-IDF:

    ```bash
    python train.py blacklist.txt whitelist.txt --l2
    ```

## Output

*   **Console Output:** The script prints detailed information to the console using the `rich` library, including:
    *   Training configuration summary
    *   Best hyperparameters (if applicable)
    *   Evaluation metrics (table)
    *   Confusion matrix
    *   Classification report
    *   Model settings
    *   Top 10 feature importances (if applicable)
*   **Output Directory (`results` by default):** The following files are saved to the output directory:
    *   `confusion_matrix.png`: Confusion matrix plot.
    *   `roc_curve.png`: ROC curve plot.
    *   `precision_recall_curve.png`: Precision-recall curve plot.
    *   `feature_importance.png`: Feature importance plot (if applicable).
*   **Saved Model (optional):** If the `-s` or `--save_model` option is used, the trained model, vectorizer, and feature names are saved to a `.pkl` file. The filename includes a timestamp for versioning. The model is saved under `model/`.

## Model Selection

The `--model` argument allows you to choose the machine learning model to use.  Consider the following when selecting a model:

*   **Random Forest:**  Generally a good starting point, robust and relatively easy to tune.
*   **Gradient Boosting:** Can achieve high accuracy but may require more careful tuning.
*   **Logistic Regression:**  A linear model, can be a good choice for interpretability and as a baseline.
*   **SVM:** Can be effective but computationally expensive, especially with large datasets.
*   **Naive Bayes:**  Simple and fast, but may not perform as well as other models.
*   **AdaBoost:** Another boosting algorithm, can be sensitive to noisy data.
*   **Ensemble:** Combines the strengths of multiple models, often resulting in improved performance. The algorithm does a 'soft voting', with weights depending on the different AUC metric achieved on the train dataset.

## Performance Considerations

*   The performance of the classifier is highly dependent on the quality and representativeness of the training data (blacklist and whitelist).
*   Feature engineering is crucial. The script generates a large number of features, which can lead to overfitting. Feature selection (using the `--feature_selection` option) or dimensionality reduction techniques can be used to improve generalization.
*   Hyperparameter tuning is essential for optimizing model performance. Use `RandomizedSearchCV` to explore different hyperparameter settings.
*   Consider experimenting with different TF-IDF vectorizer settings, such as n-gram range and regularization.
*   For imbalanced datasets, using SMOTE (`--smote`) can improve performance.
*   Parallel processing (`--max_jobs`) can significantly speed up training, especially for ensemble methods and hyperparameter tuning.
*   Loading the TLD information into memory speed-up the process significantly thanks to the LRU-cache implemented.

## Contributing

Contributions are welcome! Please feel free to submit pull requests with bug fixes, new features, or improvements to the documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   This project builds upon the work of many researchers and developers in the fields of machine learning, natural language processing, and cybersecurity.
*   The following libraries are used: pandas, scikit-learn, numpy, tldextract, urllib3, rich, joblib, imblearn, matplotlib, and seaborn.
