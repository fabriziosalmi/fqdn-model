# FQDN Classifier

This project implements a machine learning-based classifier to identify malicious Fully Qualified Domain Names (FQDNs). It leverages a combination of extensive feature engineering, TF-IDF vectorization, and various machine learning models, including ensemble methods, to achieve high accuracy in distinguishing between benign and malicious domains.

## Demo

```
python train.py blacklist.txt whitelist.txt 
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

*   **Comprehensive Feature Engineering:** Extracts a wide range of features from FQDNs, including:
    *   Length-based features (FQDN length, subdomain length, etc.)
    *   Character counts (digits, hyphens, dots, etc.)
    *   Entropy (Shannon entropy of FQDN, SLD, subdomain)
    *   Ratios (digit ratio, vowel/consonant ratio, etc.)
    *   Boolean flags (presence of hyphen, starting/ending with digit, etc.)
    *   TLD, SLD, and subdomain extraction
    *   Longest consecutive character sequences (digits, consonants, vowels)
    *   Tokenization (splitting by '.') and analysis of tokens
    *   Character transitions (vowel to consonant, digit to letter, etc.)
    *   N-grams (bigrams and trigrams)
    *   Character repetition
    *   Presence of keywords (login, account, secure, etc.)
    *   Edit distances (Levenshtein distance) to common TLDs and SLDs
    *   ... and many more!
*   **Multiple Machine Learning Models:** Supports a variety of models:
    *   Random Forest
    *   Gradient Boosting
    *   Logistic Regression
    *   Support Vector Machine (SVM)
    *   Naive Bayes
    *   AdaBoost
    *   Voting Classifier (ensemble of the above models)
*   **Hyperparameter Tuning:** Uses `RandomizedSearchCV` for efficient hyperparameter optimization.
*   **Data Preprocessing:**
    *   Handles missing values.
    *   Optional feature scaling and quantile transformation.
    *   TF-IDF vectorization with n-gram support and optional L1/L2 regularization.
    *   Optional SMOTE (Synthetic Minority Oversampling Technique) for imbalanced datasets.
*   **Iterative Feature Selection**: Option to run an iterative feature selection process to reduce dimensionality and improve performance.
*   **Detailed Evaluation:** Provides comprehensive evaluation metrics:
    *   Accuracy
    *   Precision
    *   Recall
    *   F1-score
    *   AUC (Area Under the ROC Curve)
    *   Log Loss
    *   Confusion Matrix
    *   Classification Report
    *   Average Precision
    *   Brier Score
    *   Matthews Correlation Coefficient (MCC)
*   **Visualization:** Generates and saves:
    *   Confusion Matrix
    *   ROC Curve
    *   Precision-Recall Curve
    *   Feature Importance Plot (for models that support it)
*   **Model Persistence:** Saves the trained model, vectorizer, and feature names to a file for later use.
*   **Command-Line Interface:** Uses `argparse` for a user-friendly command-line interface.
*   **Rich Output:**  Leverages the `rich` library for formatted output, progress bars, and tables.
*   **Robust Error Handling**: Includes error handling for file operations and data loading.

## Requirements

*   Python 3.8+
*   pandas
*   scikit-learn
*   numpy
*   tldextract
*   urllib
*   rich
*   joblib
*   imblearn
*   matplotlib
*   seaborn

Install the required packages using pip:

```bash
pip install pandas scikit-learn numpy tldextract urllib3 rich joblib imbalanced-learn matplotlib seaborn
```

## Usage

The script can be used to train a model, evaluate its performance, and predict the class (benign/malicious) of individual FQDNs.

**Basic Training and Evaluation:**

```bash
python model.py blacklist.txt whitelist.txt
```

*   `blacklist.txt`: Path to a text file containing a list of malicious FQDNs (one FQDN per line).
*   `whitelist.txt`: Path to a text file containing a list of benign FQDNs (one FQDN per line).

**Prediction:**

```bash
python model.py blacklist.txt whitelist.txt -p example.com
```

*   `-p` or `--predict`:  Predict the class of a single FQDN.

**Command-Line Options:**

```
usage: model.py [-h] [-p PREDICT] [-ts TEST_SIZE] [-rs RANDOM_STATE] [-m {random_forest,gradient_boosting,logistic_regression,svm,naive_bayes,adaboost,ensemble}] [-ng NGRAM_RANGE NGRAM_RANGE]
                [--skip_errors] [--scale] [--quantile_transform] [-o OUTPUT_DIR] [-s SAVE_MODEL] [--smote] [--l1] [--l2] [--feature_selection]
                [--num_iterations NUM_ITERATIONS] [--rf_n_estimators RF_N_ESTIMATORS] [--rf_max_depth RF_MAX_DEPTH] [--gb_n_estimators GB_N_ESTIMATORS]
                [--gb_learning_rate GB_LEARNING_RATE] [--lr_C LR_C] [--svm_C SVM_C] [--svm_kernel SVM_KERNEL] [--ab_n_estimators AB_N_ESTIMATORS]
                [--ab_learning_rate AB_LEARNING_RATE]
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
                        Save the best model to a file (default: best_model.pkl)
  --smote               Use SMOTE for oversampling
  --l1                  Use L1 regularization for text features
  --l2                  Use L2 regularization for text features
  --feature_selection   Perform iterative feature selection
  --num_iterations NUM_ITERATIONS
                        Number of iterations for feature selection
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

**Example with Options:**

*   Train a Gradient Boosting model with SMOTE, scaling, and save the model:

    ```bash
    python model.py blacklist.txt whitelist.txt -m gradient_boosting --smote --scale -s my_gb_model.pkl
    ```

*   Train a Voting Classifier ensemble:

    ```bash
    python model.py blacklist.txt whitelist.txt -m ensemble
    ```

* Train using iterative feature selection:
    ```bash
    python model.py blacklist.txt whitelist.txt --feature_selection --num_iterations 10
    ```

*   Use L2 regularization for TF-IDF:

    ```bash
    python model.py blacklist.txt whitelist.txt --l2
    ```

## Output

*   **Console Output:** The script prints detailed information to the console, including:
    *   Best hyperparameters (if applicable)
    *   Evaluation metrics (table)
    *   Confusion matrix
    *   Classification report
    *   Model settings
    *   Top 10 feature importances (if applicable)
*   **Output Directory (`results` by default):**
    *   `confusion_matrix.png`: Confusion matrix plot.
    *   `roc_curve.png`: ROC curve plot.
    *   `precision_recall_curve.png`: Precision-recall curve plot.
    *   `feature_importance.png`: Feature importance plot (if applicable).
*   **Saved Model (optional):** If the `-s` or `--save_model` option is used, the trained model, vectorizer, and feature names are saved to a `.pkl` file.  The filename includes a timestamp for versioning.

## Notes

*   The performance of the classifier depends heavily on the quality and quantity of the training data (blacklist and whitelist) then a decent dataset is bundled with the project :)
*   The script generates a large number of features. Feature selection or dimensionality reduction techniques (like the included iterative feature selection) can be used to improve performance and reduce overfitting.
*   Consider exploring other feature engineering techniques and models for further improvement.


