import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    log_loss,
    brier_score_loss,
)
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
import logging
import wandb
from typing import Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import os
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeRemainingColumn
from rich.logging import RichHandler
from rich.panel import Panel
import joblib

# Configure logging with RichHandler
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)]
)
log = logging.getLogger("rich")


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, filename="confusion_matrix.png"):
    """Plots a confusion matrix and saves it to a file."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    log.info(f"Confusion matrix saved to {filename}")



def train_model(
    data_path: str,
    model_type: str = "gaussian_nb",
    use_scaler: bool = False,
    use_quantile_transform: bool = False,
    n_quantiles: int = 100,
    cv_folds: int = 5,
    random_state: int = 42,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    log_to_wandb: bool = False,
    lr_penalty: str = 'l2',
    lr_C: float = 1.0,
    lr_solver: str = 'liblinear',
    rf_n_estimators: int = 100,
    rf_max_depth: Optional[int] = 10,
    rf_min_samples_split: int = 2,
    test_size: float = 0.2,
    output_dir: str = "models",
    save_predictions: bool = False
) -> None:
    """Trains a model, displays metrics, and saves the best model."""

    start_time = time.time()

    # --- Data Loading and Preprocessing ---
    with console.status("[bold green]Loading and preprocessing data..."):
        try:
            df = pd.read_json(data_path)
            log.info(f"Data loaded successfully from {data_path}")
        except (FileNotFoundError, pd.errors.EmptyDataError, Exception) as e:
            log.error(f"Error loading data: {e}")
            return

        df = df.drop(columns=['FQDN'])

        # --- KEY CHANGE: Map Overall_Score to binary (0 and 1) ---
        df['Overall_Score'] = df['Overall_Score'].map({1: 0, 2: 1})
        # ---------------------------------------------------------

        X = df.drop(columns=['Overall_Score'])
        y = df['Overall_Score']
        feature_names = X.columns

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')

        if len(numeric_cols) > 0:
             X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
             X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
        else:
             log.warning("No numeric columns found for imputation.")

        non_numeric_train = X_train.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_train) > 0:
            log.info(f"Dropping non-numeric columns from training set: {list(non_numeric_train)}")
            X_train = X_train.drop(columns=non_numeric_train)

        non_numeric_test = X_test.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_test) > 0:
            log.info(f"Dropping non-numeric columns from testing set: {list(non_numeric_test)}")
            X_test = X_test.drop(columns=non_numeric_test)
        
        feature_names = X_train.columns

    if log_to_wandb:
        run = wandb.init(project=wandb_project, entity=wandb_entity, config=locals())

    if model_type == 'gaussian_nb':
        model = GaussianNB()
    elif model_type == 'logistic_regression':
        model = LogisticRegression(penalty=lr_penalty, C=lr_C, solver=lr_solver, random_state=random_state, n_jobs=-1, max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, min_samples_split=rf_min_samples_split, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: gaussian_nb, logistic_regression, random_forest")

    progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    )

    best_model = None
    best_cv_score = -np.inf
    best_model_scaler = None
    best_model_transformer = None

    with progress:
        task = progress.add_task(f"[cyan]Cross-validating and training {model_type}...", total=cv_folds)
        cv_scores = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            X_train_fold_transformed = X_train_fold.copy()
            X_val_fold_transformed = X_val_fold.copy()

            if use_scaler:
                scaler = StandardScaler()
                X_train_fold_transformed[numeric_cols] = scaler.fit_transform(X_train_fold_transformed[numeric_cols])
                X_val_fold_transformed[numeric_cols] = scaler.transform(X_val_fold_transformed[numeric_cols])

            if use_quantile_transform:
                quantile_transformer = QuantileTransformer(n_quantiles=n_quantiles, random_state=random_state, output_distribution='normal')
                X_train_fold_transformed[numeric_cols] = quantile_transformer.fit_transform(X_train_fold_transformed[numeric_cols])
                X_val_fold_transformed[numeric_cols] = quantile_transformer.transform(X_val_fold_transformed[numeric_cols])

            model.fit(X_train_fold_transformed, y_train_fold)
            val_preds = model.predict(X_val_fold_transformed)
            fold_accuracy = accuracy_score(y_val_fold, val_preds)
            cv_scores.append(fold_accuracy)
            progress.update(task, advance=1, description=f"[cyan]Cross-validating {model_type}... (Fold {fold+1}/{cv_folds})")

            if fold_accuracy > best_cv_score:
                best_cv_score = fold_accuracy
                best_model = model
                if use_scaler:
                    best_model_scaler = scaler
                if use_quantile_transform:
                    best_model_transformer = quantile_transformer

        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        log.info(f"Cross-validation (accuracy) scores (mean ± std): {mean_cv_score:.4f} ± {std_cv_score:.4f}")

    if use_scaler:
        X_train[numeric_cols] = best_model_scaler.transform(X_train[numeric_cols])
        X_test[numeric_cols] = best_model_scaler.transform(X_test[numeric_cols])
    if use_quantile_transform:
        X_train[numeric_cols] = best_model_transformer.transform(X_train[numeric_cols])
        X_test[numeric_cols] = best_model_transformer.transform(X_test[numeric_cols])

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    if hasattr(best_model, "predict_proba"):
        y_pred_proba = best_model.predict_proba(X_test)
    else:
        y_pred_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'binary' for binary classification
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'binary'
    recall = recall_score(y_test, y_pred, average='weighted')      # Use 'binary'
    report = classification_report(y_test, y_pred)

    if y_pred_proba is not None:
        #  Binary classification, so use y_pred_proba[:, 1] (probability of class 1)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        logloss = log_loss(y_test, y_pred_proba[:, 1])
        brier = brier_score_loss(y_test, y_pred_proba[:, 1])
    else:
        roc_auc, logloss, brier = None, None, None

    cm = confusion_matrix(y_test, y_pred)

    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("Metric", style="cyan", width=20)
    results_table.add_column("Value", style="green", width=15)
    results_table.add_row("Accuracy", f"{accuracy:.4f}")
    results_table.add_row("F1 Score", f"{f1:.4f}")
    results_table.add_row("Precision", f"{precision:.4f}")
    results_table.add_row("Recall", f"{recall:.4f}")
    if roc_auc is not None:
        results_table.add_row("ROC AUC", f"{roc_auc:.4f}")
    if logloss is not None:
        results_table.add_row("Log Loss", f"{logloss:.4f}")
    if brier is not None:
        results_table.add_row("Brier Score", f"{brier:.4f}")
    results_table.add_row("CV Accuracy (Mean)", f"{mean_cv_score:.4f}")
    results_table.add_row("CV Accuracy (Std)", f"{std_cv_score:.4f}")

    console.print(results_table)
    console.print(Panel(f"\n[bold]Classification Report:[/]\n{report}"))
    console.print("\n[bold]Confusion Matrix:[/]")
    console.print(cm)
    plot_confusion_matrix(cm, classes=np.unique(y_test))

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance_table = get_feature_importance_table(importances, feature_names)
        console.print("\n[bold]Top 10 Feature Importances:[/]")
        console.print(feature_importance_table)
    elif hasattr(best_model, "coef_"):
        importances = best_model.coef_[0]
        feature_importance_table = get_feature_importance_table(importances, feature_names)
        console.print("\n[bold]Top 10 Feature Importances (Coefficients):[/]")
        console.print(feature_importance_table)
    else:
        console.print("\n[yellow]Feature importances not available for this model type.[/]")

    if log_to_wandb:
        wandb.log({
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
            "log_loss": logloss,
            "brier_score": brier,
            "cv_accuracy_mean": mean_cv_score,
            "cv_accuracy_std": std_cv_score,
            "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_test, preds=y_pred, class_names=np.unique(y_test)),
        })
        if y_pred_proba is not None:
            wandb.log({"roc_curve": wandb.plot.roc_curve(y_test, y_pred_proba, labels=np.unique(y_test))})
        if hasattr(best_model, "feature_importances_") or hasattr(best_model, "coef_"):
             wandb.log({"feature_importances": wandb.Table(dataframe=pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(10))})
        run.finish()

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(output_dir, f"{model_type}_best_model.joblib"))
    if best_model_scaler is not None:
        joblib.dump(best_model_scaler, os.path.join(output_dir, "scaler.joblib"))
    if best_model_transformer is not None:
        joblib.dump(best_model_transformer, os.path.join(output_dir, "quantile_transformer.joblib"))
    if imputer is not None:
        joblib.dump(imputer, os.path.join(output_dir, 'imputer.joblib'))

    if save_predictions:
        predictions_filepath = os.path.join(output_dir, 'predictions.csv')
        pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_csv(predictions_filepath, index=False)
        log.info(f"Predictions saved to {predictions_filepath}")

    log.info(f"Best model and preprocessing steps saved to: [green]{output_dir}[/]")
    end_time = time.time()
    console.print(f"\n[bold]Total execution time:[/bold] {end_time - start_time:.2f} seconds")



def get_feature_importance_table(importances, feature_names):
    """Creates a Rich table for feature importances."""
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Feature", style="magenta", width=30)
    table.add_column("Importance", style="green", width=12)

    sorted_indices = np.argsort(np.abs(importances))[::-1][:10]

    for rank, i in enumerate(sorted_indices):
        importance_str = f"{importances[i]:.4f}"
        table.add_row(str(rank + 1), feature_names[i], importance_str)
    return table

def main():
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument("data_path", type=str, help="Path to the JSON dataset")
    parser.add_argument("--model_type", type=str, default="random_forest", choices=['gaussian_nb', 'logistic_regression', 'random_forest'], help="Type of model to train")
    parser.add_argument("--use_scaler", action="store_true", help="Use StandardScaler")
    parser.add_argument("--use_quantile_transform", action="store_true", help="Use QuantileTransformer")
    parser.add_argument("--n_quantiles", type=int, default=100, help="Number of quantiles for QuantileTransformer")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--log_to_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--test_size", type=float, default=0.2, help="Size of the test dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the best model")

    parser.add_argument("--lr_penalty", type=str, default='l2', choices=['l1', 'l2', 'elasticnet', 'none'], help="Penalty for Logistic Regression")
    parser.add_argument("--lr_C", type=float, default=1.0, help="Inverse of regularization strength for Logistic Regression")
    parser.add_argument("--lr_solver", type=str, default='liblinear', choices=['liblinear', 'saga'], help="Solver for Logistic Regression")

    parser.add_argument("--rf_n_estimators", type=int, default=100, help="Number of trees in the Random Forest")
    parser.add_argument("--rf_max_depth", type=int, default=10, help="Maximum depth of trees in the Random Forest")
    parser.add_argument("--rf_min_samples_split", type=int, default=2, help="Minimum samples required to split a node in the Random Forest")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to a CSV file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        log.setLevel("DEBUG")
        log.debug("Verbose logging enabled.")

    train_model(
        args.data_path,
        args.model_type,
        args.use_scaler,
        args.use_quantile_transform,
        args.n_quantiles,
        args.cv_folds,
        args.random_state,
        args.wandb_project,
        args.wandb_entity,
        args.log_to_wandb,
        args.lr_penalty,
        args.lr_C,
        args.lr_solver,
        args.rf_n_estimators,
        args.rf_max_depth,
        args.rf_min_samples_split,
        args.test_size,
        args.output_dir,
        args.save_predictions
    )

if __name__ == "__main__":
    main()