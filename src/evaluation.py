"""
Evaluation helpers: metrics, confusion matrix, calibration, error analysis plots.

This module provides reusable functions for model evaluation and visualisation.
All plotting functions optionally save figures to the outputs/figures/ directory
and return the figure object for further customisation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.calibration import calibration_curve


# ── Configuration ────────────────────────────────────────────────────────────
# Resolve figure directory relative to the project root (parent of src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(_PROJECT_ROOT, 'outputs', 'figures')

# Human-readable labels for the 5 purchasing power quintiles
QUINTILE_LABELS = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']


# ── Classification metrics ──────────────────────────────────────────────────

def print_classification_metrics(y_true, y_pred, model_name: str = "Model"):
    """
    Print accuracy, macro F1, and full per-class classification report.

    Macro F1 is the primary metric because it treats all quintiles equally,
    unlike accuracy which can be dominated by well-predicted classes.
    The function returns a dict of metrics for downstream comparison.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # Print formatted results with model name as header
    print(f"\n{'='*50}")
    print(f"{model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    # Per-class precision, recall, F1 — using quintile labels for readability
    print(f"\n{classification_report(y_true, y_pred, target_names=QUINTILE_LABELS)}")

    return {'accuracy': acc, 'macro_f1': f1}


# ── Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model",
                          normalize: bool = True, save: bool = True):
    """
    Plot a confusion matrix heatmap showing prediction patterns.

    When normalised (default), each cell shows the proportion of true labels
    predicted as each class. This reveals which quintiles are most often
    confused — typically adjacent quintiles (e.g. Q2 predicted as Q3).
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalise by row (true label) to show recall-like proportions
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=QUINTILE_LABELS,
                yticklabels=QUINTILE_LABELS, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()

    # Save to outputs/figures/ with a sanitised filename
    if save:
        fig.savefig(f'{FIGURE_DIR}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── Model comparison bar chart ──────────────────────────────────────────────

def plot_model_comparison(results: dict, metric: str = 'macro_f1', save: bool = True):
    """
    Horizontal bar chart comparing models on a given metric.

    Parameters
    ----------
    results : dict
        {model_name: {'accuracy': float, 'macro_f1': float, ...}}
        Typically populated by calling print_classification_metrics for each model.
    metric : str
        Which metric to plot (default: 'macro_f1').
    """
    names = list(results.keys())
    values = [results[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, values, color='steelblue')
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(f'Model Comparison — {metric.replace("_", " ").title()}')
    ax.set_xlim(0, 1)  # metrics are between 0 and 1

    # Annotate each bar with its numeric value for quick reading
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center')

    plt.tight_layout()
    if save:
        fig.savefig(f'{FIGURE_DIR}/model_comparison_{metric}.png',
                    dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── Feature importance ──────────────────────────────────────────────────────

def plot_feature_importance(importances, feature_names, top_n: int = 20,
                           model_name: str = "Model", save: bool = True):
    """
    Plot top N feature importances as a horizontal bar chart.

    Uses the feature_importances_ attribute from tree-based models
    (Random Forest, Gradient Boosting) to show which variables contribute
    most to predictions. Higher importance = more useful for splitting.
    """
    # Sort features by importance and take the top N
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), importances[idx], color='steelblue')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances — {model_name}')
    plt.tight_layout()

    if save:
        fig.savefig(f'{FIGURE_DIR}/feature_importance_{model_name.lower().replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── Learning curves ─────────────────────────────────────────────────────────

def plot_learning_curves(train_sizes, train_scores, val_scores,
                         model_name: str = "Model", save: bool = True):
    """
    Plot learning curves (training vs validation score as a function of
    training set size).

    Learning curves diagnose:
    - Underfitting: both curves are low and close together
    - Overfitting: training score is high but validation score is low
    - Good fit: both curves converge at a reasonable level

    Shaded regions show ±1 standard deviation across CV folds.
    """
    # Compute mean and standard deviation across cross-validation folds
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot ±1 std deviation bands for visual uncertainty indication
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')

    # Plot mean scores
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation score')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title(f'Learning Curves — {model_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(f'{FIGURE_DIR}/learning_curves_{model_name.lower().replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── Calibration curves ──────────────────────────────────────────────────────

def plot_calibration(y_true, y_prob, n_classes: int = 5,
                     model_name: str = "Model", save: bool = True):
    """
    Plot calibration curves (reliability diagram) for each class.

    A well-calibrated model produces predicted probabilities that match
    observed frequencies. For example, if the model predicts P(Q5)=0.7
    for a group of samples, ~70% of those samples should actually be Q5.

    The diagonal dashed line represents perfect calibration.
    Points above the diagonal = under-confident; below = over-confident.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the perfect calibration reference line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    # Plot one calibration curve per class (quintile)
    for cls in range(n_classes):
        # Convert to binary "is this class?" for calibration analysis
        y_binary = (y_true == cls).astype(int)

        # Extract predicted probability for this class
        if y_prob.ndim == 2:
            prob = y_prob[:, cls]
        else:
            continue  # skip if probabilities are not per-class

        # Compute calibration curve: bin predictions into 10 buckets
        # and compare mean predicted probability vs actual positive fraction
        fraction_pos, mean_predicted = calibration_curve(
            y_binary, prob, n_bins=10, strategy='uniform'
        )
        ax.plot(mean_predicted, fraction_pos, 's-', label=QUINTILE_LABELS[cls])

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Curves — {model_name}')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(f'{FIGURE_DIR}/calibration_{model_name.lower().replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
    plt.show()
    return fig
