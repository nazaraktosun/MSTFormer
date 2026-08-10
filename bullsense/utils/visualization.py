"""
Visualization functions
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_history(history, save_path='training_curves.png'):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved: {save_path}")
    plt.show()


def plot_confusion_matrix(cm, target_names=None, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = np.asarray(cm)
    if cm.ndim != 2:
        raise ValueError("Confusion matrix must be 2-dimensional.")

    if target_names is None or len(target_names) != cm.shape[0]:
        target_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(9, 7))
    
    # Calculate percentages
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_normalized = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
    
    # Create annotations with both counts and percentages
    annotations = []
    for i in range(cm.shape[0]):
        row_annot = []
        for j in range(cm.shape[1]):
            row_annot.append(f"{int(cm[i, j])}\n({cm_normalized[i, j]:.1%})")
        annotations.append(row_annot)
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names,
                cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='gray')
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved: {save_path}")
    plt.show()


def plot_class_distribution(labels, title='Class Distribution', save_path=None):
    """Plot class distribution"""
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ['STABLE', 'UP', 'DOWN']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(unique)), counts, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique])
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        pct = count / sum(counts) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution saved: {save_path}")
    
    plt.show()


def plot_experiment_comparison(results_list, metric='test_acc', 
                               labels=None, save_path='comparison.png'):
    """Compare multiple experiments"""
    if labels is None:
        labels = [f"Exp {i+1}" for i in range(len(results_list))]
    
    values = [r[metric] for r in results_list]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(values)), values, color='steelblue')
    
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Experiment Comparison: {metric}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(values)), labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved: {save_path}")
    plt.show()
