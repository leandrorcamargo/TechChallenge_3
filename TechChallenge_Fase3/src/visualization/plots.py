"""Módulo para visualizações."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix


def plot_distribuicao_target(y, title='Distribuição do Target'):
    """Plota distribuição da variável target."""
    import pandas as pd
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if isinstance(y, pd.Series):
        counts = y.value_counts()
    else:
        import numpy as np
        unique, counts = np.unique(y, return_counts=True)
        counts = pd.Series(counts, index=unique)
    
    ax.bar(counts.index, counts.values, color=['#e74c3c', '#2ecc71'])
    ax.set_xlabel('Classe')
    ax.set_ylabel('Frequência')
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Não Atingiu', 'Atingiu Meta'])
    
    for i, v in enumerate(counts.values):
        pct = v / counts.sum() * 100
        ax.text(i, v + 200, f'{v:,}\n({pct:.1f}%)', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plota matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Valores absolutos
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Atinge', 'Atinge'],
                yticklabels=['Não Atinge', 'Atinge'], ax=axes[0])
    axes[0].set_title('Matriz de Confusão (absolutos)')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predito')
    
    # Valores percentuais
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=['Não Atinge', 'Atinge'],
                yticklabels=['Não Atinge', 'Atinge'], ax=axes[1])
    axes[1].set_title('Matriz de Confusão (percentual)')
    axes[1].set_ylabel('Real')
    axes[1].set_xlabel('Predito')
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba):
    """Plota curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.4f}')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Aleatório')
    ax.set_xlabel('FPR (Taxa de Falsos Positivos)')
    ax.set_ylabel('TPR (Taxa de Verdadeiros Positivos)')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_proba):
    """Plota curva Precision-Recall."""
    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (prec_curve[:-1] * rec_curve[:-1]) / (prec_curve[:-1] + rec_curve[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Curva PR
    axes[0].plot(rec_curve, prec_curve, color='darkcyan', lw=2)
    axes[0].plot(rec_curve[best_idx], prec_curve[best_idx], 'ro', markersize=8,
                 label=f'Melhor F1={f1_scores[best_idx]:.3f}')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Curva Precision-Recall')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Métricas vs Threshold
    axes[1].plot(thresholds_pr, prec_curve[:-1], label='Precision')
    axes[1].plot(thresholds_pr, rec_curve[:-1], label='Recall')
    axes[1].plot(thresholds_pr, f1_scores, label='F1')
    axes[1].axvline(thresholds_pr[best_idx], color='gray', linestyle='--')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Métricas vs Threshold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Melhor F1: {f1_scores[best_idx]:.4f} (threshold={thresholds_pr[best_idx]:.4f})")


def plot_feature_importance(df_importance, top_n=15):
    """Plota importância das features."""
    df_top = df_importance.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(df_top)), df_top['importance'], color='steelblue')
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importância')
    ax.set_title(f'Top {top_n} Features Mais Importantes')
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
