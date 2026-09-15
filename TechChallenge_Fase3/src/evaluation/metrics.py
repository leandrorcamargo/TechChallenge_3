"""Módulo para cálculo de métricas de avaliação."""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calcula métricas de classificação."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def print_metrics(y_true, y_pred, y_proba=None, dataset_name=''):
    """Imprime métricas de forma formatada."""
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    print(f"\n{'='*60}")
    if dataset_name:
        print(f"MÉTRICAS - {dataset_name}")
    else:
        print("MÉTRICAS")
    print(f"{'='*60}")
    
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"AUC-ROC:   {metrics['roc_auc']:.4f}")
    
    print(f"{'='*60}\n")
    
    return metrics


def print_confusion_matrix(y_true, y_pred):
    """Imprime matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    print("Matriz de Confusão:")
    print(f"  TN={TN:,} | FP={FP:,}")
    print(f"  FN={FN:,} | TP={TP:,}")
    print(f"\n  Sensibilidade: {TP/(TP+FN)*100:.2f}%")
    print(f"  Especificidade: {TN/(TN+FP)*100:.2f}%\n")
    
    return cm


def print_classification_report(y_true, y_pred):
    """Imprime relatório de classificação."""
    print("\nRelatório de Classificação:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Não Alfabetizado', 'Alfabetizado']
    ))


def get_feature_importance(pipeline, feature_names, top_n=15):
    """Extrai feature importance do modelo."""
    import pandas as pd
    
    modelo = pipeline.named_steps['modelo']
    importances = modelo.feature_importances_
    
    if len(feature_names) != len(importances):
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    df_imp = df_imp.sort_values('importance', ascending=False)
    
    return df_imp.head(top_n)
