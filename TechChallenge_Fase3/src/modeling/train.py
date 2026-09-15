"""Módulo para treinar modelos."""
from sklearn.model_selection import StratifiedKFold, cross_val_score


def train_model(pipeline, X_train, y_train, cross_validate=True):
    """Treina modelo com validação cruzada opcional."""
    if cross_validate:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    pipeline.fit(X_train, y_train)
    print('Modelo treinado com sucesso.')
    return pipeline


def predict(pipeline, X_test):
    """Realiza predições."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def otimizar_threshold(y_test, y_proba):
    """Encontra melhor threshold para maximizar accuracy."""
    import numpy as np
    from sklearn.metrics import accuracy_score
    
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_acc = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        acc = accuracy_score(y_test, y_pred_thresh)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    
    print(f"Melhor threshold: {best_threshold:.3f} (accuracy: {best_acc:.4f})")
    return best_threshold, best_acc
