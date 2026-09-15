"""Módulo para construção de pipelines de ML."""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(num_features, cat_features):
    """Constrói preprocessador para features numéricas e categóricas."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    
    return preprocessor


def build_xgboost_pipeline(X_train, modelo_tipo='agregado'):
    """Constrói pipeline completo com XGBoost."""
    from xgboost import XGBClassifier
    
    # Separar features numéricas e categóricas
    num_features = X_train.select_dtypes(include=['int32', 'int64', 'float64']).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Construir preprocessador
    preprocessor = build_preprocessor(num_features, cat_features)
    
    # Configurar XGBoost
    if modelo_tipo == 'agregado':
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    else:  # individual
        xgb = XGBClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            tree_method='hist'
        )
    
    # Pipeline completo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('modelo', xgb)
    ])
    
    return pipeline, num_features, cat_features


def get_feature_names(pipeline, num_features, cat_features):
    """Obtém nomes das features após preprocessamento."""
    try:
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_names = ohe.get_feature_names_out(cat_features).tolist()
    except:
        cat_names = cat_features
    
    return num_features + cat_names
