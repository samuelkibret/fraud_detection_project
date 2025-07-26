# src/model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline # Alias to avoid conflict with sklearn.pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data(filepath):
    """
    Loads a processed dataset.
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded processed data from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Processed data not found at {filepath}. Please ensure the file exists.")
        return None
    except Exception as e:
        logger.error(f"Error loading processed data from {filepath}: {e}")
        return None

def split_data(df, target_column, test_size=0.3, random_state=42):
    """
    Splits the DataFrame into features (X) and target (y), then into
    training and testing sets. It also identifies numerical and categorical columns
    from the X dataframe *after* the target column is dropped.
    """
    if df is None:
        logger.error("DataFrame is None, cannot split data.")
        return None, None, None, None, None, None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify categorical and numerical columns *from this X* for return
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    logger.info(f"Splitting data into training and testing sets (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logger.info(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
    logger.info(f"Testing target distribution:\n{y_test.value_counts(normalize=True)}")

    return X_train, X_test, y_train, y_test, categorical_cols, numerical_cols


def handle_imbalance(X_train, y_train, strategy='SMOTE', random_state=42):
    """
    Handles class imbalance in the training data using specified strategy.
    'SMOTE': Synthetic Minority Over-sampling Technique
    'RandomUnderSampler': Randomly undersamples the majority class
    'SMOTE_and_Undersample': Combines SMOTE with RandomUnderSampler
    Assumes X_train has NO NaNs before this function is called (NaN handling should be done prior).
    """
    if X_train is None or y_train is None:
        logger.error("X_train or y_train is None, cannot handle imbalance.")
        return None, None

    # CRITICAL: SMOTE and other imblearn samplers do NOT accept NaNs.
    # NaN handling MUST be done before calling this function, typically
    # in the preprocessing pipeline that feeds into X_train.
    # If X_train is a sparse matrix, it will not have .isnull() method.

    logger.info(f"Handling class imbalance using strategy: {strategy}...")
    initial_minority_count = y_train.value_counts().get(1, 0) # Use .get() for robustness
    initial_majority_count = y_train.value_counts().get(0, 0)

    if strategy == 'SMOTE':
        sampler = SMOTE(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif strategy == 'RandomUnderSampler':
        sampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif strategy == 'SMOTE_and_Undersample':
        # This pipeline first oversamples, then undersamples to a desired ratio
        # Adjust sampling_strategy based on your specific imbalance and desired outcome
        over = SMOTE(sampling_strategy=0.1, random_state=random_state) # Oversample minority to 10% of majority
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state) # Undersample majority to 50% of new minority
        steps = [('o', over), ('u', under)]
        pipeline = ImbPipeline(steps=steps) # Using ImbPipeline
        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    else:
        logger.warning(f"Unknown imbalance handling strategy: {strategy}. No resampling applied.")
        return X_train, y_train

    logger.info(f"Resampling complete. Original minority count: {initial_minority_count}, majority count: {initial_majority_count}")
    logger.info(f"Resampled data shape: {X_resampled.shape}, target shape: {y_resampled.shape}")
    logger.info(f"Resampled target distribution:\n{y_resampled.value_counts(normalize=True)}")

    return X_resampled, y_resampled

def evaluate_model(model, X_test, y_test, y_pred, y_prob):
    """
    Evaluates a trained model and prints key metrics.
    """
    logger.info("Evaluating model performance...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {roc_auc:.4f}")
    logger.info("\nConfusion Matrix:\n" + str(cm))
    logger.info("\nClassification Report:\n" + report)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }
# src/model_training.py (continued)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# For deep learning models (CNN, RNN/LSTM), you would typically use TensorFlow/Keras or PyTorch
# We'll add placeholders for these, but their implementation is more involved.
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
# from tensorflow.keras.optimizers import Adam # Example optimizer

def train_logistic_regression(X_train, y_train, random_state=42, **kwargs):
    """Trains a Logistic Regression model."""
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(random_state=random_state, solver='liblinear', **kwargs) # 'liblinear' is good for small datasets and L1/L2 regularization
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return model

def train_decision_tree(X_train, y_train, random_state=42, **kwargs):
    """Trains a Decision Tree Classifier model."""
    logger.info("Training Decision Tree Classifier model...")
    model = DecisionTreeClassifier(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    logger.info("Decision Tree Classifier training complete.")
    return model

def train_random_forest(X_train, y_train, random_state=42, n_estimators=100, **kwargs):
    """Trains a Random Forest Classifier model."""
    logger.info("Training Random Forest Classifier model...")
    model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, n_jobs=-1, **kwargs) # n_jobs=-1 uses all available cores
    model.fit(X_train, y_train)
    logger.info("Random Forest Classifier training complete.")
    return model

def train_gradient_boosting(X_train, y_train, random_state=42, n_estimators=100, learning_rate=0.1, **kwargs):
    """Trains a Gradient Boosting Classifier model."""
    logger.info("Training Gradient Boosting Classifier model...")
    model = GradientBoostingClassifier(random_state=random_state, n_estimators=n_estimators, learning_rate=learning_rate, **kwargs)
    model.fit(X_train, y_train)
    logger.info("Gradient Boosting Classifier training complete.")
    return model


# Placeholder for Deep Learning models (CNN/RNN/LSTM)
# These would typically require TensorFlow/Keras or PyTorch setup
# and would be more complex, involving model architecture definition.
# We will address these if you decide to implement them.

# def build_cnn_model(input_shape):
#     model = Sequential([
#         Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape),
#         MaxPooling1D(pool_size=2),
#         Flatten(),
#         Dense(50, activation='relu'),
#         Dropout(0.5),
#         Dense(1, activation='sigmoid') # Binary classification
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def build_rnn_lstm_model(input_shape):
#     model = Sequential([
#         LSTM(50, activation='relu', input_shape=input_shape),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model
# Placeholder for model training functions (will add these later)
# def train_logistic_regression(X_train, y_train):
#     pass

# def train_decision_tree(X_train, y_train):
#     pass

# ... and so on for other models