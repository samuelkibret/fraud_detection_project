# src/model_explainability.py

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.sparse import issparse # NEW IMPORT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def explain_model_shap(model, X_data, feature_names=None):
    """
    Generates SHAP values for a given model and dataset.
    Handles different model types (tree-based, linear, etc.).
    Ensures X_data is a dense, numeric NumPy array for SHAP.
    """
    logger.info(f"Generating SHAP explanations for model: {type(model).__name__}...")

    # --- NEW: Ensure X_data is a dense, numeric NumPy array ---
    if issparse(X_data):
        X_data_dense = X_data.toarray()
        logger.info("Converted sparse X_data to dense array for SHAP.")
    elif isinstance(X_data, pd.DataFrame):
        X_data_dense = X_data.values # Convert DataFrame to NumPy array
        logger.info("Converted DataFrame X_data to NumPy array for SHAP.")
    else:
        X_data_dense = X_data # Assume it's already a NumPy array

    # Ensure the dtype is float
    if X_data_dense.dtype != np.float64 and X_data_dense.dtype != np.float32:
        X_data_dense = X_data_dense.astype(np.float64)
        logger.info(f"Converted X_data dtype to {X_data_dense.dtype} for SHAP.")
    # --- END NEW ---

    # For tree-based models (RandomForest, GradientBoosting, DecisionTree)
    if "Tree" in type(model).__name__ or "Forest" in type(model).__name__ or "Boosting" in type(model).__name__:
        explainer = shap.TreeExplainer(model)
    # For linear models (Logistic Regression)
    # Note: LinearExplainer can sometimes be sensitive to sparse inputs as well,
    # so passing the dense array is safer.
    elif "LogisticRegression" in type(model).__name__:
        explainer = shap.LinearExplainer(model, X_data_dense) # Pass dense array here
    # For other models (e.g., MLPClassifier, or if TreeExplainer/LinearExplainer don't apply)
    else:
        logger.warning("Using KernelExplainer, which can be slow for large datasets. Consider sampling X_data.")
        # KernelExplainer needs a background dataset. Use a sample of the dense data.
        explainer = shap.KernelExplainer(
            model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
            shap.sample(X_data_dense, 100) # Pass dense array sample
        )

    # Compute SHAP values using the dense array
    shap_values = explainer.shap_values(X_data_dense) # Use dense array here

    logger.info("SHAP value generation complete.")
    return explainer, shap_values

def plot_shap_summary(shap_values, feature_names, plot_type='dot', max_display=20, title="SHAP Feature Importance Summary"):
    """
    Generates and displays a SHAP summary plot.
    """
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # For multi-class or models returning list (e.g., predict_proba for binary)
        # We usually care about the SHAP values for the positive class (class 1)
        shap_values_to_plot = shap_values[1] # Assuming binary classification, index 1 for positive class
    else:
        shap_values_to_plot = shap_values

    logger.info(f"Generating SHAP summary plot (type: {plot_type})...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_to_plot,
        features=pd.DataFrame(shap_values_to_plot, columns=feature_names), # Pass DataFrame for proper feature display
        feature_names=feature_names,
        plot_type=plot_type,
        max_display=max_display,
        show=False # Don't show immediately, allow customization
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    logger.info("SHAP summary plot displayed.")

def plot_shap_dependence(explainer, shap_values, feature, X_data, feature_names, interaction_feature=None, title="SHAP Dependence Plot"):
    """
    Generates and displays a SHAP dependence plot for a single feature.
    """
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    logger.info(f"Generating SHAP dependence plot for feature: {feature}...")
    plt.figure(figsize=(10, 6))
    # Ensure X_data is a DataFrame for dependence_plot
    if not isinstance(X_data, pd.DataFrame):
        X_data_df = pd.DataFrame(X_data, columns=feature_names)
    else:
        X_data_df = X_data

    shap.dependence_plot(
        ind=feature,
        shap_values=shap_values_to_plot,
        features=X_data_df, # Pass DataFrame for proper plotting
        feature_names=feature_names,
        interaction_index=interaction_feature,
        show=False
    )
    plt.title(f"{title}: {feature}", fontsize=16)
    plt.tight_layout()
    plt.show()
    logger.info(f"SHAP dependence plot for {feature} displayed.")

def plot_shap_force(explainer, shap_values, X_data, feature_names, row_index=0, title="SHAP Force Plot for Individual Prediction"):
    """
    Generates and displays a SHAP force plot for a single prediction.
    """
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    logger.info(f"Generating SHAP force plot for row index: {row_index}...")
    # Force plot requires the explainer and raw data row
    # Ensure X_data is a DataFrame with feature names for proper display
    if not isinstance(X_data, pd.DataFrame):
        X_data_df = pd.DataFrame(X_data, columns=feature_names)
    else:
        X_data_df = X_data

    # Use JS visualization for force plot
    shap.initjs()
    # For a single instance, pass the explainer, shap_values for that instance, and the raw instance data
    # If model.predict_proba was used, explainer.expected_value will be an array, pick the positive class
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) > 1 else explainer.expected_value

    # Display the force plot
    display(shap.force_plot(
        expected_value,
        shap_values_to_plot[row_index],
        X_data_df.iloc[row_index],
        matplotlib=False # Render as JS in notebooks
    ))
    logger.info(f"SHAP force plot for row index {row_index} displayed.")