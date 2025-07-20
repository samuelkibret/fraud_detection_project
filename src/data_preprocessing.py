# src/data_preprocessing.py

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_fraud_data(filepath='C:/Users/skibret/Downloads/KAIM/Week 8/Project/fraud_detection_project/data/raw/Fraud_Data.csv'): # Corrected path
    """
    Loads the e-commerce fraud dataset.
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded Fraud_Data.csv from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Fraud_Data.csv not found at {filepath}. Please ensure the file exists and path is correct.")
        return None
    except Exception as e:
        logger.error(f"Error loading Fraud_Data.csv: {e}")
        return None

def load_ip_country_data(filepath='C:/Users/skibret/Downloads/KAIM/Week 8/Project/fraud_detection_project/data/raw/IpAddress_to_Country.csv'): # Corrected path
    """
    Loads the IP address to country mapping dataset.
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded IpAddress_to_Country.csv from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"IpAddress_to_Country.csv not found at {filepath}. Please ensure the file exists and path is correct.")
        return None
    except Exception as e:
        logger.error(f"Error loading IpAddress_to_Country.csv: {e}")
        return None

def load_creditcard_data(filepath='C:/Users/skibret/Downloads/KAIM/Week 8/Project/fraud_detection_project/data/raw/creditcard.csv'): # Corrected path
    """
    Loads the credit card fraud dataset.
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded creditcard.csv from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"creditcard.csv not found at {filepath}. Please ensure the file exists and path is correct.")
        return None
    except Exception as e:
        logger.error(f"Error loading creditcard.csv: {e}")
        return None

def ip_to_int(ip_address_series):
    """
    Converts a series of IP addresses from string to integer format.
    Handles non-string/NaN values gracefully.
    """
    def convert_single_ip(ip):
        if pd.isna(ip):
            return np.nan
        try:
            # Ensure it's a string before splitting
            ip_str = str(ip)
            # Basic validation for IPv4 format (e.g., contains dots)
            if '.' not in ip_str:
                logger.warning(f"Invalid IP format '{ip_str}'. Returning NaN.")
                return np.nan
            # Convert each octet to binary and join, then convert binary string to int
            return int("".join([f'{int(x):08b}' for x in ip_str.split('.')]), 2)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not convert IP '{ip}' to int: {e}. Returning NaN.")
            return np.nan

    return ip_address_series.apply(convert_single_ip)

def merge_fraud_with_ip_data(fraud_df, ip_country_df):
    """
    Merges the fraud data with IP address to country mapping.
    Assumes ip_country_df has 'lower_bound_ip_address', 'upper_bound_ip_address', 'country'.
    """
    if fraud_df is None or ip_country_df is None:
        logger.error("Cannot merge: one or both input dataframes are None.")
        return None

    logger.info("Starting merge of fraud data with IP address to country mapping...")

    # Convert IP address columns to integer for efficient merging
    fraud_df['ip_address_int'] = ip_to_int(fraud_df['ip_address'])
    ip_country_df['lower_bound_ip_address_int'] = ip_to_int(ip_country_df['lower_bound_ip_address'])
    ip_country_df['upper_bound_ip_address_int'] = ip_to_int(ip_country_df['upper_bound_ip_address'])

    # Sort IP country data for efficient range lookup (important for merge_asof)
    ip_country_df = ip_country_df.sort_values('lower_bound_ip_address_int')

    # Perform a merge_asof to find the country for each IP address
    # This is suitable for range-based lookups
    merged_df = pd.merge_asof(
        fraud_df.sort_values('ip_address_int'),
        ip_country_df[['lower_bound_ip_address_int', 'upper_bound_ip_address_int', 'country']],
        left_on='ip_address_int',
        right_on='lower_bound_ip_address_int',
        direction='backward' # Finds the last row where left_on <= right_on
    )

    # Filter out rows where the IP address is not within the found range
    merged_df = merged_df[
        (merged_df['ip_address_int'] >= merged_df['lower_bound_ip_address_int']) &
        (merged_df['ip_address_int'] <= merged_df['upper_bound_ip_address_int'])
    ].copy() # Add .copy() to avoid SettingWithCopyWarning

    logger.info(f"Merge complete. Merged dataframe shape: {merged_df.shape}")
    return merged_df

def handle_missing_values(df, strategy='drop'):
    """
    Handles missing values based on the specified strategy.
    'drop': drops rows with any missing values.
    'impute_median': imputes numerical columns with median.
    'impute_mode': imputes categorical columns with mode.
    """
    if df is None:
        logger.warning("DataFrame is None, cannot handle missing values.")
        return None

    df_cleaned = df.copy() # Work on a copy to avoid modifying original df
    initial_rows = df_cleaned.shape[0]

    if strategy == 'drop':
        df_cleaned.dropna(inplace=True)
        logger.info(f"Dropped {initial_rows - df_cleaned.shape[0]} rows with missing values.")
    elif strategy == 'impute_median':
        for col in df_cleaned.select_dtypes(include=np.number).columns:
            if df_cleaned[col].isnull().any():
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                logger.info(f"Imputed missing values in numerical column '{col}' with median: {median_val}")
    elif strategy == 'impute_mode':
        for col in df_cleaned.select_dtypes(include='object').columns:
            if df_cleaned[col].isnull().any():
                mode_val = df_cleaned[col].mode()[0]
                df_cleaned[col].fillna(mode_val, inplace=True)
                logger.info(f"Imputed missing values in categorical column '{col}' with mode: {mode_val}")
    else:
        logger.warning(f"Unknown missing value strategy: {strategy}. No action taken.")

    return df_cleaned

def remove_duplicates(df):
    """
    Removes duplicate rows from the DataFrame.
    """
    if df is None:
        logger.warning("DataFrame is None, cannot remove duplicates.")
        return None

    initial_rows = df.shape[0]
    df_cleaned = df.drop_duplicates().copy()
    logger.info(f"Removed {initial_rows - df_cleaned.shape[0]} duplicate rows.")
    return df_cleaned

def correct_data_types(df, datetime_cols=None, category_cols=None):
    """
    Corrects data types for specified columns.
    datetime_cols: list of columns to convert to datetime.
    category_cols: list of columns to convert to category.
    """
    if df is None:
        logger.warning("DataFrame is None, cannot correct data types.")
        return None

    df_corrected = df.copy()
    if datetime_cols:
        for col in datetime_cols:
            if col in df_corrected.columns:
                df_corrected[col] = pd.to_datetime(df_corrected[col], errors='coerce')
                logger.info(f"Converted '{col}' to datetime.")
            else:
                logger.warning(f"Datetime column '{col}' not found in DataFrame.")

    if category_cols:
        for col in category_cols:
            if col in df_corrected.columns:
                df_corrected[col] = df_corrected[col].astype('category')
                logger.info(f"Converted '{col}' to category.")
            else:
                logger.warning(f"Category column '{col}' not found in DataFrame.")
    return df_corrected

def create_time_features(df, timestamp_col):
    """
    Creates time-based features from a timestamp column.
    Adds 'hour_of_day', 'day_of_week', 'time_since_signup' (if signup_time exists).
    """
    if df is None:
        logger.warning("DataFrame is None, cannot create time features.")
        return None
    if timestamp_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        logger.error(f"Timestamp column '{timestamp_col}' not found or not in datetime format. Skipping time feature creation.")
        return df

    df_features = df.copy()
    df_features['hour_of_day'] = df_features[timestamp_col].dt.hour
    df_features['day_of_week'] = df_features[timestamp_col].dt.dayofweek # Monday=0, Sunday=6
    logger.info(f"Created 'hour_of_day' and 'day_of_week' features from '{timestamp_col}'.")

    if 'signup_time' in df_features.columns and pd.api.types.is_datetime64_any_dtype(df_features['signup_time']):
        df_features['time_since_signup'] = (df_features[timestamp_col] - df_features['signup_time']).dt.total_seconds() / (24 * 3600) # in days
        logger.info("Created 'time_since_signup' feature (in days).")
    else:
        logger.warning("'signup_time' column not found or not in datetime format. 'time_since_signup' not created.")

    return df_features

def create_transaction_frequency_velocity(df, user_id_col, timestamp_col, window_days=7):
    """
    Calculates transaction frequency and velocity for each user.
    Frequency: number of transactions in a given window.
    Velocity: sum of purchase values in a given window.
    """
    if df is None:
        logger.warning("DataFrame is None, cannot create frequency/velocity features.")
        return None
    if user_id_col not in df.columns or timestamp_col not in df.columns:
        logger.error(f"Required columns '{user_id_col}' or '{timestamp_col}' not found for frequency/velocity.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        logger.error(f"Timestamp column '{timestamp_col}' not in datetime format for frequency/velocity.")
        return df

    df_features = df.copy()
    df_features = df_features.sort_values(by=[user_id_col, timestamp_col])

    # Calculate time difference to previous transaction for each user (optional, but good for sequential analysis)
    df_features['time_diff_prev_transaction'] = df_features.groupby(user_id_col)[timestamp_col].diff().dt.total_seconds()

    # Calculate rolling features
    # For frequency, count transactions in a rolling window
    df_features[f'transactions_last_{window_days}d'] = df_features.groupby(user_id_col)[timestamp_col].rolling(
        f'{window_days}D', on=timestamp_col
    ).count().reset_index(level=0, drop=True)

    # For velocity, sum purchase_value in a rolling window
    if 'purchase_value' in df_features.columns:
        df_features[f'purchase_value_last_{window_days}d'] = df_features.groupby(user_id_col)['purchase_value'].rolling(
            f'{window_days}D', on=timestamp_col
        ).sum().reset_index(level=0, drop=True)
    else:
        logger.warning("'purchase_value' column not found for velocity calculation.")

    logger.info(f"Created transaction frequency and velocity features for last {window_days} days.")
    return df_features