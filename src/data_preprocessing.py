import pandas as pd
import numpy as np
import logging
import ipaddress
from intervaltree import IntervalTree

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_fraud_data(filepath='C:/Users/skibret/Downloads/KAIM/Week 8/Project/fraud_detection_project/data/raw/Fraud_Data.csv'):
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

def load_ip_country_data(filepath='C:/Users/skibret/Downloads/KAIM/Week 8/Project/fraud_detection_project/data/raw/IpAddress_to_Country.csv'):
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

def load_creditcard_data(filepath='C:/Users/skibret/Downloads/KAIM/Week 8/Project/fraud_detection_project/data/raw/creditcard.csv'):
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
    def convert_single_ip(ip):
        try:
            return int(ipaddress.IPv4Address(str(ip)))
        except:
            return np.nan
    return ip_address_series.apply(convert_single_ip)

def build_ip_interval_tree(ip_country_df):
    ip_country_df = ip_country_df.copy()
    ip_country_df['lower_bound_ip_address_int'] = ip_to_int(ip_country_df['lower_bound_ip_address'])
    ip_country_df['upper_bound_ip_address_int'] = ip_to_int(ip_country_df['upper_bound_ip_address'])

    tree = IntervalTree()
    for _, row in ip_country_df.iterrows():
        if not pd.isna(row['lower_bound_ip_address_int']) and not pd.isna(row['upper_bound_ip_address_int']):
            tree[row['lower_bound_ip_address_int']:row['upper_bound_ip_address_int'] + 1] = row['country']
    logger.info("Built IntervalTree for IP to country mapping.")
    return tree

def merge_fraud_with_ip_data(fraud_df, ip_country_df):
    if fraud_df is None or ip_country_df is None:
        logger.error("Cannot merge: one or both input dataframes are None.")
        return None

    logger.info("Starting fast IP-to-country mapping using IntervalTree...")
    fraud_df = fraud_df.copy()
    fraud_df['ip_address_int'] = ip_to_int(fraud_df['ip_address'])
    ip_tree = build_ip_interval_tree(ip_country_df)
    fraud_df['country'] = fraud_df['ip_address_int'].apply(lambda ip: list(ip_tree[ip])[0].data if ip in ip_tree else np.nan)

    logger.info(f"Completed mapping IP addresses to countries. Resulting shape: {fraud_df.shape}")
    return fraud_df

def handle_missing_values(df, strategy='drop'):
    if df is None:
        logger.warning("DataFrame is None, cannot handle missing values.")
        return None

    df_cleaned = df.copy()
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
    if df is None:
        logger.warning("DataFrame is None, cannot remove duplicates.")
        return None
    initial_rows = df.shape[0]
    df_cleaned = df.drop_duplicates().copy()
    logger.info(f"Removed {initial_rows - df_cleaned.shape[0]} duplicate rows.")
    return df_cleaned

def correct_data_types(df, datetime_cols=None, category_cols=None):
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
    if df is None:
        logger.warning("DataFrame is None, cannot create time features.")
        return None
    if timestamp_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        logger.error(f"Timestamp column '{timestamp_col}' not found or not in datetime format. Skipping time feature creation.")
        return df

    df_features = df.copy()
    df_features['hour_of_day'] = df_features[timestamp_col].dt.hour
    df_features['day_of_week'] = df_features[timestamp_col].dt.dayofweek
    logger.info(f"Created 'hour_of_day' and 'day_of_week' features from '{timestamp_col}'.")

    if 'signup_time' in df_features.columns and pd.api.types.is_datetime64_any_dtype(df_features['signup_time']):
        df_features['time_since_signup'] = (df_features[timestamp_col] - df_features['signup_time']).dt.total_seconds() / (24 * 3600)
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
        logger.error(f"Required columns '{user_id_col}' or '{timestamp_col}' not found.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        logger.error(f"Timestamp column '{timestamp_col}' is not in datetime format.")
        return df

    df_features = df.copy()
    df_features = df_features.sort_values(by=[user_id_col, timestamp_col])
    df_features['time_diff_prev_transaction'] = (
        df_features.groupby(user_id_col)[timestamp_col].diff().dt.total_seconds()
    )

    # Set index for rolling
    df_temp = df_features[[user_id_col, timestamp_col, 'purchase_value']].copy()
    df_temp = df_temp.set_index(timestamp_col)

    # --- Transaction frequency ---
    freq = (
        df_temp.groupby(user_id_col)
        .rolling(f'{window_days}D')
        .count()
        .rename(columns={"purchase_value": f'transactions_last_{window_days}d'})
        .reset_index()
    )

    # --- Transaction velocity ---
    velocity = (
        df_temp.groupby(user_id_col)['purchase_value']
        .rolling(f'{window_days}D')
        .sum()
        .reset_index()
        .rename(columns={"purchase_value": f'purchase_value_last_{window_days}d'})
    )

    # Merge both features
    df_merged = pd.merge(freq, velocity, on=[user_id_col, timestamp_col], how='outer')

    # Merge to main df on user_id and timestamp
    df_result = pd.merge(
        df_features,
        df_merged,
        how='left',
        left_on=[user_id_col, timestamp_col],
        right_on=[user_id_col, timestamp_col]
    )

    logger.info(f"Created frequency and velocity features for {window_days} day window.")
    return df_result


