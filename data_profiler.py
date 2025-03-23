import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

def profile_data(df):
    """
    Generate a comprehensive profile for each column in the dataframe
    Returns a dictionary with column profiles
    """
    profile = {}
    
    # Iterate through columns
    for column in df.columns:
        col_profile = {}
        
        # Basic information
        col_profile['total_values'] = len(df[column])
        col_profile['missing_values'] = df[column].isna().sum()
        col_profile['missing_percentage'] = (col_profile['missing_values'] / col_profile['total_values'] * 100).round(2)
        col_profile['unique_values'] = df[column].nunique()
        col_profile['data_type'] = str(df[column].dtype)
        
        # Numeric columns specific profile
        if pd.api.types.is_numeric_dtype(df[column]):
            col_profile['min'] = df[column].min()
            col_profile['max'] = df[column].max()
            col_profile['mean'] = df[column].mean()
            col_profile['median'] = df[column].median()
            col_profile['std'] = df[column].std()
            col_profile['skewness'] = df[column].skew()
            col_profile['kurtosis'] = df[column].kurtosis()
        
        # Categorical columns specific profile
        elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == 'object':
            col_profile['top_5_values'] = df[column].value_counts().head().to_dict()
        
        # Datetime columns specific profile
        elif pd.api.types.is_datetime64_dtype(df[column]):
            col_profile['min_date'] = df[column].min()
            col_profile['max_date'] = df[column].max()
        
        profile[column] = col_profile
    
    return profile

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a numeric column
    Returns boolean mask where True indicates an outlier
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return np.zeros(len(df), dtype=bool), None
    
    # IQR method (Interquartile Range)
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Identify outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        return outliers, lower_bound
    
    # Z-Score method
    elif method == 'z-score':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
        return outliers, threshold
    
    return np.zeros(len(df), dtype=bool), None

def plot_correlation(df):
    """
    Create a correlation heatmap for numeric columns
    Returns the matplotlib figure
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number'])
    
    # Compute correlation matrix
    corr_matrix = numeric_cols.corr()
    
    # Create figure and plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    
    return plt
