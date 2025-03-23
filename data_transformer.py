import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re

def apply_transformations(df, operation, **kwargs):
    """
    Apply various data transformations to the dataframe
    Returns the transformed dataframe
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Handle missing values
    if operation == "missing_values":
        columns = kwargs.get("columns", [])
        method = kwargs.get("method", "Drop")
        fill_value = kwargs.get("fill_value", None)
        
        for col in columns:
            if col in result_df.columns:
                if method == "Drop":
                    result_df = result_df.dropna(subset=[col])
                elif method == "Fill with mean" and pd.api.types.is_numeric_dtype(result_df[col]):
                    result_df[col] = result_df[col].fillna(result_df[col].mean())
                elif method == "Fill with median" and pd.api.types.is_numeric_dtype(result_df[col]):
                    result_df[col] = result_df[col].fillna(result_df[col].median())
                elif method == "Fill with mode":
                    result_df[col] = result_df[col].fillna(result_df[col].mode()[0])
                elif method == "Fill with value":
                    # Try to convert fill_value to the column's data type
                    try:
                        if pd.api.types.is_numeric_dtype(result_df[col]):
                            fill_val = float(fill_value)
                        elif pd.api.types.is_datetime64_dtype(result_df[col]):
                            fill_val = pd.to_datetime(fill_value)
                        else:
                            fill_val = str(fill_value)
                        result_df[col] = result_df[col].fillna(fill_val)
                    except:
                        # If conversion fails, use the string value
                        result_df[col] = result_df[col].fillna(fill_value)
    
    # Remove outliers
    elif operation == "remove_outliers":
        columns = kwargs.get("columns", [])
        method = kwargs.get("method", "IQR")
        threshold = kwargs.get("threshold", 1.5)
        
        for col in columns:
            if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col]):
                if method == "IQR":
                    Q1 = result_df[col].quantile(0.25)
                    Q3 = result_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # Filter out outliers
                    result_df = result_df[(result_df[col] >= lower_bound) & (result_df[col] <= upper_bound)]
                
                elif method == "Z-Score":
                    # Calculate z-scores and filter based on threshold
                    z_scores = np.abs((result_df[col] - result_df[col].mean()) / result_df[col].std())
                    result_df = result_df[z_scores <= threshold]
    
    # Scale/Normalize
    elif operation == "scale":
        columns = kwargs.get("columns", [])
        method = kwargs.get("method", "Min-Max Scaling")
        
        for col in columns:
            if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col]):
                # Handle NaN values for scaling
                col_data = result_df[col].values.reshape(-1, 1)
                not_nan = ~np.isnan(col_data).flatten()
                
                if method == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                    # Scale non-NaN values
                    if np.any(not_nan):
                        col_data[not_nan] = scaler.fit_transform(col_data[not_nan].reshape(-1, 1))
                    result_df[col] = col_data
                
                elif method == "Standard Scaling":
                    scaler = StandardScaler()
                    # Scale non-NaN values
                    if np.any(not_nan):
                        col_data[not_nan] = scaler.fit_transform(col_data[not_nan].reshape(-1, 1))
                    result_df[col] = col_data
                
                elif method == "Log Transform":
                    # Handle zeros and negative values
                    if (result_df[col] <= 0).any():
                        min_val = result_df[col].min()
                        offset = 1
                        if min_val <= 0:
                            offset = abs(min_val) + 1
                        result_df[col] = np.log(result_df[col] + offset)
                    else:
                        result_df[col] = np.log(result_df[col])
    
    # Encode categorical variables
    elif operation == "encode":
        columns = kwargs.get("columns", [])
        method = kwargs.get("method", "One-Hot Encoding")
        
        for col in columns:
            if col in result_df.columns:
                if method == "One-Hot Encoding":
                    # Create dummies and drop original column
                    dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=False)
                    result_df = pd.concat([result_df, dummies], axis=1)
                    result_df = result_df.drop(columns=[col])
                
                elif method == "Label Encoding":
                    # Use category codes as labels
                    result_df[col] = pd.Categorical(result_df[col]).codes
    
    # Filter data
    elif operation == "filter":
        column = kwargs.get("column", "")
        condition = kwargs.get("condition", "")
        value = kwargs.get("value", "")
        
        if column in result_df.columns:
            if pd.api.types.is_numeric_dtype(result_df[column]):
                try:
                    value = float(value)
                except:
                    return result_df  # Return original if conversion fails
            
            if condition == "equal to":
                result_df = result_df[result_df[column] == value]
            elif condition == "not equal to":
                result_df = result_df[result_df[column] != value]
            elif condition == "greater than" and pd.api.types.is_numeric_dtype(result_df[column]):
                result_df = result_df[result_df[column] > value]
            elif condition == "less than" and pd.api.types.is_numeric_dtype(result_df[column]):
                result_df = result_df[result_df[column] < value]
            elif condition == "contains" and (pd.api.types.is_string_dtype(result_df[column]) or pd.api.types.is_categorical_dtype(result_df[column])):
                result_df = result_df[result_df[column].astype(str).str.contains(str(value), case=False)]
    
    # Create new column
    elif operation == "new_column":
        new_column_name = kwargs.get("new_column_name", "new_column")
        formula = kwargs.get("formula", "")
        
        if new_column_name and formula:
            try:
                # Replace column references with actual references to the dataframe
                pattern = r'`([^`]+)`'
                
                def replace_col_ref(match):
                    col_name = match.group(1)
                    return f"result_df['{col_name}']"
                
                formula_code = re.sub(pattern, replace_col_ref, formula)
                
                # Evaluate the formula
                result_df[new_column_name] = eval(formula_code)
            except Exception as e:
                print(f"Error creating new column: {str(e)}")
    
    # Drop columns
    elif operation == "drop_columns":
        columns = kwargs.get("columns", [])
        if columns:
            result_df = result_df.drop(columns=columns)
    
    return result_df