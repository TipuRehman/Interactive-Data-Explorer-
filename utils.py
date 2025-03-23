import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime

def generate_download_link(df, file_name, file_format="csv"):
    """
    Generate a download link for a dataframe
    Returns an HTML link
    """
    if file_format == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv" class="download-button">Download CSV</a>'
        return href
    
    elif file_format == "excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
        
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{file_name}.xlsx" class="download-button">Download Excel</a>'
        return href
    
    elif file_format == "json":
        json_str = df.to_json(orient='records')
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{file_name}.json" class="download-button">Download JSON</a>'
        return href
    
    return ""

def format_number(value):
    """
    Format a number with thousands separators and proper decimal places
    """
    if pd.isna(value):
        return "N/A"
    
    if isinstance(value, (int, np.integer)):
        return f"{value:,d}"
    elif isinstance(value, (float, np.floating)):
        if value == int(value):
            return f"{int(value):,d}"
        elif abs(value) < 0.01:
            return f"{value:.6f}"
        else:
            return f"{value:,.2f}"
    else:
        return str(value)

def get_file_size(num_bytes):
    """
    Convert bytes to human-readable file size
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"

def infer_datetime_format(df, column):
    """
    Try to infer datetime format from a column
    Returns True if conversion is successful
    """
    try:
        # Try pandas automatic parsing
        pd.to_datetime(df[column])
        return True
    except:
        # Try common datetime formats
        formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y',
            '%Y/%m/%d', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S'
        ]
        
        for format in formats:
            try:
                pd.to_datetime(df[column], format=format)
                return True
            except:
                continue
    
    return False

def detect_outliers_iqr(series):
    """
    Detect outliers using the IQR method
    Returns a boolean mask where True indicates an outlier
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return (series < lower_bound) | (series > upper_bound)

def detect_encoding(file_path):
    """
    Attempt to detect file encoding
    Returns most likely encoding
    """
    from chardet.universaldetector import UniversalDetector
    
    detector = UniversalDetector()
    with open(file_path, 'rb') as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    
    return detector.result['encoding']