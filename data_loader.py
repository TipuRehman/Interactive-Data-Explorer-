import pandas as pd
import json
import os

def load_data(uploaded_file):
    """
    Load data from various file formats (CSV, Excel, JSON)
    Returns the loaded dataframe and a success/error message
    """
    try:
        # Get file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Load based on file type
        if file_extension in ['.csv', '.txt']:
            # Try different delimiters
            try:
                df = pd.read_csv(uploaded_file)
            except:
                try:
                    df = pd.read_csv(uploaded_file, sep=';')
                except:
                    try:
                        df = pd.read_csv(uploaded_file, sep='\t')
                    except:
                        return None, "Could not determine the delimiter for this CSV file."
        
        elif file_extension in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                return None, f"Error reading Excel file: {str(e)}"
        
        elif file_extension == '.json':
            try:
                data = json.load(uploaded_file)
                df = pd.json_normalize(data)
            except:
                try:
                    # Try reading as a JSON lines file
                    df = pd.read_json(uploaded_file, lines=True)
                except Exception as e:
                    return None, f"Error reading JSON file: {str(e)}"
        else:
            return None, f"Unsupported file format: {file_extension}"
        
        # Validate dataframe
        if df.empty:
            return None, "The file is empty."
        
        # Convert date columns to datetime
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        return df, "Success"
    
    except Exception as e:
        return None, f"Error loading data: {str(e)}"