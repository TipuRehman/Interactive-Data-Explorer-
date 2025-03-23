import pandas as pd
import base64
import io
from datetime import datetime
import os

def export_data(df, format="csv", file_name="exported_data", **kwargs):
    """
    Export dataframe to different formats and generate a download link
    Returns the HTML download link for the data
    """
    if format == "csv":
        # Get CSV options
        sep = kwargs.get("sep", ",")
        index = kwargs.get("index", False)
        
        # Generate CSV data
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, sep=sep, index=index)
        csv_data = csv_buffer.getvalue()
        
        # Convert to base64
        b64 = base64.b64encode(csv_data.encode()).decode()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{file_name}_{timestamp}.csv"
        
        # Create download link
        href = f'<a href="data:text/csv;base64,{b64}" download="{file_name}" class="btn">Download CSV</a>'
        return href
    
    elif format == "excel":
        # Get Excel options
        sheet_name = kwargs.get("sheet_name", "Data")
        index = kwargs.get("index", False)
        
        # Generate Excel data
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
            writer.save()
        
        # Convert to base64
        excel_data = excel_buffer.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{file_name}_{timestamp}.xlsx"
        
        # Create download link
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{file_name}" class="btn">Download Excel</a>'
        return href
    
    elif format == "json":
        # Get JSON options
        orient = kwargs.get("orient", "records")
        
        # Generate JSON data
        json_data = df.to_json(orient=orient)
        
        # Convert to base64
        b64 = base64.b64encode(json_data.encode()).decode()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{file_name}_{timestamp}.json"
        
        # Create download link
        href = f'<a href="data:application/json;base64,{b64}" download="{file_name}" class="btn">Download JSON</a>'
        return href
    
    # Default return if format not supported
    return ""