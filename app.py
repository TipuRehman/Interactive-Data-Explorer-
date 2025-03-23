import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
from io import BytesIO
import base64
import os
from datetime import datetime

from data_loader import load_data
from data_profiler import profile_data, detect_outliers, plot_correlation
from data_transformer import apply_transformations
from data_exporter import export_data
from utils import generate_download_link

# Set page configuration
st.set_page_config(
    page_title="Interactive Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
    .metric-container {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #2c3e50;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'transformed_data' not in st.session_state:
    st.session_state.transformed_data = None

# Title and Introduction
st.title("📊 Interactive Data Explorer")
st.markdown("""
Upload your data files to explore, analyze, and transform them interactively.
Get insights through automatic profiling, visualization, and outlier detection.
""")

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    
    uploaded_file = st.file_uploader("Choose a CSV, Excel, or JSON file", 
                                     type=["csv", "xlsx", "xls", "json"])
    
    if uploaded_file is not None:
        # Get file name without extension
        file_name = os.path.splitext(uploaded_file.name)[0]
        
        # Load data
        with st.spinner("Loading data..."):
            data, message = load_data(uploaded_file)
            
            if data is not None:
                st.session_state.data = data
                st.session_state.file_name = file_name
                st.success(f"Successfully loaded {uploaded_file.name}")
                st.info(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
            else:
                st.error(message)
    
    # Sample Data Option
    st.markdown("---")
    st.subheader("Or try sample data")
    sample_option = st.selectbox("Select sample dataset:", 
                                ["None", "Iris", "Titanic", "Housing", "Sales"])
    
    if sample_option != "None":
        with st.spinner(f"Loading {sample_option} dataset..."):
            if sample_option == "Iris":
                data = sns.load_dataset("iris")
                st.session_state.file_name = "iris_dataset"
            elif sample_option == "Titanic":
                data = sns.load_dataset("titanic")
                st.session_state.file_name = "titanic_dataset"
            elif sample_option == "Housing":
                from sklearn.datasets import fetch_california_housing
                housing = fetch_california_housing()
                data = pd.DataFrame(housing.data, columns=housing.feature_names)
                data['target'] = housing.target
                st.session_state.file_name = "california_housing"
            elif sample_option == "Sales":
                # Create sample sales data
                np.random.seed(42)
                dates = pd.date_range('2021-01-01', '2022-12-31', freq='D')
                products = ['Product A', 'Product B', 'Product C', 'Product D']
                regions = ['North', 'South', 'East', 'West']
                
                sales_data = []
                for _ in range(1000):
                    date = np.random.choice(dates)
                    product = np.random.choice(products)
                    region = np.random.choice(regions)
                    units = np.random.randint(1, 100)
                    price = np.random.uniform(10, 1000)
                    sales_data.append({
                        'Date': date,
                        'Product': product,
                        'Region': region,
                        'Units': units,
                        'Price': price,
                        'Revenue': units * price
                    })
                data = pd.DataFrame(sales_data)
                st.session_state.file_name = "sales_sample"
            
            st.session_state.data = data
            st.success(f"Loaded {sample_option} dataset")
            st.info(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

# Main content
if st.session_state.data is not None:
    # Create tabs for different functionalities
    tabs = st.tabs(["Data Overview", "Data Profiling", "Visualization", "Transformation", "Export"])
    
    # Tab 1: Data Overview
    with tabs[0]:
        st.header("Data Overview")
        
        # Display basic information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.data.shape[0])
        with col2:
            st.metric("Columns", st.session_state.data.shape[1])
        with col3:
            st.metric("Missing Values", st.session_state.data.isna().sum().sum())
        
        # Display the data
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(100), use_container_width=True)
        
        # Display data types
        st.subheader("Data Types")
        dtypes_df = pd.DataFrame({
            'Column': st.session_state.data.dtypes.index,
            'Data Type': st.session_state.data.dtypes.values
        })
        st.dataframe(dtypes_df, use_container_width=True)
        
        # Display missing values
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
            'Column': st.session_state.data.columns,
            'Missing Values': st.session_state.data.isna().sum().values,
            'Percentage': (st.session_state.data.isna().sum().values / len(st.session_state.data) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)
        
        # Plot missing values
        if missing_df['Missing Values'].sum() > 0:
            st.subheader("Missing Values Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(st.session_state.data.isna(), yticklabels=False, cbar=False, cmap='viridis')
            st.pyplot(fig)
    
    # Tab 2: Data Profiling
    with tabs[1]:
        st.header("Data Profiling")
        
        # Compute profile if not already done
        if st.session_state.profile is None:
            with st.spinner("Generating data profile..."):
                st.session_state.profile = profile_data(st.session_state.data)
        
        # Display profile for each column
        for column in st.session_state.data.columns:
            with st.expander(f"Profile for: {column}"):
                if column in st.session_state.profile:
                    profile = st.session_state.profile[column]
                    
                    # Create columns for layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display basic statistics
                        st.subheader("Basic Statistics")
                        stats_df = pd.DataFrame({
                            'Statistic': profile.keys(),
                            'Value': profile.values()
                        })
                        st.dataframe(stats_df, use_container_width=True)
                    
                    with col2:
                        # Display distribution plot
                        st.subheader("Distribution")
                        if pd.api.types.is_numeric_dtype(st.session_state.data[column]):
                            fig = px.histogram(st.session_state.data, x=column, marginal="box")
                            st.plotly_chart(fig, use_container_width=True)
                        elif pd.api.types.is_categorical_dtype(st.session_state.data[column]) or st.session_state.data[column].nunique() < 10:
                            fig = px.bar(st.session_state.data[column].value_counts().reset_index(), 
                                         x='index', y=column, title=f"Count of {column}")
                            fig.update_layout(xaxis_title=column, yaxis_title='Count')
                            st.plotly_chart(fig, use_container_width=True)
        
        # Outlier Detection
        st.subheader("Outlier Detection")
        numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            col_for_outliers = st.selectbox("Select column for outlier detection:", numeric_columns)
            outliers, threshold = detect_outliers(st.session_state.data, col_for_outliers)
            
            if outliers.any():
                st.info(f"Found {outliers.sum()} outliers in {col_for_outliers} (threshold: {threshold:.2f})")
                
                # Plot with outliers highlighted
                fig = px.box(st.session_state.data, y=col_for_outliers)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display outliers
                st.subheader("Outlier Values")
                st.dataframe(st.session_state.data[outliers][[col_for_outliers]], use_container_width=True)
            else:
                st.success(f"No outliers detected in {col_for_outliers}")
        else:
            st.info("No numeric columns available for outlier detection")
    
    # Tab 3: Visualization
    with tabs[2]:
        st.header("Visualization")
        
        # Select visualization type
        viz_type = st.selectbox("Select visualization type:", 
                               ["Correlation Matrix", "Scatter Plot", "Line Chart", 
                                "Bar Chart", "Box Plot", "Histogram", "Pair Plot"])
        
        if viz_type == "Correlation Matrix":
            numeric_data = st.session_state.data.select_dtypes(include=['number'])
            if not numeric_data.empty:
                st.subheader("Correlation Matrix")
                fig = plot_correlation(numeric_data)
                st.pyplot(fig)
            else:
                st.warning("No numeric columns available for correlation analysis")
        
        elif viz_type == "Scatter Plot":
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y-axis:", [col for col in numeric_cols if col != x_col], index=min(1, len(numeric_cols)-1))
                with col3:
                    color_col = st.selectbox("Color by (optional):", ["None"] + st.session_state.data.columns.tolist())
                
                color = None if color_col == "None" else color_col
                fig = px.scatter(st.session_state.data, x=x_col, y=y_col, color=color, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for scatter plot")
        
        elif viz_type == "Line Chart":
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            date_cols = [col for col in st.session_state.data.columns if pd.api.types.is_datetime64_dtype(st.session_state.data[col])]
            
            # Try to identify date columns
            for col in st.session_state.data.columns:
                if not pd.api.types.is_datetime64_dtype(st.session_state.data[col]):
                    try:
                        pd.to_datetime(st.session_state.data[col])
                        date_cols.append(col)
                    except:
                        pass
            
            if date_cols and numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis (Date):", date_cols)
                with col2:
                    y_col = st.selectbox("Y-axis (Value):", numeric_cols)
                
                # Convert to datetime if needed
                chart_data = st.session_state.data.copy()
                if not pd.api.types.is_datetime64_dtype(chart_data[x_col]):
                    chart_data[x_col] = pd.to_datetime(chart_data[x_col])
                
                fig = px.line(chart_data, x=x_col, y=y_col, title=f"{y_col} over time")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need date column and numeric column for line chart")
        
        elif viz_type == "Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Categories (X-axis):", st.session_state.data.columns.tolist())
            with col2:
                y_col = st.selectbox("Values (Y-axis):", ["Count"] + st.session_state.data.select_dtypes(include=['number']).columns.tolist())
            
            if y_col == "Count":
                count_data = st.session_state.data[x_col].value_counts().reset_index()
                count_data.columns = [x_col, 'Count']
                fig = px.bar(count_data, x=x_col, y='Count', title=f"Count of {x_col}")
            else:
                fig = px.bar(st.session_state.data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Value (Y-axis):", numeric_cols)
            with col2:
                x_col = st.selectbox("Group by (X-axis, optional):", ["None"] + categorical_cols)
            
            if x_col == "None":
                fig = px.box(st.session_state.data, y=y_col, title=f"Box Plot of {y_col}")
            else:
                fig = px.box(st.session_state.data, x=x_col, y=y_col, title=f"Box Plot of {y_col} by {x_col}")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Histogram":
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    col = st.selectbox("Select column:", numeric_cols)
                with col2:
                    bins = st.slider("Number of bins:", min_value=5, max_value=100, value=20)
                
                fig = px.histogram(st.session_state.data, x=col, nbins=bins, title=f"Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for histogram")
        
        elif viz_type == "Pair Plot":
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 1:
                columns_to_use = st.multiselect("Select columns to plot:", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
                if len(columns_to_use) >= 2:
                    color_col = st.selectbox("Color by (optional):", ["None"] + st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist())
                    color = None if color_col == "None" else color_col
                    
                    if len(columns_to_use) > 5:
                        st.warning("Using many columns may slow down the visualization")
                    
                    with st.spinner("Generating pair plot..."):
                        fig = px.scatter_matrix(st.session_state.data, dimensions=columns_to_use, color=color)
                        fig.update_layout(title="Pair Plot")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least 2 columns")
            else:
                st.warning("Need at least 2 numeric columns for pair plot")
    
    # Tab 4: Transformation
    with tabs[3]:
        st.header("Data Transformation")
        
        # Select transformation operation
        operation = st.selectbox("Select transformation:", 
                                ["Handle Missing Values", "Remove Outliers", 
                                 "Scale/Normalize", "Encode Categorical Variables", 
                                 "Filter Data", "Create New Column", "Drop Columns"])
        
        # For tracking the transformed data
        if st.session_state.transformed_data is None:
            st.session_state.transformed_data = st.session_state.data.copy()
        
        if operation == "Handle Missing Values":
            # Select columns to handle missing values
            col1, col2 = st.columns(2)
            with col1:
                missing_cols = st.multiselect("Select columns:", 
                                             st.session_state.transformed_data.columns[st.session_state.transformed_data.isna().any()].tolist())
            with col2:
                handling_method = st.selectbox("Method:", ["Drop", "Fill with mean", "Fill with median", "Fill with mode", "Fill with value"])
            
            fill_value = None
            if handling_method == "Fill with value":
                fill_value = st.text_input("Enter fill value:")
            
            if st.button("Apply Missing Value Handling"):
                with st.spinner("Applying transformation..."):
                    st.session_state.transformed_data = apply_transformations(
                        st.session_state.transformed_data, 
                        operation="missing_values",
                        columns=missing_cols,
                        method=handling_method,
                        fill_value=fill_value
                    )
                st.success("Successfully handled missing values")
        
        elif operation == "Remove Outliers":
            numeric_cols = st.session_state.transformed_data.select_dtypes(include=['number']).columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                outlier_cols = st.multiselect("Select columns:", numeric_cols)
            with col2:
                method = st.selectbox("Method:", ["IQR", "Z-Score"])
                threshold = st.slider("Threshold:", 1.5, 5.0, 3.0, 0.1)
            
            if st.button("Remove Outliers"):
                with st.spinner("Removing outliers..."):
                    st.session_state.transformed_data = apply_transformations(
                        st.session_state.transformed_data,
                        operation="remove_outliers",
                        columns=outlier_cols,
                        method=method,
                        threshold=threshold
                    )
                st.success("Successfully removed outliers")
        
        elif operation == "Scale/Normalize":
            numeric_cols = st.session_state.transformed_data.select_dtypes(include=['number']).columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                scale_cols = st.multiselect("Select columns:", numeric_cols)
            with col2:
                scaling_method = st.selectbox("Method:", ["Min-Max Scaling", "Standard Scaling", "Log Transform"])
            
            if st.button("Apply Scaling"):
                with st.spinner("Applying scaling..."):
                    st.session_state.transformed_data = apply_transformations(
                        st.session_state.transformed_data,
                        operation="scale",
                        columns=scale_cols,
                        method=scaling_method
                    )
                st.success("Successfully applied scaling")
        
        elif operation == "Encode Categorical Variables":
            cat_cols = st.session_state.transformed_data.select_dtypes(include=['object', 'category']).columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                encode_cols = st.multiselect("Select columns:", cat_cols)
            with col2:
                encoding_method = st.selectbox("Method:", ["One-Hot Encoding", "Label Encoding"])
            
            if st.button("Apply Encoding"):
                with st.spinner("Applying encoding..."):
                    st.session_state.transformed_data = apply_transformations(
                        st.session_state.transformed_data,
                        operation="encode",
                        columns=encode_cols,
                        method=encoding_method
                    )
                st.success("Successfully applied encoding")
        
        elif operation == "Filter Data":
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_col = st.selectbox("Select column:", st.session_state.transformed_data.columns.tolist())
            
            with col2:
                condition = st.selectbox("Condition:", ["equal to", "not equal to", "greater than", "less than", "contains"])
            
            with col3:
                filter_value = st.text_input("Value:")
            
            if st.button("Apply Filter"):
                with st.spinner("Filtering data..."):
                    st.session_state.transformed_data = apply_transformations(
                        st.session_state.transformed_data,
                        operation="filter",
                        column=filter_col,
                        condition=condition,
                        value=filter_value
                    )
                st.success("Successfully filtered data")
        
        elif operation == "Create New Column":
            col1, col2 = st.columns(2)
            with col1:
                new_col_name = st.text_input("New column name:")
            with col2:
                formula_type = st.selectbox("Formula type:", ["Expression", "Combine Columns"])
            
            if formula_type == "Expression":
                cols = st.session_state.transformed_data.columns.tolist()
                info_text = "Available columns: " + ", ".join([f"`{col}`" for col in cols])
                st.info(info_text)
                formula = st.text_area("Enter formula (use column names in backticks):", height=100)
                
                formula_example = "Example: `column1` * 2 + `column2`"
                st.caption(formula_example)
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    col_a = st.selectbox("First column:", st.session_state.transformed_data.columns.tolist())
                with col2:
                    operation_type = st.selectbox("Operation:", ["+", "-", "*", "/", "Concatenate"])
                with col3:
                    col_b = st.selectbox("Second column:", st.session_state.transformed_data.columns.tolist())
                
                formula = f"`{col_a}` {operation_type} `{col_b}`"
                if operation_type == "Concatenate":
                    formula = f"`{col_a}`.astype(str) + `{col_b}`.astype(str)"
            
            if st.button("Create New Column"):
                with st.spinner("Creating new column..."):
                    st.session_state.transformed_data = apply_transformations(
                        st.session_state.transformed_data,
                        operation="new_column",
                        new_column_name=new_col_name,
                        formula=formula
                    )
                st.success(f"Successfully created new column: {new_col_name}")
        
        elif operation == "Drop Columns":
            drop_cols = st.multiselect("Select columns to drop:", st.session_state.transformed_data.columns.tolist())
            
            if st.button("Drop Columns"):
                with st.spinner("Dropping columns..."):
                    st.session_state.transformed_data = apply_transformations(
                        st.session_state.transformed_data,
                        operation="drop_columns",
                        columns=drop_cols
                    )
                st.success("Successfully dropped columns")
        
        # Display preview of transformed data
        st.subheader("Transformed Data Preview")
        st.dataframe(st.session_state.transformed_data.head(10), use_container_width=True)
        
        # Show transformation summary
        if st.session_state.transformed_data is not None:
            st.info(f"Original data shape: {st.session_state.data.shape}, Transformed data shape: {st.session_state.transformed_data.shape}")
        
        # Reset transformations button
        if st.button("Reset Transformations"):
            st.session_state.transformed_data = st.session_state.data.copy()
            st.success("Reset to original data")
    
    # Tab 5: Export
    with tabs[4]:
        st.header("Export Data")
        
        # Choose which data to export
        export_data_option = st.radio("Select data to export:", ["Original Data", "Transformed Data"])
        
        data_to_export = st.session_state.data if export_data_option == "Original Data" else st.session_state.transformed_data
        
        # Export format options
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox("Export format:", ["CSV", "Excel", "JSON"])
        with col2:
            file_name = st.text_input("File name:", st.session_state.file_name + "_exported")
        
        # Additional options based on format
        if export_format == "CSV":
            col1, col2 = st.columns(2)
            with col1:
                sep = st.selectbox("Separator:", [",", ";", "\\t", "|"])
            with col2:
                index = st.checkbox("Include index", value=False)
        
        elif export_format == "Excel":
            col1, col2 = st.columns(2)
            with col1:
                sheet_name = st.text_input("Sheet name:", "Data")
            with col2:
                index = st.checkbox("Include index", value=False)
        
        elif export_format == "JSON":
            orient = st.selectbox("JSON orientation:", ["records", "columns", "index", "split", "table"])
        
        # Export button
        if st.button("Export Data"):
            with st.spinner("Preparing export..."):
                # Call the export function with the selected options
                if export_format == "CSV":
                    download_link = export_data(data_to_export, format="csv", file_name=file_name, 
                                             sep=sep, index=index)
                elif export_format == "Excel":
                    download_link = export_data(data_to_export, format="excel", file_name=file_name, 
                                             sheet_name=sheet_name, index=index)
                elif export_format == "JSON":
                    download_link = export_data(data_to_export, format="json", file_name=file_name, 
                                             orient=orient)
                
                st.success("Data ready for export")
                st.markdown(download_link, unsafe_allow_html=True)
else:
    # If no data is loaded, show instructions
    st.info("👈 Please upload a data file or select a sample dataset from the sidebar to get started.")
    
    # Show key features
    st.subheader("Key Features")
    
    features = [
        {"icon": "📊", "title": "Data Profiling", "description": "Automatically analyze your data to get insights about distributions, statistics, and patterns."},
        {"icon": "📈", "title": "Interactive Visualizations", "description": "Create various charts and plots to visually explore your data."},
        {"icon": "🔍", "title": "Outlier Detection", "description": "Identify and handle outliers in your numeric data."},
        {"icon": "🔄", "title": "Data Transformation", "description": "Clean, transform, and prepare your data with various operations."},
        {"icon": "💾", "title": "Export Options", "description": "Export your processed data in various formats for further use."}
    ]
    
    for feature in features:
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown(f"### {feature['icon']}")
        with col2:
            st.subheader(feature["title"])
            st.markdown(feature["description"])
    
    # Footer
    st.markdown("---")
    st.caption("Interactive Data Explorer | Built with Streamlit")