import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
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
        
        # Handle different visualization types
        if viz_type == "Correlation Matrix":
            # Add code for correlation matrix visualization
            st.subheader("Correlation Matrix")
            numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            if numeric_columns:
                corr_matrix = st.session_state.data[numeric_columns].corr()
                fig = plot_correlation(corr_matrix)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available for correlation analysis")
        
        elif viz_type == "Scatter Plot":
            # Code for scatter plot
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis:", st.session_state.data.select_dtypes(include=['number']).columns.tolist())
            with col2:
                y_col = st.selectbox("Y-axis:", [c for c in st.session_state.data.select_dtypes(include=['number']).columns.tolist() if c != x_col])
            with col3:
                color_cols = ["None"] + st.session_state.data.columns.tolist()
                color_col = st.selectbox("Color by:", color_cols)
            
            fig = px.scatter(
                st.session_state.data, x=x_col, y=y_col, 
                color=None if color_col == "None" else color_col,
                title=f"{y_col} vs {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart":
            # Code for line chart
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis (time/sequence):", st.session_state.data.columns.tolist())
            with col2:
                y_cols = st.multiselect("Y-axis (values):", st.session_state.data.select_dtypes(include=['number']).columns.tolist())
            
            if y_cols:
                fig = px.line(st.session_state.data, x=x_col, y=y_cols, title=f"Line Chart of {', '.join(y_cols)} over {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one column for Y-axis")
        
        elif viz_type == "Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Categories (X-axis):", st.session_state.data.columns.tolist())
            with col2:
                # Dynamically adjust Y-axis options based on column types
                y_options = ["Count"]
                if st.session_state.data.select_dtypes(include=['number']).columns.tolist():
                    y_options.extend(st.session_state.data.select_dtypes(include=['number']).columns.tolist())
                y_col = st.selectbox("Values (Y-axis):", y_options)
            
            try:
                if y_col == "Count":
                    # Create count data with proper index reset and column naming
                    count_data = st.session_state.data[x_col].value_counts().reset_index()
                    count_data.columns = ['Category', 'Count']
                    
                    # Sort data for better visualization
                    count_data = count_data.sort_values('Count', ascending=False)
                    
                    # Create bar plot with improved handling
                    fig = px.bar(
                        count_data, 
                        x='Category', 
                        y='Count', 
                        title=f"Count of {x_col}",
                        labels={'Category': x_col, 'Count': 'Frequency'}
                    )
                else:
                    # For numeric value aggregation
                    agg_data = st.session_state.data.groupby(x_col)[y_col].agg(['mean', 'median', 'count']).reset_index()
                    agg_data.columns = [x_col, 'Mean', 'Median', 'Count']
                    
                    # Sort by mean value for better visualization
                    agg_data = agg_data.sort_values('Mean', ascending=False)
                    
                    # Create bar plot with multiple metrics
                    fig = px.bar(
                        agg_data, 
                        x=x_col, 
                        y='Mean', 
                        title=f"{y_col} by {x_col}",
                        hover_data=['Median', 'Count']
                    )
                
                # Improve plot readability
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title='Value',
                    height=500,
                    width=800
                )
                
                # Rotate x-axis labels if many categories
                if len(st.session_state.data[x_col].unique()) > 10:
                    fig.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error creating bar chart: {str(e)}")
        
        elif viz_type == "Box Plot":
            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Values to plot:", st.session_state.data.select_dtypes(include=['number']).columns.tolist())
            with col2:
                group_cols = ["None"] + [c for c in st.session_state.data.columns if c != y_col and st.session_state.data[c].nunique() < 20]
                group_col = st.selectbox("Group by:", group_cols)
            
            if group_col == "None":
                fig = px.box(st.session_state.data, y=y_col, title=f"Box Plot of {y_col}")
            else:
                fig = px.box(st.session_state.data, y=y_col, x=group_col, title=f"Box Plot of {y_col} by {group_col}")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Histogram":
            col1, col2 = st.columns(2)
            with col1:
                hist_col = st.selectbox("Select column:", st.session_state.data.select_dtypes(include=['number']).columns.tolist())
            with col2:
                bins = st.slider("Number of bins:", min_value=5, max_value=100, value=30)
            
            fig = px.histogram(st.session_state.data, x=hist_col, nbins=bins, title=f"Histogram of {hist_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pair Plot":
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 1:
                selected_cols = st.multiselect("Select columns (2-5 recommended):", numeric_cols, default=numeric_cols[:3])
                
                if len(selected_cols) > 1:
                    color_cols = ["None"] + [c for c in st.session_state.data.columns if c not in selected_cols and st.session_state.data[c].nunique() < 10]
                    color_col = st.selectbox("Color by:", color_cols)
                    
                    # Create pair plot using plotly
                    fig = px.scatter_matrix(
                        st.session_state.data, 
                        dimensions=selected_cols,
                        color=None if color_col == "None" else color_col,
                        title="Pair Plot"
                    )
                    fig.update_layout(height=800)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least 2 columns for pair plot")
            else:
                st.info("Not enough numeric columns for pair plot")
    
    # Tab 4: Transformation
    with tabs[3]:
        st.header("Data Transformation")
        
        # Add transformation options here
        transform_options = ["Filter Data", "Sort Data", "Handle Missing Values", 
                             "Create New Column", "Drop Columns", "Rename Columns"]
        transform_choice = st.selectbox("Select transformation:", transform_options)
        
        # Apply transformations based on choice
        if transform_choice == "Filter Data":
            st.subheader("Filter Data")
            # Filtering implementation here
            col_to_filter = st.selectbox("Select column to filter:", st.session_state.data.columns)
            
            # Adjust filter UI based on column type
            if pd.api.types.is_numeric_dtype(st.session_state.data[col_to_filter]):
                min_val = float(st.session_state.data[col_to_filter].min())
                max_val = float(st.session_state.data[col_to_filter].max())
                filter_range = st.slider(f"Filter range for {col_to_filter}:", 
                                        min_value=min_val, max_value=max_val, 
                                        value=(min_val, max_val))
                filter_condition = (st.session_state.data[col_to_filter] >= filter_range[0]) & (st.session_state.data[col_to_filter] <= filter_range[1])
            else:
                unique_values = st.session_state.data[col_to_filter].unique().tolist()
                selected_values = st.multiselect(f"Select values for {col_to_filter}:", 
                                                unique_values, default=unique_values)
                filter_condition = st.session_state.data[col_to_filter].isin(selected_values)
            
            filtered_data = st.session_state.data[filter_condition]
            st.write(f"Filtered data has {filtered_data.shape[0]} rows")
            st.dataframe(filtered_data.head(100), use_container_width=True)
            
            if st.button("Apply Filter"):
                st.session_state.transformed_data = filtered_data
                st.success("Filter applied successfully!")
        
        # Add other transformation options here
        
        # Display transformed data if available
        if st.session_state.transformed_data is not None:
            st.subheader("Transformed Data Preview")
            st.dataframe(st.session_state.transformed_data.head(100), use_container_width=True)
    
    # Tab 5: Export
    with tabs[4]:
        st.header("Export Data")
        
        # Export options
        export_options = ["CSV", "Excel", "JSON"]
        export_format = st.selectbox("Select export format:", export_options)
        
        # Select data to export
        data_to_export = st.radio("Data to export:", 
                                  ["Original Data", "Transformed Data" if st.session_state.transformed_data is not None else "Original Data"])
        
        # Export button
        if st.button("Export Data"):
            export_data = st.session_state.transformed_data if data_to_export == "Transformed Data" and st.session_state.transformed_data is not None else st.session_state.data
            
            # Generate file for download
            if export_format == "CSV":
                result = export_data(export_data, "csv", st.session_state.file_name)
            elif export_format == "Excel":
                result = export_data(export_data, "excel", st.session_state.file_name)
            elif export_format == "JSON":
                result = export_data(export_data, "json", st.session_state.file_name)
            
            if result["success"]:
                st.success("Data exported successfully!")
                st.markdown(generate_download_link(result["data"], result["filename"]), unsafe_allow_html=True)
            else:
                st.error(f"Export failed: {result['message']}")

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
