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
from data_profiler import detect_outliers  # Removed plot_correlation import
from data_transformer import apply_transformations
from data_exporter import export_data
from utils import generate_download_link

# Define a function to profile data (to replace the imported one)
def profile_data(df):
    """
    Generate profile for each column in the dataframe
    """
    profile = {}
    
    for column in df.columns:
        col_profile = {}
        
        # Basic counts
        col_profile['count'] = df[column].count()
        col_profile['missing'] = df[column].isna().sum()
        col_profile['missing_pct'] = round(df[column].isna().sum() / len(df) * 100, 2)
        
        if pd.api.types.is_numeric_dtype(df[column]):
            # Numeric statistics
            col_profile['mean'] = round(df[column].mean(), 2) if not pd.isna(df[column].mean()) else None
            col_profile['median'] = round(df[column].median(), 2) if not pd.isna(df[column].median()) else None
            col_profile['std'] = round(df[column].std(), 2) if not pd.isna(df[column].std()) else None
            col_profile['min'] = round(df[column].min(), 2) if not pd.isna(df[column].min()) else None
            col_profile['max'] = round(df[column].max(), 2) if not pd.isna(df[column].max()) else None
            col_profile['25%'] = round(df[column].quantile(0.25), 2) if not pd.isna(df[column].quantile(0.25)) else None
            col_profile['75%'] = round(df[column].quantile(0.75), 2) if not pd.isna(df[column].quantile(0.75)) else None
            
        else:
            # Categorical statistics
            col_profile['unique'] = df[column].nunique()
            col_profile['top'] = df[column].value_counts().index[0] if not df[column].value_counts().empty else None
            col_profile['top_count'] = df[column].value_counts().iloc[0] if not df[column].value_counts().empty else None
            col_profile['top_pct'] = round(df[column].value_counts().iloc[0] / df[column].count() * 100, 2) if not df[column].value_counts().empty else None
        
        profile[column] = col_profile
    
    return profile

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
                            try:
                                fig = px.histogram(st.session_state.data, x=column, marginal="box")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating histogram: {str(e)}")
                                # Fallback to matplotlib
                                fig, ax = plt.subplots()
                                ax.hist(st.session_state.data[column].dropna())
                                st.pyplot(fig)
                        elif pd.api.types.is_categorical_dtype(st.session_state.data[column]) or st.session_state.data[column].nunique() < 10:
                            try:
                                value_counts = st.session_state.data[column].value_counts().reset_index()
                                value_counts.columns = ['value', 'count']
                                fig = px.bar(value_counts, x='value', y='count', title=f"Count of {column}")
                                fig.update_layout(xaxis_title=column, yaxis_title='Count')
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating bar chart: {str(e)}")
                                # Fallback to matplotlib
                                fig, ax = plt.subplots()
                                st.session_state.data[column].value_counts().plot(kind='bar', ax=ax)
                                st.pyplot(fig)
        
        # Outlier Detection
        st.subheader("Outlier Detection")
        numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            col_for_outliers = st.selectbox("Select column for outlier detection:", numeric_columns)
            try:
                outliers, threshold = detect_outliers(st.session_state.data, col_for_outliers)
                
                if outliers.any():
                    st.info(f"Found {outliers.sum()} outliers in {col_for_outliers} (threshold: {threshold:.2f})")
                    
                    # Plot with outliers highlighted
                    try:
                        fig = px.box(st.session_state.data, y=col_for_outliers)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating box plot: {str(e)}")
                        # Fallback to matplotlib
                        fig, ax = plt.subplots()
                        sns.boxplot(y=st.session_state.data[col_for_outliers], ax=ax)
                        st.pyplot(fig)
                    
                    # Display outliers
                    st.subheader("Outlier Values")
                    st.dataframe(st.session_state.data[outliers][[col_for_outliers]], use_container_width=True)
                else:
                    st.success(f"No outliers detected in {col_for_outliers}")
            except Exception as e:
                st.error(f"Error detecting outliers: {str(e)}")
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
            st.subheader("Correlation Matrix")
            numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            if numeric_columns:
                try:
                    # Calculate correlation matrix
                    corr_matrix = st.session_state.data[numeric_columns].corr()
                    
                    # Create heatmap using go.Heatmap instead of px.imshow
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu_r',
                        zmin=-1,
                        zmax=1
                    ))
                    
                    fig.update_layout(
                        title="Correlation Matrix",
                        height=600,
                        width=800
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display correlation table
                    if st.checkbox("Show correlation values"):
                        st.dataframe(corr_matrix.round(2), use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating correlation matrix: {str(e)}")
                    # Fallback to matplotlib
                    try:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
                        st.pyplot(fig)
                    except Exception as e2:
                        st.error(f"Fallback visualization also failed: {str(e2)}")
            else:
                st.info("No numeric columns available for correlation analysis")
        
        elif viz_type == "Scatter Plot":
            # Code for scatter plot
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols)
                with col2:
                    y_cols = [c for c in numeric_cols if c != x_col]
                    y_col = st.selectbox("Y-axis:", y_cols)
                with col3:
                    color_cols = ["None"] + st.session_state.data.columns.tolist()
                    color_col = st.selectbox("Color by:", color_cols)
                
                try:
                    fig = px.scatter(
                        st.session_state.data, 
                        x=x_col, 
                        y=y_col, 
                        color=None if color_col == "None" else color_col,
                        title=f"{y_col} vs {x_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")
                    # Fallback to matplotlib
                    fig, ax = plt.subplots()
                    ax.scatter(st.session_state.data[x_col], st.session_state.data[y_col])
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)
            else:
                st.info("Need at least 2 numeric columns for scatter plot")
        
        elif viz_type == "Line Chart":
            # Code for line chart
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis (time/sequence):", st.session_state.data.columns.tolist())
                with col2:
                    y_cols = st.multiselect("Y-axis (values):", numeric_cols)
                
                if y_cols:
                    try:
                        fig = px.line(st.session_state.data, x=x_col, y=y_cols, title=f"Line Chart of {', '.join(y_cols)} over {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating line chart: {str(e)}")
                        # Fallback to matplotlib
                        fig, ax = plt.subplots()
                        for col in y_cols:
                            ax.plot(st.session_state.data[x_col], st.session_state.data[col], label=col)
                        ax.set_xlabel(x_col)
                        ax.legend()
                        st.pyplot(fig)
                else:
                    st.info("Please select at least one column for Y-axis")
            else:
                st.info("No numeric columns available for line chart")
        
        elif viz_type == "Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Categories (X-axis):", st.session_state.data.columns.tolist())
            with col2:
                # Dynamically adjust Y-axis options based on column types
                numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                y_options = ["Count"]
                if numeric_cols:
                    y_options.extend(numeric_cols)
                y_col = st.selectbox("Values (Y-axis):", y_options)
            
            try:
                if y_col == "Count":
                    # Create count data
                    count_data = st.session_state.data[x_col].value_counts().reset_index()
                    count_data.columns = ['Category', 'Count']
                    
                    # Sort data for better visualization
                    count_data = count_data.sort_values('Count', ascending=False)
                    
                    # Create simple bar chart using go.Bar
                    fig = go.Figure(data=go.Bar(
                        x=count_data['Category'],
                        y=count_data['Count'],
                        marker_color='skyblue'
                    ))
                    
                    fig.update_layout(
                        title=f"Count of {x_col}",
                        xaxis_title=x_col,
                        yaxis_title="Count",
                        height=500
                    )
                else:
                    # For numeric value aggregation
                    grouped = st.session_state.data.groupby(x_col)[y_col].agg(['mean', 'median', 'count']).reset_index()
                    grouped.columns = [x_col, 'Mean', 'Median', 'Count']
                    
                    # Sort by mean value
                    grouped = grouped.sort_values('Mean', ascending=False)
                    
                    # Create bar chart using go.Bar
                    fig = go.Figure(data=go.Bar(
                        x=grouped[x_col],
                        y=grouped['Mean'],
                        marker_color='skyblue',
                        hovertemplate=
                        '<b>%{x}</b><br>' +
                        'Mean: %{y:.2f}<br>' +
                        'Median: %{customdata[0]:.2f}<br>' +
                        'Count: %{customdata[1]}<br>',
                        customdata=np.column_stack((grouped['Median'], grouped['Count']))
                    ))
                    
                    fig.update_layout(
                        title=f"{y_col} by {x_col} (Mean)",
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        height=500
                    )
                
                # Rotate x-axis labels if many categories
                if len(st.session_state.data[x_col].unique()) > 10:
                    fig.update_layout(xaxis_tickangle=-45)
                
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error creating bar chart: {str(e)}")
                # Fallback to matplotlib
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if y_col == "Count":
                        st.session_state.data[x_col].value_counts().plot(kind='bar', ax=ax)
                    else:
                        st.session_state.data.groupby(x_col)[y_col].mean().sort_values(ascending=False).plot(kind='bar', ax=ax)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e2:
                    st.error(f"Fallback visualization also failed: {str(e2)}")
        
        elif viz_type == "Box Plot":
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    y_col = st.selectbox("Values to plot:", numeric_cols)
                with col2:
                    group_cols = ["None"] + [c for c in st.session_state.data.columns if c != y_col and st.session_state.data[c].nunique() < 20]
                    group_col = st.selectbox("Group by:", group_cols)
                
                try:
                    if group_col == "None":
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=st.session_state.data[y_col],
                            name=y_col,
                            boxmean=True
                        ))
                        fig.update_layout(
                            title=f"Box Plot of {y_col}",
                            yaxis_title=y_col,
                            height=500
                        )
                    else:
                        fig = px.box(
                            st.session_state.data, 
                            y=y_col, 
                            x=group_col, 
                            title=f"Box Plot of {y_col} by {group_col}"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating box plot: {str(e)}")
                    # Fallback to matplotlib
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if group_col == "None":
                        sns.boxplot(y=st.session_state.data[y_col], ax=ax)
                    else:
                        sns.boxplot(x=st.session_state.data[group_col], y=st.session_state.data[y_col], ax=ax)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("No numeric columns available for box plot")
        
        elif viz_type == "Histogram":
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    hist_col = st.selectbox("Select column:", numeric_cols)
                with col2:
                    bins = st.slider("Number of bins:", min_value=5, max_value=100, value=30)
                
                try:
                    # Create histogram using go.Histogram
                    fig = go.Figure(data=go.Histogram(
                        x=st.session_state.data[hist_col],
                        nbinsx=bins,
                        marker_color='steelblue'
                    ))
                    
                    fig.update_layout(
                        title=f"Histogram of {hist_col}",
                        xaxis_title=hist_col,
                        yaxis_title="Count",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating histogram: {str(e)}")
                    # Fallback to matplotlib
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(st.session_state.data[hist_col].dropna(), bins=bins)
                    ax.set_xlabel(hist_col)
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
            else:
                st.info("No numeric columns available for histogram")
        
        elif viz_type == "Pair Plot":
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 1:
                # Limit options to prevent performance issues
                if len(numeric_cols) > 10:
                    numeric_cols = numeric_cols[:10]
                    st.warning("Limiting to first 10 numeric columns for performance")
                
                selected_cols = st.multiselect("Select columns (2-5 recommended):", 
                                               numeric_cols, 
                                               default=numeric_cols[:min(3, len(numeric_cols))])
                
                if len(selected_cols) > 1:
                    if len(selected_cols) > 5:
                        st.warning("Using many columns might slow down the browser")
                    
                    color_cols = ["None"] + [c for c in st.session_state.data.columns if c not in selected_cols and st.session_state.data[c].nunique() < 10]
                    color_col = st.selectbox("Color by:", color_cols)
                    
                    try:
                        # Create pair plot using go.Splom (Scatter Plot Matrix)
                        if color_col == "None":
                            fig = px.scatter_matrix(
                                st.session_state.data,
                                dimensions=selected_cols,
                                title="Pair Plot"
                            )
                        else:
                            fig = px.scatter_matrix(
                                st.session_state.data,
                                dimensions=selected_cols,
                                color=color_col,
                                title="Pair Plot"
                            )
                        
                        # Update to make it more compact and readable
                        fig.update_layout(
                            height=800,
                            width=800
                        )
                        
                        # Less opacity for better visibility of overlapping points
                        fig.update_traces(marker=dict(opacity=0.7))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating pair plot: {str(e)}")
                        # Fallback to matplotlib
                        try:
                            if len(selected_cols) <= 4:  # Seaborn pairplot can be slow with many columns
                                fig = sns.pairplot(
                                    st.session_state.data[selected_cols] if color_col == "None" 
                                    else st.session_state.data[selected_cols + [color_col]], 
                                    hue=None if color_col == "None" else color_col,
                                    diag_kind="kde"
                                )
                                st.pyplot(fig)
                            else:
                                st.error("Too many columns for fallback visualization")
                        except Exception as e2:
                            st.error(f"Fallback visualization also failed: {str(e2)}")
                else:
                    st.info("Please select at least 2 columns for pair plot")
            else:
                st.info("Need at least 2 numeric columns for pair plot")
    
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
                
        elif transform_choice == "Sort Data":
            st.subheader("Sort Data")
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by:", st.session_state.data.columns)
            with col2:
                sort_order = st.radio("Order:", ["Ascending", "Descending"])
            
            ascending = sort_order == "Ascending"
            sorted_data = st.session_state.data.sort_values(by=sort_by, ascending=ascending)
            
            st.dataframe(sorted_data.head(100), use_container_width=True)
            
            if st.button("Apply Sorting"):
                st.session_state.transformed_data = sorted_data
                st.success("Sorting applied successfully!")
                
        elif transform_choice == "Handle Missing Values":
            st.subheader("Handle Missing Values")
            
            # Show columns with missing values
            missing_cols = st.session_state.data.columns[st.session_state.data.isna().any()].tolist()
            if not missing_cols:
                st.success("No missing values in the dataset!")
            else:
                st.write(f"Columns with missing values: {', '.join(missing_cols)}")
                
                col_to_fill = st.selectbox("Select column to handle:", missing_cols)
                fill_method = st.selectbox("Fill method:", 
                                          ["Fill with value", "Fill with mean", "Fill with median", 
                                           "Fill with mode", "Forward fill", "Backward fill", "Drop rows"])
                
                preview_data = st.session_state.data.copy()
                
                if fill_method == "Fill with value":
                    fill_value = st.text_input("Fill value:")
                    if fill_value:
                        preview_data[col_to_fill] = preview_data[col_to_fill].fillna(fill_value)
                elif fill_method == "Fill with mean" and pd.api.types.is_numeric_dtype(preview_data[col_to_fill]):
                    preview_data[col_to_fill] = preview_data[col_to_fill].fillna(preview_data[col_to_fill].mean())
                elif fill_method == "Fill with median" and pd.api.types.is_numeric_dtype(preview_data[col_to_fill]):
                    preview_data[col_to_fill] = preview_data[col_to_fill].fillna(preview_data[col_to_fill].median())
                elif fill_method == "Fill with mode":
                    preview_data[col_to_fill] = preview_data[col_to_fill].fillna(preview_data[col_to_fill].mode()[0])
                elif fill_method == "Forward fill":
                    preview_data[col_to_fill] = preview_data[col_to_fill].ffill()
                elif fill_method == "Backward fill":
                    preview_data[col_to_fill] = preview_data[col_to_fill].bfill()
                elif fill_method == "Drop rows":
                    preview_data = preview_data.dropna(subset=[col_to_fill])
                
                st.write(f"Preview (showing first 100 rows):")
                st.dataframe(preview_data.head(100), use_container_width=True)
                
                if st.button("Apply Missing Value Handling"):
                    st.session_state.transformed_data = preview_data
                    st.success(f"Missing values in {col_to_fill} handled successfully!")
        
        elif transform_choice == "Create New Column":
            st.subheader("Create New Column")
            
            new_col_name = st.text_input("New column name:")
            
            if new_col_name:
                formula_type = st.selectbox("Formula type:", 
                                           ["Simple arithmetic", "Categorical mapping", "Text transformation"])
                
                preview_data = st.session_state.data.copy()
                
                if formula_type == "Simple arithmetic":
                    numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                    if not numeric_cols:
                        st.error("No numeric columns available for arithmetic operations")
                    else:
                        col1 = st.selectbox("First column:", numeric_cols)
                        operation = st.selectbox("Operation:", ["+", "-", "*", "/", "**"])
                        col2_or_value = st.radio("Second operand type:", ["Column", "Value"])
                        
                        if col2_or_value == "Column":
                            col2 = st.selectbox("Second column:", [c for c in numeric_cols if c != col1])
                            if operation == "+":
                                preview_data[new_col_name] = preview_data[col1] + preview_data[col2]
                            elif operation == "-":
                                preview_data[new_col_name] = preview_data[col1] - preview_data[col2]
                            elif operation == "*":
                                preview_data[new_col_name] = preview_data[col1] * preview_data[col2]
                            elif operation == "/":
                                preview_data[new_col_name] = preview_data[col1] / preview_data[col2].replace(0, np.nan)
                            elif operation == "**":
                                preview_data[new_col_name] = preview_data[col1] ** preview_data[col2]
                        else:
                            value = st.number_input("Value:")
                            if operation == "+":
                                preview_data[new_col_name] = preview_data[col1] + value
                            elif operation == "-":
                                preview_data[new_col_name] = preview_data[col1] - value
                            elif operation == "*":
                                preview_data[new_col_name] = preview_data[col1] * value
                            elif operation == "/":
                                preview_data[new_col_name] = preview_data[col1] / value if value != 0 else np.nan
                            elif operation == "**":
                                preview_data[new_col_name] = preview_data[col1] ** value
                
                elif formula_type == "Categorical mapping":
                    cat_col = st.selectbox("Column to map:", st.session_state.data.columns)
                    unique_values = st.session_state.data[cat_col].unique()
                    
                    st.write("Define mapping:")
                    mapping = {}
                    for val in unique_values:
                        mapping[val] = st.text_input(f"Map '{val}' to:", val)
                    
                    preview_data[new_col_name] = preview_data[cat_col].map(mapping)
                
                elif formula_type == "Text transformation":
                    text_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
                    if not text_cols:
                        st.error("No text columns available for text transformation")
                    else:
                        text_col = st.selectbox("Text column:", text_cols)
                        transform_type = st.selectbox("Transformation type:", 
                                                     ["Uppercase", "Lowercase", "Length", "Extract substring"])
                        
                        if transform_type == "Uppercase":
                            preview_data[new_col_name] = preview_data[text_col].str.upper()
                        elif transform_type == "Lowercase":
                            preview_data[new_col_name] = preview_data[text_col].str.lower()
                        elif transform_type == "Length":
                            preview_data[new_col_name] = preview_data[text_col].str.len()
                        elif transform_type == "Extract substring":
                            start_pos = st.number_input("Start position:", min_value=0, value=0)
                            end_pos = st.number_input("End position:", min_value=1, value=1)
                            preview_data[new_col_name] = preview_data[text_col].str[start_pos:end_pos]
                
                st.write("Preview with new column:")
                st.dataframe(preview_data.head(100), use_container_width=True)
                
                if st.button("Create Column"):
                    st.session_state.transformed_data = preview_data
                    st.success(f"Column '{new_col_name}' created successfully!")
        
        elif transform_choice == "Drop Columns":
            st.subheader("Drop Columns")
            
            cols_to_drop = st.multiselect("Select columns to drop:", st.session_state.data.columns)
            
            if cols_to_drop:
                preview_data = st.session_state.data.drop(columns=cols_to_drop)
                
                st.write("Preview after dropping columns:")
                st.dataframe(preview_data.head(100), use_container_width=True)
                
                if st.button("Drop Columns"):
                    st.session_state.transformed_data = preview_data
                    st.success(f"Dropped {len(cols_to_drop)} columns successfully!")
        
        elif transform_choice == "Rename Columns":
            st.subheader("Rename Columns")
            
            st.write("Enter new names for columns:")
            rename_map = {}
            for col in st.session_state.data.columns:
                new_name = st.text_input(f"Rename '{col}' to:", col)
                if new_name != col:
                    rename_map[col] = new_name
            
            if rename_map:
                preview_data = st.session_state.data.rename(columns=rename_map)
                
                st.write("Preview with renamed columns:")
                st.dataframe(preview_data.head(100), use_container_width=True)
                
                if st.button("Rename Columns"):
                    st.session_state.transformed_data = preview_data
                    st.success(f"Renamed {len(rename_map)} columns successfully!")
        
        # Display transformed data if available
        if st.session_state.transformed_data is not None:
            st.subheader("Transformed Data Preview")
            st.dataframe(st.session_state.transformed_data.head(100), use_container_width=True)
            
            if st.button("Reset Transformation"):
                st.session_state.transformed_data = None
                st.success("Transformation reset successfully!")
    
    # Tab 5: Export
    with tabs[4]:
        st.header("Export Data")
        
        # Export options
        export_options = ["CSV", "Excel", "JSON"]
        export_format = st.selectbox("Select export format:", export_options)
        
        # Select data to export
        data_to_export = st.radio("Data to export:", 
                                  ["Original Data", "Transformed Data"],
                                  disabled=(st.session_state.transformed_data is None))
        
        # Export filename
        export_filename = st.text_input("Export filename (without extension):", 
                                       value=st.session_state.file_name or "exported_data")
        
        # Export button
        if st.button("Export Data"):
            export_data_df = st.session_state.transformed_data if data_to_export == "Transformed Data" and st.session_state.transformed_data is not None else st.session_state.data
            
            try:
                if export_format == "CSV":
                    csv_data = export_data_df.to_csv(index=False)
                    b64 = base64.b64encode(csv_data.encode()).decode()
                    href = f'<a href="data:text/csv;base64,{b64}" download="{export_filename}.csv">Download CSV file</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success(f"Data exported as CSV: {export_filename}.csv")
                
                elif export_format == "Excel":
                    buffer = BytesIO()
                    export_data_df.to_excel(buffer, index=False)
                    b64 = base64.b64encode(buffer.getvalue()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{export_filename}.xlsx">Download Excel file</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success(f"Data exported as Excel: {export_filename}.xlsx")
                
                elif export_format == "JSON":
                    json_data = export_data_df.to_json(orient="records")
                    b64 = base64.b64encode(json_data.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="{export_filename}.json">Download JSON file</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success(f"Data exported as JSON: {export_filename}.json")
            
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

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
