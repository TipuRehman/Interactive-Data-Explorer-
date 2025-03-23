# Interactive Data Explorer

🚀 **Interactive Data Explorer** is a powerful and user-friendly web application built with **Streamlit** that allows users to upload, analyze, transform, and export datasets effortlessly. This tool provides interactive visualizations, data profiling, outlier detection, and data transformation capabilities.

## 📌 Features

- **📊 Data Profiling** - Get insights into distributions, statistics, and patterns.
- **📈 Interactive Visualizations** - Generate charts and plots to explore data visually.
- **🔍 Outlier Detection** - Detect and handle outliers in numeric datasets.
- **🔄 Data Transformation** - Perform various data cleaning and transformation operations.
- **💾 Export Options** - Export processed data in CSV, Excel, and JSON formats.

## 🛠️ Installation

Ensure you have **Python 3.7+** installed. Then, clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/Interactive-Data-Explorer.git
cd Interactive-Data-Explorer

# Install required dependencies
pip install -r requirements.txt
```

## 🚀 Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

## 📥 Data Upload

- Supports CSV, Excel, and JSON files.
- Drag and drop feature for quick uploads.
- Displays real-time status messages (e.g., *Successfully loaded* message).
- Shows file name, size, and data summary (number of rows and columns).
- Option to select a sample dataset for quick exploration.

## 📊 Data Overview & Profiling

- Provides a summary of dataset including:
  - **Number of Rows and Columns**
  - **Missing Values Count**
- Displays a preview of the dataset (first few rows) in a table format.

## 📈 Interactive Visualizations

- Enables users to create various charts and plots for data exploration.
- Supports **bar charts, scatter plots, histograms, and pie charts**.
- Customizable options for selecting columns and aggregation methods.

## 🔍 Outlier Detection

- Identifies potential outliers in numeric columns.
- Highlights abnormal values to help with data cleaning.

## 🔄 Data Transformation

- Allows users to **clean, transform, and prepare** data.
- Features include:
  - **Handling Missing Values**
  - **Data Type Conversions**
  - **Filtering and Sorting**

## 💾 Export Options

- Allows exporting processed data in multiple formats:
  - **CSV** - With customizable separator and index options.
  - **Excel** - With sheet name and index selection.
  - **JSON** - With customizable orientation.

## 📤 Deployment

This app can be deployed on **Streamlit Cloud**, **Heroku**, or any cloud platform supporting Python applications.

For Heroku deployment:

```bash
heroku create
heroku buildpacks:add heroku/python
git push heroku main
```

## 📷 Screenshots

!![image](https://github.com/user-attachments/assets/eb573fef-dcfe-47f1-88da-52be64c291f8) ![image](https://github.com/user-attachments/assets/f0aebedd-9f4b-480c-9c7d-b424c66c77bc)



## 📜 License

This project is licensed under the **MIT License**.

---
Author: TIPU REHMAN  app link: https://cr2esc2j9seeeufl9nhaa4.streamlit.app/#outlier-detection
