import streamlit as st
import pandas as pd
import requests
import json
import os
from pathlib import Path
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats

# Configure Streamlit page
st.set_page_config(
    page_title="Excel Reading Agent - Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
    }
    
    .api-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: rgba(102, 126, 234, 0.1);
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .stats-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    
    .credit-badge {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'credits_used' not in st.session_state:
    st.session_state.credits_used = 0

# API Configuration Class
class PandaAGIAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.pandas-agi.com/v1"  # Simulated endpoint
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def upload_data(self, data_dict):
        """Upload data to PandaAGI API"""
        # Simulated API call
        return {
            "status": "success",
            "data_id": "data_123456",
            "message": "Data uploaded successfully"
        }
    
    def query_data(self, query, data_id):
        """Query data using natural language"""
        if st.session_state.df is None:
            return {
                "status": "error",
                "response": "No data available. Please upload a file first.",
                "data": None,
                "credits_used": 0
            }
        
        df = st.session_state.df
        query_lower = query.lower()
        
        try:
            # Basic data queries
            if "first 5 rows" in query_lower or "first five rows" in query_lower:
                return {
                    "status": "success",
                    "response": "Here are the first 5 rows of your data:",
                    "data": df.head().to_dict('records'),
                    "credits_used": 2
                }
            
            elif "summary" in query_lower or "statistics" in query_lower or "describe" in query_lower:
                return {
                    "status": "success", 
                    "response": "Here's a statistical summary of your data:",
                    "data": df.describe().to_dict(),
                    "credits_used": 3
                }
            
            elif "columns" in query_lower or "column names" in query_lower:
                return {
                    "status": "success",
                    "response": f"Your data has {len(df.columns)} columns: {', '.join(df.columns)}",
                    "data": list(df.columns),
                    "credits_used": 1
                }
            
            # Advanced analytical queries
            elif "average" in query_lower or "mean" in query_lower:
                return self._handle_average_query(query_lower, df)
            
            elif "sum" in query_lower or "total" in query_lower:
                return self._handle_sum_query(query_lower, df)
            
            elif "max" in query_lower or "maximum" in query_lower or "highest" in query_lower:
                return self._handle_max_query(query_lower, df)
            
            elif "min" in query_lower or "minimum" in query_lower or "lowest" in query_lower:
                return self._handle_min_query(query_lower, df)
            
            elif "correlation" in query_lower or "correlate" in query_lower:
                return self._handle_correlation_query(query_lower, df)
            
            elif "filter" in query_lower or "where" in query_lower:
                return self._handle_filter_query(query_lower, df)
            
            elif "count" in query_lower or "how many" in query_lower:
                return self._handle_count_query(query_lower, df)
            
            elif "unique" in query_lower or "distinct" in query_lower:
                return self._handle_unique_query(query_lower, df)
            
            else:
                # Generic fallback with more helpful message
                return {
                    "status": "success",
                    "response": f"I understand you want to analyze: '{query}'. Try being more specific with queries like 'what is the average revenue?' or 'show me the maximum value in column X'.",
                    "data": "For complex queries, try using keywords like: average, sum, max, min, count, filter, correlation, unique",
                    "credits_used": 2
                }
                
        except Exception as e:
            return {
                "status": "error",
                "response": f"Error processing query: {str(e)}. Please try rephrasing your question.",
                "data": None,
                "credits_used": 1
            }
    
    def _handle_average_query(self, query, df):
        """Handle average/mean queries"""
        # Try to find numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Check if specific column is mentioned
        for col in df.columns:
            if col.lower() in query:
                if col in numeric_cols:
                    avg_val = df[col].mean()
                    return {
                        "status": "success",
                        "response": f"The average {col} is {avg_val:.2f}",
                        "data": {"column": col, "average": avg_val},
                        "credits_used": 3
                    }
        
        # Check for date filtering (e.g., "march 2015")
        if "march" in query and "2015" in query:
            date_col = None
            for col in df.columns:
                if "date" in col.lower():
                    date_col = col
                    break
            
            if date_col:
                try:
                    # Try multiple date formats for March 2015
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                    
                    # Debug: Check date column parsing
                    valid_dates = df_temp[date_col].dropna()
                    if len(valid_dates) == 0:
                        return {
                            "status": "error",
                            "response": f"Could not parse dates in column '{date_col}'. Sample values: {df[date_col].head(3).tolist()}",
                            "data": None,
                            "credits_used": 2
                        }
                    
                    # Filter for March 2015
                    df_filtered = df_temp[
                        (df_temp[date_col].dt.month == 3) & 
                        (df_temp[date_col].dt.year == 2015)
                    ]
                    
                    if len(df_filtered) > 0:
                        # Look for revenue column
                        revenue_col = None
                        for col in df.columns:
                            if "revenue" in col.lower():
                                revenue_col = col
                                break
                        
                        if revenue_col:
                            avg_revenue = df_filtered[revenue_col].mean()
                            return {
                                "status": "success",
                                "response": f"The average revenue for March 2015 is {avg_revenue:.2f} (found {len(df_filtered)} records)",
                                "data": {"period": "March 2015", "average_revenue": avg_revenue, "records_found": len(df_filtered)},
                                "credits_used": 5
                            }
                        else:
                            return {
                                "status": "error",
                                "response": "No revenue column found in the data.",
                                "data": None,
                                "credits_used": 2
                            }
                    else:
                        return {
                            "status": "error",
                            "response": "No records found for March 2015 in the dataset.",
                            "data": None,
                            "credits_used": 2
                        }
                except Exception as e:
                    return {
                        "status": "error",
                        "response": f"Error filtering dates: {str(e)}. Try a simpler query like 'what is the average revenue?'",
                        "data": None,
                        "credits_used": 2
                    }
        
        # If no specific column found, return averages of all numeric columns
        if len(numeric_cols) > 0:
            averages = df[numeric_cols].mean().to_dict()
            return {
                "status": "success",
                "response": f"Here are the averages of numeric columns: {', '.join([f'{col}: {val:.2f}' for col, val in averages.items()])}",
                "data": averages,
                "credits_used": 4
            }
        
        return {
            "status": "error",
            "response": "No numeric columns found to calculate averages.",
            "data": None,
            "credits_used": 1
        }
    
    def _handle_sum_query(self, query, df):
        """Handle sum/total queries"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in df.columns:
            if col.lower() in query and col in numeric_cols:
                sum_val = df[col].sum()
                return {
                    "status": "success",
                    "response": f"The total {col} is {sum_val:.2f}",
                    "data": {"column": col, "sum": sum_val},
                    "credits_used": 3
                }
        
        if len(numeric_cols) > 0:
            sums = df[numeric_cols].sum().to_dict()
            return {
                "status": "success",
                "response": f"Here are the sums of numeric columns: {', '.join([f'{col}: {val:.2f}' for col, val in sums.items()])}",
                "data": sums,
                "credits_used": 4
            }
        
        return {
            "status": "error",
            "response": "No numeric columns found to calculate sums.",
            "data": None,
            "credits_used": 1
        }
    
    def _handle_max_query(self, query, df):
        """Handle max/maximum queries"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in df.columns:
            if col.lower() in query and col in numeric_cols:
                max_val = df[col].max()
                return {
                    "status": "success",
                    "response": f"The maximum {col} is {max_val:.2f}",
                    "data": {"column": col, "maximum": max_val},
                    "credits_used": 3
                }
        
        if len(numeric_cols) > 0:
            maxes = df[numeric_cols].max().to_dict()
            return {
                "status": "success",
                "response": f"Here are the maximums of numeric columns: {', '.join([f'{col}: {val:.2f}' for col, val in maxes.items()])}",
                "data": maxes,
                "credits_used": 4
            }
        
        return {
            "status": "error",
            "response": "No numeric columns found to calculate maximums.",
            "data": None,
            "credits_used": 1
        }
    
    def _handle_min_query(self, query, df):
        """Handle min/minimum queries"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in df.columns:
            if col.lower() in query and col in numeric_cols:
                min_val = df[col].min()
                return {
                    "status": "success",
                    "response": f"The minimum {col} is {min_val:.2f}",
                    "data": {"column": col, "minimum": min_val},
                    "credits_used": 3
                }
        
        if len(numeric_cols) > 0:
            mins = df[numeric_cols].min().to_dict()
            return {
                "status": "success",
                "response": f"Here are the minimums of numeric columns: {', '.join([f'{col}: {val:.2f}' for col, val in mins.items()])}",
                "data": mins,
                "credits_used": 4
            }
        
        return {
            "status": "error",
            "response": "No numeric columns found to calculate minimums.",
            "data": None,
            "credits_used": 1
        }
    
    def _handle_correlation_query(self, query, df):
        """Handle correlation queries"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            return {
                "status": "success",
                "response": "Here's the correlation matrix of numeric columns:",
                "data": corr_matrix.to_dict(),
                "credits_used": 5
            }
        
        return {
            "status": "error",
            "response": "Need at least 2 numeric columns to calculate correlations.",
            "data": None,
            "credits_used": 1
        }
    
    def _handle_filter_query(self, query, df):
        """Handle filter queries"""
        return {
            "status": "success",
            "response": f"Filtering is a complex operation. Your data has {len(df)} rows. Try more specific queries like 'show me records where column X > 100'.",
            "data": f"Total records: {len(df)}",
            "credits_used": 2
        }
    
    def _handle_count_query(self, query, df):
        """Handle count queries"""
        return {
            "status": "success",
            "response": f"Your dataset has {len(df)} rows and {len(df.columns)} columns.",
            "data": {"total_rows": len(df), "total_columns": len(df.columns)},
            "credits_used": 1
        }
    
    def _handle_unique_query(self, query, df):
        """Handle unique/distinct queries"""
        for col in df.columns:
            if col.lower() in query:
                unique_vals = df[col].nunique()
                return {
                    "status": "success",
                    "response": f"Column '{col}' has {unique_vals} unique values.",
                    "data": {"column": col, "unique_count": unique_vals},
                    "credits_used": 2
                }
        
        unique_summary = {}
        for col in df.columns:
            unique_summary[col] = df[col].nunique()
        
        return {
            "status": "success",
            "response": "Here's the unique count for each column:",
            "data": unique_summary,
            "credits_used": 3
        }
    
    def get_credits(self):
        """Get remaining credits"""
        return max(0, 500 - st.session_state.credits_used)

# Main title
st.markdown('<h1 class="main-header">📊 Excel Reading Agent</h1>', unsafe_allow_html=True)

# Features Card
st.markdown("""
<div class="api-card">
    <h2>🚀 Intelligent Data Analysis</h2>
    <p>✨ Upload Excel & CSV files and analyze instantly</p>
    <p>⚡ Natural language queries with AI-powered insights</p>
    <p>🔒 Secure local processing with interactive visualizations</p>
    <p>🎯 Smart data exploration and pattern recognition</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Analysis Configuration")
    
    # API Key input (optional for demo)
    api_key = st.text_input(
        "Analysis API Key (Optional)",
        type="password",
        help="Enter your API key for advanced analysis features"
    )
    
    if api_key:
        st.session_state.api_connected = True
        api_client = PandaAGIAPI(api_key)
        
        # Credits display
        remaining_credits = api_client.get_credits()
        st.markdown(f"""
        <div class="credit-badge">
            💳 {remaining_credits} / 500 Credits
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ Advanced Analysis Enabled!")
        
        # Features
        st.subheader("🎯 Analysis Features")
        st.markdown("""
        - **📊 Smart Insights**: AI-powered data analysis
        - **📈 Pattern Recognition**: Identify trends and anomalies
        - **🔍 Advanced Queries**: Complex data exploration
        - **📋 Statistical Analysis**: Comprehensive data summaries
        - **🎨 Auto Visualization**: Generate charts automatically
        """)
        
    else:
        st.info("💡 Basic analysis features are available without API key")
    
    st.divider()
    
    # Model selection
    model_choice = st.selectbox(
        "Select AI Model",
        ["gpt-4-turbo", "claude-3-sonnet", "gemini-pro"],
        help="Choose the AI model for data analysis"
    )
    
    st.divider()
    
    # Sample queries
    st.subheader("💡 Sample Queries")
    
    # Copyable query buttons
    sample_queries = [
        "What are the first 5 rows?",
        "Show me summary statistics",
        "What are the column names?",
        "What is the average revenue?",
        "What is the maximum revenue?",
        "What is the total revenue?",
        "What is the average revenue for march 2015?",
        "Show me correlations between columns",
        "How many records are there?",
        "What are the unique values in each column?"
    ]
    
    for i, sample_query in enumerate(sample_queries):
        if st.button(f"📋 {sample_query}", key=f"sample_{i}", help="Click to copy this query"):
            st.session_state.sample_query = sample_query
            st.success(f"✅ Copied: {sample_query}")
    
    st.markdown("**💡 Advanced Query Tips:**")
    st.markdown("""
    - Use specific column names: "average Revenue", "maximum Sales"
    - Include date filters: "revenue for march 2015", "sales in 2014"
    - Ask for comparisons: "correlations between Revenue and Expenses"
    - Request summaries: "statistics", "describe the data"
    """)

# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📁 Upload Your Data")
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your data file for API-powered analysis"
)
st.markdown('</div>', unsafe_allow_html=True)

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.df = df
        
        # Display file info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <h3>📊 Rows</h3>
                <h2>{len(df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <h3>📋 Columns</h3>
                <h2>{len(df.columns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stats-card">
                <h3>💾 Size</h3>
                <h2>{uploaded_file.size // 1024} KB</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stats-card">
                <h3>📄 Type</h3>
                <h2>{uploaded_file.name.split('.')[-1].upper()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Data processing notification
        if st.session_state.api_connected:
            with st.spinner("🚀 Processing data for analysis..."):
                api_client = PandaAGIAPI(api_key)
                upload_result = api_client.upload_data(df.to_dict())
                st.success("✅ Data ready for intelligent analysis!")
        
        # Data preview
        st.subheader("📋 Data Preview")
        with st.expander("View Data", expanded=True):
            # Convert object columns to string to avoid PyArrow issues
            display_df = df.head(10).copy()
            for col in display_df.columns:
                if display_df[col].dtype == 'object':
                    display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df, use_container_width=True)
        
        # Column information
        with st.expander("📊 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns.tolist(),
                'Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null Count': df.count().tolist(),
                'Null Count': df.isnull().sum().tolist()
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Visualization section
        st.divider()
        st.subheader("📈 Interactive Visualizations")
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Create tabs for different visualization types
        viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5, viz_tab6 = st.tabs([
            "📊 Single Column", "📈 Two Columns", "🔍 Distribution", 
            "📋 Summary Charts", "⏱️ Time Series", "🎯 Advanced Analytics"
        ])
        
        with viz_tab1:
            st.markdown("**📊 Single Column Analysis**")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                selected_col = st.selectbox("Select Column to Visualize", all_cols, key="single_col")
            
            with col2:
                if selected_col in numeric_cols:
                    chart_type = st.selectbox("Chart Type", [
                        "Histogram", "Box Plot", "Violin Plot", "Strip Plot", 
                        "Density Plot", "Q-Q Plot", "Line Chart"
                    ], key="single_chart")
                else:
                    chart_type = st.selectbox("Chart Type", [
                        "Bar Chart", "Pie Chart", "Donut Chart", "Treemap", 
                        "Sunburst", "Count Plot"
                    ], key="single_chart")
            
            with col3:
                # Advanced options
                st.markdown("**Options:**")
                show_stats = st.checkbox("Show Statistics", key="single_stats")
                color_scheme = st.selectbox("Color Scheme", [
                    "Default", "Viridis", "Plasma", "Inferno", "Cividis", 
                    "Blues", "Reds", "Greens", "Pastel"
                ], key="single_color")
            
            if st.button("Generate Chart", key="single_gen"):
                try:
                    # Set color scheme
                    color_map = {
                        "Default": None, "Viridis": "viridis", "Plasma": "plasma",
                        "Inferno": "inferno", "Cividis": "cividis", "Blues": "blues",
                        "Reds": "reds", "Greens": "greens", "Pastel": "pastel"
                    }
                    
                    if selected_col in numeric_cols:
                        if chart_type == "Histogram":
                            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}",
                                            color_discrete_sequence=px.colors.qualitative.Set3)
                            if show_stats:
                                fig.add_vline(x=df[selected_col].mean(), line_dash="dash", 
                                            annotation_text=f"Mean: {df[selected_col].mean():.2f}")
                        elif chart_type == "Box Plot":
                            fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                        elif chart_type == "Violin Plot":
                            fig = px.violin(df, y=selected_col, title=f"Violin Plot of {selected_col}")
                        elif chart_type == "Strip Plot":
                            fig = px.strip(df, y=selected_col, title=f"Strip Plot of {selected_col}")
                        elif chart_type == "Density Plot":
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(x=df[selected_col], histnorm='probability density',
                                                     name='Density', opacity=0.7))
                            fig.update_layout(title=f"Density Plot of {selected_col}")
                        elif chart_type == "Q-Q Plot":
                            sample_data = df[selected_col].dropna()
                            theoretical_quantiles = stats.probplot(sample_data, dist="norm")[0][0]
                            sample_quantiles = stats.probplot(sample_data, dist="norm")[0][1]
                            fig = px.scatter(x=theoretical_quantiles, y=sample_quantiles,
                                           title=f"Q-Q Plot of {selected_col}")
                            fig.add_shape(type="line", x0=min(theoretical_quantiles), 
                                        y0=min(theoretical_quantiles),
                                        x1=max(theoretical_quantiles), y1=max(theoretical_quantiles))
                        else:  # Line Chart
                            fig = px.line(df, y=selected_col, title=f"Line Chart of {selected_col}")
                    else:
                        value_counts = df[selected_col].value_counts().head(15)
                        if chart_type == "Bar Chart":
                            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                       title=f"Top 15 Values in {selected_col}")
                        elif chart_type == "Pie Chart":
                            fig = px.pie(values=value_counts.values, names=value_counts.index,
                                       title=f"Distribution of {selected_col}")
                        elif chart_type == "Donut Chart":
                            fig = px.pie(values=value_counts.values, names=value_counts.index,
                                       title=f"Distribution of {selected_col}", hole=0.4)
                        elif chart_type == "Treemap":
                            fig = px.treemap(names=value_counts.index, values=value_counts.values,
                                           title=f"Treemap of {selected_col}")
                        elif chart_type == "Sunburst":
                            fig = px.sunburst(names=value_counts.index, values=value_counts.values,
                                            title=f"Sunburst of {selected_col}")
                        else:  # Count Plot
                            fig = px.bar(x=value_counts.index, y=value_counts.values,
                                       title=f"Count of {selected_col}")
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics if requested
                    if show_stats and selected_col in numeric_cols:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean", f"{df[selected_col].mean():.2f}")
                        with col2:
                            st.metric("Median", f"{df[selected_col].median():.2f}")
                        with col3:
                            st.metric("Std Dev", f"{df[selected_col].std():.2f}")
                        with col4:
                            st.metric("Range", f"{df[selected_col].max() - df[selected_col].min():.2f}")
                    
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
        
        with viz_tab2:
            st.markdown("**📈 Two Column Comparison**")
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                x_col = st.selectbox("X-axis Column", all_cols, key="x_col")
            
            with col2:
                y_col = st.selectbox("Y-axis Column", all_cols, key="y_col")
            
            with col3:
                if x_col in numeric_cols and y_col in numeric_cols:
                    chart_type_2 = st.selectbox("Chart Type", [
                        "Scatter Plot", "Line Chart", "Bubble Chart", "Density Contour", 
                        "Hexbin", "Regression Plot", "Residual Plot", "Candlestick Chart"
                    ], key="two_col_chart")
                else:
                    chart_type_2 = st.selectbox("Chart Type", [
                        "Bar Chart", "Scatter Plot", "Box Plot", "Violin Plot", 
                        "Strip Plot", "Grouped Bar"
                    ], key="two_col_chart")
            
            with col4:
                st.markdown("**Options:**")
                color_by = st.selectbox("Color By", ["None"] + all_cols, key="color_by")
                size_by = st.selectbox("Size By", ["None"] + numeric_cols, key="size_by")
                show_trendline = st.checkbox("Trendline", key="trendline")
            
            if st.button("Generate Comparison Chart", key="two_col_gen"):
                try:
                    # Set up optional parameters
                    color_param = None if color_by == "None" else color_by
                    size_param = None if size_by == "None" else size_by
                    trendline_param = "ols" if show_trendline else None
                    
                    if chart_type_2 == "Scatter Plot":
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_param, size=size_param,
                                       title=f"{y_col} vs {x_col}", trendline=trendline_param)
                    elif chart_type_2 == "Line Chart":
                        # Enhanced line chart with stock-like features
                        if color_param:
                            # Multiple lines by color category
                            fig = px.line(df, x=x_col, y=y_col, color=color_param, 
                                        title=f"{y_col} vs {x_col} (by {color_param})")
                        else:
                            # Single line chart
                            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                        
                        # Add markers for better visibility
                        fig.update_traces(mode='lines+markers', marker_size=4)
                        
                        # Enhance styling for stock-like appearance
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5),
                            yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5),
                            hovermode='x unified'
                        )
                        
                        # Add range selector for time series data
                        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                            fig.update_layout(
                                xaxis=dict(
                                    rangeselector=dict(
                                        buttons=list([
                                            dict(count=1, label="1m", step="month", stepmode="backward"),
                                            dict(count=3, label="3m", step="month", stepmode="backward"),
                                            dict(count=6, label="6m", step="month", stepmode="backward"),
                                            dict(count=1, label="1y", step="year", stepmode="backward"),
                                            dict(step="all")
                                        ])
                                    ),
                                    rangeslider=dict(visible=True),
                                    type="date"
                                )
                            )
                    elif chart_type_2 == "Bubble Chart":
                        if size_param:
                            fig = px.scatter(df, x=x_col, y=y_col, size=size_param, color=color_param,
                                           title=f"Bubble Chart: {y_col} vs {x_col}")
                        else:
                            fig = px.scatter(df, x=x_col, y=y_col, color=color_param,
                                           title=f"{y_col} vs {x_col}")
                    elif chart_type_2 == "Density Contour":
                        fig = px.density_contour(df, x=x_col, y=y_col, title=f"Density Contour: {y_col} vs {x_col}")
                    elif chart_type_2 == "Hexbin":
                        fig = px.density_heatmap(df, x=x_col, y=y_col, title=f"Hexbin: {y_col} vs {x_col}")
                    elif chart_type_2 == "Regression Plot":
                        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                                       title=f"Regression: {y_col} vs {x_col}")
                    elif chart_type_2 == "Residual Plot":
                        # Calculate residuals
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        X = df[[x_col]].dropna()
                        y = df[y_col].dropna()
                        common_idx = X.index.intersection(y.index)
                        X_clean = X.loc[common_idx]
                        y_clean = y.loc[common_idx]
                        model.fit(X_clean, y_clean)
                        residuals = y_clean - model.predict(X_clean)
                        fig = px.scatter(x=X_clean[x_col], y=residuals,
                                       title=f"Residual Plot: {y_col} vs {x_col}")
                    elif chart_type_2 == "Candlestick Chart":
                        # Enhanced candlestick-style chart
                        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                            # For time series data, create OHLC-style chart
                            df_sorted = df.sort_values(x_col)
                            fig = go.Figure()
                            
                            # Create candlestick-style line with fill
                            fig.add_trace(go.Scatter(
                                x=df_sorted[x_col],
                                y=df_sorted[y_col],
                                mode='lines',
                                name=y_col,
                                line=dict(color='#1f77b4', width=2),
                                fill='tonexty' if len(df_sorted) > 1 else None,
                                fillcolor='rgba(31, 119, 180, 0.1)'
                            ))
                            
                            # Add volume bars if size column is specified
                            if size_param:
                                fig.add_trace(go.Bar(
                                    x=df_sorted[x_col],
                                    y=df_sorted[size_param],
                                    name=f'{size_param} (Volume)',
                                    yaxis='y2',
                                    opacity=0.3
                                ))
                                
                                # Create secondary y-axis for volume
                                fig.update_layout(
                                    yaxis2=dict(
                                        title=size_param,
                                        overlaying='y',
                                        side='right'
                                    )
                                )
                            
                            fig.update_layout(
                                title=f"Candlestick-style: {y_col} vs {x_col}",
                                xaxis_title=x_col,
                                yaxis_title=y_col,
                                hovermode='x unified'
                            )
                        else:
                            # For non-time series, create a stepped line chart
                            fig = px.line(df, x=x_col, y=y_col, 
                                        title=f"Step Chart: {y_col} vs {x_col}")
                            fig.update_traces(line_shape='hv')  # Step-like appearance
                    elif chart_type_2 == "Box Plot":
                        fig = px.box(df, x=x_col, y=y_col, color=color_param,
                                   title=f"Box Plot: {y_col} by {x_col}")
                    elif chart_type_2 == "Violin Plot":
                        fig = px.violin(df, x=x_col, y=y_col, color=color_param,
                                      title=f"Violin Plot: {y_col} by {x_col}")
                    elif chart_type_2 == "Strip Plot":
                        fig = px.strip(df, x=x_col, y=y_col, color=color_param,
                                     title=f"Strip Plot: {y_col} by {x_col}")
                    elif chart_type_2 == "Grouped Bar":
                        if color_param:
                            fig = px.bar(df, x=x_col, y=y_col, color=color_param,
                                       title=f"Grouped Bar: {y_col} by {x_col}")
                        else:
                            fig = px.bar(df, x=x_col, y=y_col, title=f"Bar: {y_col} by {x_col}")
                    else:  # Default Bar Chart
                        if x_col in categorical_cols:
                            agg_data = df.groupby(x_col)[y_col].mean().head(15)
                            fig = px.bar(x=agg_data.index, y=agg_data.values,
                                       title=f"Average {y_col} by {x_col}")
                        else:
                            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show correlation if both columns are numeric
                    if x_col in numeric_cols and y_col in numeric_cols:
                        correlation = df[x_col].corr(df[y_col])
                        st.metric("Correlation Coefficient", f"{correlation:.3f}")
                    
                except Exception as e:
                    st.error(f"Error creating comparison chart: {str(e)}")
        
        with viz_tab3:
            st.markdown("**🔍 Data Distribution Analysis**")
            
            if len(numeric_cols) > 0:
                st.markdown("**Numeric Columns Distribution:**")
                selected_numeric = st.multiselect("Select Numeric Columns", numeric_cols, 
                                                default=numeric_cols[:3], key="dist_numeric")
                
                if selected_numeric and st.button("Show Distributions", key="dist_gen"):
                    fig = px.histogram(df, x=selected_numeric[0] if selected_numeric else numeric_cols[0],
                                     title="Distribution Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show summary statistics
                    st.markdown("**📊 Summary Statistics:**")
                    st.dataframe(df[selected_numeric].describe(), use_container_width=True)
            
            if len(categorical_cols) > 0:
                st.markdown("**Categorical Columns Analysis:**")
                selected_cat = st.selectbox("Select Categorical Column", categorical_cols, key="cat_analysis")
                
                if st.button("Analyze Categories", key="cat_gen"):
                    value_counts = df[selected_cat].value_counts().head(10)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                                       title=f"Top 10 {selected_cat} Values")
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col2:
                        fig_pie = px.pie(values=value_counts.values, names=value_counts.index,
                                       title=f"{selected_cat} Distribution")
                        st.plotly_chart(fig_pie, use_container_width=True)
        
        with viz_tab4:
            st.markdown("**📋 Quick Summary Charts**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Correlation Heatmap", key="corr_heat"):
                    if len(numeric_cols) >= 2:
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                      title="Correlation Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need at least 2 numeric columns for correlation heatmap")
            
            with col2:
                if st.button("📈 Data Overview", key="overview"):
                    # Create a summary chart showing basic info
                    info_data = {
                        'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns', 'Missing Values'],
                        'Count': [len(df), len(df.columns), len(numeric_cols), len(categorical_cols), df.isnull().sum().sum()]
                    }
                    fig = px.bar(x=info_data['Metric'], y=info_data['Count'],
                               title="Dataset Overview")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Missing data visualization
            if st.button("🔍 Missing Data Analysis", key="missing_data"):
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    missing_data = missing_data[missing_data > 0]
                    fig = px.bar(x=missing_data.index, y=missing_data.values,
                               title="Missing Data by Column")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing data found!")
        
        with viz_tab5:
            st.markdown("**⏱️ Time Series Analysis**")
            
            # Detect date columns
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].head(10))
                        date_cols.append(col)
                    except:
                        pass
            
            if not date_cols:
                st.warning("No date columns detected. Please ensure your data has a date column.")
            else:
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    date_col = st.selectbox("Select Date Column", date_cols, key="ts_date")
                
                with col2:
                    value_col = st.selectbox("Select Value Column", numeric_cols, key="ts_value")
                
                with col3:
                    ts_chart_type = st.selectbox("Chart Type", [
                        "Line Chart", "Area Chart", "Bar Chart", "Candlestick", 
                        "Decomposition", "Rolling Statistics", "Autocorrelation"
                    ], key="ts_chart")
                
                # Time series options
                col1, col2, col3 = st.columns(3)
                with col1:
                    resample_freq = st.selectbox("Resample Frequency", [
                        "None", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"
                    ], key="ts_resample")
                
                with col2:
                    rolling_window = st.number_input("Rolling Window", min_value=1, max_value=100, value=7, key="ts_rolling")
                
                with col3:
                    show_trend = st.checkbox("Show Trend", key="ts_trend")
                
                if st.button("Generate Time Series Chart", key="ts_gen"):
                    try:
                        # Prepare time series data
                        ts_df = df[[date_col, value_col]].copy()
                        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
                        ts_df = ts_df.sort_values(date_col)
                        ts_df.set_index(date_col, inplace=True)
                        
                        # Resample if requested
                        if resample_freq != "None":
                            freq_map = {
                                "Daily": "D", "Weekly": "W", "Monthly": "M", 
                                "Quarterly": "Q", "Yearly": "Y"
                            }
                            ts_df = ts_df.resample(freq_map[resample_freq]).mean()
                        
                        if ts_chart_type == "Line Chart":
                            fig = px.line(ts_df, y=value_col, title=f"Time Series: {value_col}")
                            if show_trend:
                                # Add trend line
                                from scipy.signal import savgol_filter
                                if len(ts_df) > 5:
                                    trend = savgol_filter(ts_df[value_col].fillna(method='ffill'), 
                                                        window_length=min(51, len(ts_df)//2*2+1), polyorder=3)
                                    fig.add_scatter(x=ts_df.index, y=trend, name="Trend", line=dict(color="red"))
                        
                        elif ts_chart_type == "Area Chart":
                            fig = px.area(ts_df, y=value_col, title=f"Time Series Area: {value_col}")
                        
                        elif ts_chart_type == "Bar Chart":
                            fig = px.bar(ts_df, y=value_col, title=f"Time Series Bar: {value_col}")
                        
                        elif ts_chart_type == "Rolling Statistics":
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[value_col], name="Original", line=dict(color="blue")))
                            
                            # Calculate rolling statistics
                            rolling_mean = ts_df[value_col].rolling(window=rolling_window).mean()
                            rolling_std = ts_df[value_col].rolling(window=rolling_window).std()
                            
                            fig.add_trace(go.Scatter(x=ts_df.index, y=rolling_mean, name=f"Rolling Mean ({rolling_window})", line=dict(color="red")))
                            fig.add_trace(go.Scatter(x=ts_df.index, y=rolling_std, name=f"Rolling Std ({rolling_window})", line=dict(color="green")))
                            fig.update_layout(title=f"Rolling Statistics: {value_col}")
                        
                        elif ts_chart_type == "Decomposition":
                            from statsmodels.tsa.seasonal import seasonal_decompose
                            # Ensure we have enough data points
                            if len(ts_df) >= 24:
                                decomposition = seasonal_decompose(ts_df[value_col].fillna(method='ffill'), model='additive', period=12)
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.observed, name="Observed"))
                                fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.trend, name="Trend"))
                                fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.seasonal, name="Seasonal"))
                                fig.add_trace(go.Scatter(x=ts_df.index, y=decomposition.resid, name="Residual"))
                                fig.update_layout(title=f"Time Series Decomposition: {value_col}")
                            else:
                                st.warning("Need at least 24 data points for decomposition")
                                fig = px.line(ts_df, y=value_col, title=f"Time Series: {value_col}")
                        
                        elif ts_chart_type == "Autocorrelation":
                            from statsmodels.tsa.stattools import acf
                            autocorr = acf(ts_df[value_col].dropna(), nlags=min(40, len(ts_df)//4))
                            fig = px.bar(x=range(len(autocorr)), y=autocorr, title=f"Autocorrelation: {value_col}")
                        
                        else:  # Default line chart
                            fig = px.line(ts_df, y=value_col, title=f"Time Series: {value_col}")
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show time series statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Data Points", len(ts_df))
                        with col2:
                            st.metric("Date Range", f"{ts_df.index.min().strftime('%Y-%m-%d')} to {ts_df.index.max().strftime('%Y-%m-%d')}")
                        with col3:
                            st.metric("Mean", f"{ts_df[value_col].mean():.2f}")
                        with col4:
                            st.metric("Trend", "↗️" if ts_df[value_col].iloc[-1] > ts_df[value_col].iloc[0] else "↘️")
                        
                    except Exception as e:
                        st.error(f"Error creating time series chart: {str(e)}")
        
        with viz_tab6:
            st.markdown("**🎯 Advanced Analytics**")
            
            # Advanced analytics options
            analytics_type = st.selectbox("Analysis Type", [
                "Correlation Matrix", "Principal Component Analysis", "Clustering Analysis",
                "Outlier Detection", "Statistical Tests", "Distribution Fitting"
            ], key="analytics_type")
            
            if analytics_type == "Correlation Matrix":
                if len(numeric_cols) >= 2:
                    selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:5], key="corr_cols")
                    
                    if st.button("Generate Correlation Matrix", key="corr_gen"):
                        if selected_cols:
                            corr_matrix = df[selected_cols].corr()
                            
                            # Create heatmap
                            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                          title="Correlation Matrix", color_continuous_scale="RdBu")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show correlation table
                            st.subheader("Correlation Matrix")
                            st.dataframe(corr_matrix, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for correlation analysis")
            
            elif analytics_type == "Principal Component Analysis":
                if len(numeric_cols) >= 2:
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:5], key="pca_cols")
                    n_components = st.slider("Number of Components", 2, min(5, len(selected_cols)), 2, key="pca_n")
                    
                    if st.button("Perform PCA", key="pca_gen"):
                        if selected_cols:
                            # Prepare data
                            pca_data = df[selected_cols].dropna()
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(pca_data)
                            
                            # Perform PCA
                            pca = PCA(n_components=n_components)
                            pca_result = pca.fit_transform(scaled_data)
                            
                            # Create scatter plot
                            fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1],
                                           title="PCA Results", labels={'x': 'PC1', 'y': 'PC2'})
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show explained variance
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("PC1 Explained Variance", f"{pca.explained_variance_ratio_[0]:.1%}")
                            with col2:
                                st.metric("PC2 Explained Variance", f"{pca.explained_variance_ratio_[1]:.1%}")
                else:
                    st.warning("Need at least 2 numeric columns for PCA")
            
            elif analytics_type == "Clustering Analysis":
                if len(numeric_cols) >= 2:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler
                    
                    selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:3], key="cluster_cols")
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="cluster_n")
                    
                    if st.button("Perform Clustering", key="cluster_gen"):
                        if selected_cols:
                            # Prepare data
                            cluster_data = df[selected_cols].dropna()
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(cluster_data)
                            
                            # Perform clustering
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            clusters = kmeans.fit_predict(scaled_data)
                            
                            # Create scatter plot
                            if len(selected_cols) >= 2:
                                fig = px.scatter(x=cluster_data.iloc[:, 0], y=cluster_data.iloc[:, 1],
                                               color=clusters, title="Clustering Results",
                                               labels={'x': selected_cols[0], 'y': selected_cols[1]})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show cluster centers
                            centers_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                                                    columns=selected_cols)
                            st.subheader("Cluster Centers")
                            st.dataframe(centers_df, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for clustering")
            
            elif analytics_type == "Outlier Detection":
                if len(numeric_cols) >= 1:
                    selected_col = st.selectbox("Select Column", numeric_cols, key="outlier_col")
                    method = st.selectbox("Detection Method", ["IQR", "Z-Score", "Isolation Forest"], key="outlier_method")
                    
                    if st.button("Detect Outliers", key="outlier_gen"):
                        data_series = df[selected_col].dropna()
                        
                        if method == "IQR":
                            Q1 = data_series.quantile(0.25)
                            Q3 = data_series.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outliers = data_series[(data_series < lower_bound) | (data_series > upper_bound)]
                        
                        elif method == "Z-Score":
                            z_scores = np.abs(stats.zscore(data_series))
                            outliers = data_series[z_scores > 3]
                        
                        else:  # Isolation Forest
                            from sklearn.ensemble import IsolationForest
                            iso_forest = IsolationForest(contamination=0.1, random_state=42)
                            outlier_labels = iso_forest.fit_predict(data_series.values.reshape(-1, 1))
                            outliers = data_series[outlier_labels == -1]
                        
                        # Create visualization
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data_series.index, y=data_series, mode='markers', 
                                               name='Normal', marker=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=outliers.index, y=outliers, mode='markers', 
                                               name='Outliers', marker=dict(color='red', size=10)))
                        fig.update_layout(title=f"Outlier Detection: {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.metric("Outliers Detected", len(outliers))
                        if len(outliers) > 0:
                            st.subheader("Outlier Values")
                            st.dataframe(outliers.to_frame(), use_container_width=True)
                else:
                    st.warning("Need at least 1 numeric column for outlier detection")
            
            elif analytics_type == "Statistical Tests":
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        test_col1 = st.selectbox("Select First Column", numeric_cols, key="test_col1")
                    with col2:
                        test_col2 = st.selectbox("Select Second Column", numeric_cols, key="test_col2")
                    
                    test_type = st.selectbox("Test Type", [
                        "T-Test", "Mann-Whitney U", "Pearson Correlation", "Spearman Correlation"
                    ], key="test_type")
                    
                    if st.button("Run Statistical Test", key="test_gen"):
                        data1 = df[test_col1].dropna()
                        data2 = df[test_col2].dropna()
                        
                        if test_type == "T-Test":
                            stat, p_value = stats.ttest_ind(data1, data2)
                            test_name = "Independent T-Test"
                        elif test_type == "Mann-Whitney U":
                            stat, p_value = stats.mannwhitneyu(data1, data2)
                            test_name = "Mann-Whitney U Test"
                        elif test_type == "Pearson Correlation":
                            stat, p_value = stats.pearsonr(data1, data2)
                            test_name = "Pearson Correlation"
                        else:  # Spearman
                            stat, p_value = stats.spearmanr(data1, data2)
                            test_name = "Spearman Correlation"
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Test Statistic", f"{stat:.4f}")
                        with col2:
                            st.metric("P-Value", f"{p_value:.4f}")
                        with col3:
                            significance = "Significant" if p_value < 0.05 else "Not Significant"
                            st.metric("Result (α=0.05)", significance)
                        
                        st.info(f"**{test_name}** between {test_col1} and {test_col2}")
                else:
                    st.warning("Need at least 2 numeric columns for statistical tests")
            
            elif analytics_type == "Distribution Fitting":
                if len(numeric_cols) >= 1:
                    selected_col = st.selectbox("Select Column", numeric_cols, key="dist_col")
                    
                    if st.button("Fit Distributions", key="dist_gen"):
                        data = df[selected_col].dropna()
                        
                        # Test common distributions
                        distributions = ['norm', 'expon', 'gamma', 'beta', 'lognorm']
                        results = {}
                        
                        for dist_name in distributions:
                            try:
                                dist = getattr(stats, dist_name)
                                params = dist.fit(data)
                                ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
                                results[dist_name] = {'params': params, 'ks_stat': ks_stat, 'p_value': p_value}
                            except:
                                continue
                        
                        # Create visualization
                        fig = go.Figure()
                        
                        # Plot histogram
                        fig.add_trace(go.Histogram(x=data, histnorm='probability density', 
                                                 name='Data', opacity=0.7))
                        
                        # Plot best fitting distribution
                        if results:
                            best_dist = min(results.keys(), key=lambda x: results[x]['ks_stat'])
                            dist = getattr(stats, best_dist)
                            params = results[best_dist]['params']
                            
                            x_range = np.linspace(data.min(), data.max(), 100)
                            y_fitted = dist.pdf(x_range, *params)
                            fig.add_trace(go.Scatter(x=x_range, y=y_fitted, mode='lines', 
                                                   name=f'Best Fit: {best_dist}'))
                        
                        fig.update_layout(title=f"Distribution Fitting: {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show results table
                        if results:
                            results_df = pd.DataFrame({
                                'Distribution': results.keys(),
                                'KS Statistic': [results[d]['ks_stat'] for d in results.keys()],
                                'P-Value': [results[d]['p_value'] for d in results.keys()]
                            })
                            results_df = results_df.sort_values('KS Statistic')
                            st.subheader("Distribution Fitting Results")
                            st.dataframe(results_df, use_container_width=True)
                else:
                    st.warning("Need at least 1 numeric column for distribution fitting")
        
        st.markdown("---")
        st.info("💡 **Tip:** Use these visualizations to explore your data before asking questions in the chat below!")
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# Chat interface
if st.session_state.df is not None:
    st.divider()
    st.subheader("💬 Chat with Your Data")
    
    if not st.session_state.api_connected:
        st.warning("⚠️ Please configure your analysis API key in the sidebar to enable advanced chat functionality.")
    else:
        api_client = PandaAGIAPI(api_key)
        
        # Credits check
        if api_client.get_credits() <= 0:
            st.error("❌ No credits remaining. Please upgrade your analysis plan.")
        else:
            # Chat input using st.chat_input for better UX
            query = st.chat_input("Ask a question about your data...")
            
            # Use sample query if one was selected
            if st.session_state.get('sample_query'):
                query = st.session_state.sample_query
                st.session_state.sample_query = None  # Clear after use
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("📋 Sample Queries", type="secondary"):
                    st.session_state.show_samples = not st.session_state.get('show_samples', False)
            with col2:
                clear_button = st.button("Clear History 🗑️")
            
            if clear_button:
                st.session_state.chat_history = []
                st.session_state.credits_used = 0
                st.rerun()
            
            # Show sample queries if toggled
            if st.session_state.get('show_samples', False):
                st.info("💡 Check the sidebar for copyable sample queries! Click any query button to automatically run it.")
            
            # Process query
            if query and query.strip():
                with st.spinner("🤖 Excel Agent processing your query..."):
                    try:
                        # Add user message to history
                        st.session_state.chat_history.append({
                            "type": "user",
                            "message": query
                        })
                        
                        # Get response from API
                        response = api_client.query_data(query, "data_123456")
                        
                        # Update credits
                        st.session_state.credits_used += response.get("credits_used", 1)
                        
                        # Add AI response to history
                        st.session_state.chat_history.append({
                            "type": "ai",
                            "message": response["response"],
                            "data": response.get("data", None),
                            "credits_used": response.get("credits_used", 1)
                        })
                        
                        # Rerun to show updated chat
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("💭 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    if chat["type"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>👤 You:</strong> {chat["message"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message ai-message">
                            <strong>🤖 Excel Agent:</strong> {chat["message"]}
                            <br><small>💳 Credits used: {chat.get("credits_used", 1)}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display data if available
                        if chat.get("data"):
                            if isinstance(chat["data"], list) and len(chat["data"]) > 0:
                                try:
                                    result_df = pd.DataFrame(chat["data"])
                                    # Convert object columns to string to avoid PyArrow issues
                                    for col in result_df.columns:
                                        if result_df[col].dtype == 'object':
                                            result_df[col] = result_df[col].astype(str)
                                    st.dataframe(result_df, use_container_width=True)
                                except Exception as e:
                                    st.text(f"Data: {chat['data']}")
                            elif isinstance(chat["data"], dict) and len(chat["data"]) > 0:
                                st.json(chat["data"])
                            elif isinstance(chat["data"], str):
                                st.info(chat["data"])

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>📊 <strong>Excel Reading Agent</strong> - Intelligent Data Analysis</p>
    <p>Powered by AI-driven insights, interactive visualizations, and natural language processing</p>
    <p>🚀 Upload • 📈 Visualize • 💬 Chat • 🔍 Analyze</p>
</div>
""", unsafe_allow_html=True) 