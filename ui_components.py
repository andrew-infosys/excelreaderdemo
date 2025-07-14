"""
Simple and Clean UI Components for Excel Reading Agent
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional

class UIComponents:
    """
    Simple, clean UI components for the Excel Reading Agent
    """
    
    @staticmethod
    def setup_page_config():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Excel Reading Agent",
            page_icon="📊",
            layout="wide"
        )
    
    @staticmethod
    def apply_custom_css():
        """Apply minimal custom CSS styling"""
        st.markdown("""
        <style>
        .main-header { 
            color: #2E86AB; 
            font-size: 2.5rem; 
            margin-bottom: 1rem;
        }
        .chat-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 0.75rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render clean application header"""
        st.markdown('<h1 class="main-header">📊 Excel Reading Agent</h1>', unsafe_allow_html=True)
        st.markdown("**Upload your Excel/CSV file and ask questions about your data in natural language!**")
        st.markdown("---")
    
    @staticmethod
    def render_file_upload():
        """Render simple file upload section"""
        st.subheader("📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose an Excel (.xlsx) or CSV (.csv) file",
            type=['csv', 'xlsx', 'xls'],
            help="Maximum file size: 50MB"
        )
        return uploaded_file
    
    @staticmethod
    def render_data_preview(df: pd.DataFrame):
        """Render clean data preview"""
        if df is not None and not df.empty:
            st.subheader("📋 Data Preview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Show first few rows
            st.dataframe(df.head(10), use_container_width=True)
            
            with st.expander("📊 Column Information"):
                col_info = []
                for col in df.columns:
                    col_info.append({
                        "Column": col,
                        "Type": str(df[col].dtype),
                        "Non-Null": df[col].count(),
                        "Null": df[col].isnull().sum()
                    })
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    
    @staticmethod
    def render_chat_interface(api_client, df: pd.DataFrame = None):
        """Render simplified chat interface"""
        st.subheader("💬 Chat with Your Data")
        
        if df is None or df.empty:
            st.info("👆 Please upload a file first to start asking questions about your data.")
            return None
        
        # Simple query input
        query = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., what is expenses on jan 3rd of 2015?",
            help="Try asking about specific dates, calculations, or data exploration"
        )
        
        # Sample queries
        with st.expander("💡 Sample Questions"):
            sample_queries = [
                "What are the first 5 rows?",
                "What is the average revenue?", 
                "What is expenses on jan 3rd of 2015?",
                "Show me data for March 2015",
                "What is the total profit?",
                "Show me correlations between columns"
            ]
            
            cols = st.columns(2)
            for i, sample in enumerate(sample_queries):
                with cols[i % 2]:
                    if st.button(sample, key=f"sample_{i}"):
                        query = sample
                        st.rerun()
        
        # Process query if provided
        if query:
            with st.spinner("Processing your question..."):
                result = api_client.query_data(query, "user_data", df)
                
                # Display results
                UIComponents.render_query_result(result)
        
        return query
    
    @staticmethod
    def render_query_result(result: Dict[str, Any]):
        """Render query results in a clean format"""
        if result['status'] == 'success':
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write("✅", result['response'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show data if available
            if result.get('data') is not None:
                data = result['data']
                
                # Handle different data types
                if isinstance(data, pd.DataFrame) and not data.empty:
                    st.subheader("📊 Results")
                    st.dataframe(data, use_container_width=True)
                
                elif isinstance(data, list) and len(data) > 0:
                    st.subheader("📊 Results")
                    # Convert list to DataFrame for display
                    try:
                        df_display = pd.DataFrame(data)
                        st.dataframe(df_display, use_container_width=True)
                    except:
                        # Fallback to showing as JSON
                        st.json(data)
                
                elif isinstance(data, dict):
                    # Handle dictionary results (like aggregations)
                    if 'result' in data:
                        st.metric("📈 Result", data['result'])
                        if 'records_found' in data:
                            st.caption(f"Based on {data['records_found']} records")
                    else:
                        st.subheader("📊 Results")
                        st.json(data)
                
                elif isinstance(data, str):
                    st.info(data)
                
                else:
                    st.text(str(data))
        else:
            st.error(f"❌ {result['response']}")
    
    @staticmethod
    def render_simple_visualizations(df: pd.DataFrame):
        """Render basic visualization options"""
        if df is None or df.empty:
            return
        
        st.subheader("📈 Quick Visualizations")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_col = st.selectbox("Select column to visualize:", numeric_cols)
            
            with col2:
                chart_type = st.selectbox("Chart type:", ["Bar Chart", "Line Chart", "Histogram"])
            
            if st.button("Generate Chart"):
                if chart_type == "Bar Chart":
                    st.bar_chart(df[selected_col].value_counts().head(10))
                elif chart_type == "Line Chart":
                    st.line_chart(df[selected_col])
                elif chart_type == "Histogram":
                    st.bar_chart(df[selected_col].value_counts().sort_index())
        else:
            st.info("No numeric columns found for visualization.")
    
    @staticmethod
    def render_error_message(error: str):
        """Render clean error message"""
        st.error(f"⚠️ Error: {error}")
    
    @staticmethod
    def render_success_message(message: str):
        """Render clean success message"""
        st.success(f"✅ {message}") 