"""
API Client for Excel Reading Agent with Enhanced Query Processing
"""

import requests
import json
from typing import Dict, Any, Optional, List
import pandas as pd
from config import API_CONFIG
from query_processor import EnhancedQueryProcessor

class ExcelReaderAPI:
    """
    Enhanced API client for Excel data analysis with improved query processing
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = API_CONFIG["base_url"]
        self.timeout = API_CONFIG["timeout"]
        self.max_retries = API_CONFIG["max_retries"]
        self.headers = {
            "Authorization": f"Bearer {api_key}" if api_key else None,
            "Content-Type": "application/json"
        }
        
        # Initialize enhanced query processor
        self.query_processor = EnhancedQueryProcessor()
    
    def upload_data(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload data to API (simulated for demo)
        """
        # Simulated API response - in production this would make actual API call
        return {
            "status": "success",
            "data_id": "data_123456",
            "message": "Data uploaded successfully",
            "records": len(data_dict) if isinstance(data_dict, list) else "N/A"
        }
    
    def query_data(self, query: str, data_id: str, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Process natural language query using enhanced query processor
        """
        if df is None:
            return {
                "status": "error",
                "response": "No data available. Please upload a file first.",
                "data": None,
                "credits_used": 0
            }
        
        try:
            # Use enhanced query processor for better understanding
            result = self.query_processor.process_query(query, df)
            
            # Add some additional metadata
            result["timestamp"] = pd.Timestamp.now().isoformat()
            result["data_id"] = data_id
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "response": f"Error processing query: {str(e)}",
                "data": None,
                "credits_used": 1,
                "timestamp": pd.Timestamp.now().isoformat()
            }
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive data summary
        """
        if df is None or df.empty:
            return {"error": "No data available"}
        
        try:
            summary = {
                "basic_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "dtypes": df.dtypes.value_counts().to_dict()
                },
                "column_info": [],
                "data_quality": {
                    "missing_values": df.isnull().sum().to_dict(),
                    "duplicate_rows": df.duplicated().sum(),
                    "completeness_ratio": (1 - df.isnull().sum() / len(df)).to_dict()
                },
                "numeric_summary": {},
                "categorical_summary": {}
            }
            
            # Column information
            for col in df.columns:
                col_info = {
                    "name": col,
                    "type": str(df[col].dtype),
                    "non_null_count": df[col].count(),
                    "null_count": df[col].isnull().sum(),
                    "unique_count": df[col].nunique()
                }
                
                if df[col].dtype in ['int64', 'float64']:
                    col_info.update({
                        "min": float(df[col].min()) if not df[col].isnull().all() else None,
                        "max": float(df[col].max()) if not df[col].isnull().all() else None,
                        "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                        "median": float(df[col].median()) if not df[col].isnull().all() else None
                    })
                
                summary["column_info"].append(col_info)
            
            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
            # Categorical columns summary
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                summary["categorical_summary"][col] = {
                    "top_values": df[col].value_counts().head(5).to_dict(),
                    "unique_count": df[col].nunique()
                }
            
            return summary
            
        except Exception as e:
            return {"error": f"Error generating summary: {str(e)}"}
    
    def validate_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate and suggest improvements for queries
        """
        if not query or not query.strip():
            return {
                "valid": False,
                "issues": ["Query is empty"],
                "suggestions": ["Try asking about specific data aspects like 'show first 5 rows' or 'what are the column names?'"]
            }
        
        validation_result = {
            "valid": True,
            "issues": [],
            "suggestions": [],
            "query_type": "unknown",
            "confidence": 0.0
        }
        
        query_lower = query.lower().strip()
        
        # Check for common query patterns
        if any(word in query_lower for word in ["average", "mean", "sum", "total", "max", "min"]):
            validation_result["query_type"] = "aggregation"
            validation_result["confidence"] = 0.8
            
            # Check if column is specified
            column_found = False
            for col in df.columns:
                if col.lower() in query_lower:
                    column_found = True
                    break
            
            if not column_found:
                validation_result["suggestions"].append(
                    f"Consider specifying a column name. Available columns: {', '.join(df.columns[:3])}{'...' if len(df.columns) > 3 else ''}"
                )
        
        elif any(word in query_lower for word in ["january", "february", "march", "2015", "2020", "q1", "quarter"]):
            validation_result["query_type"] = "date_filter"
            validation_result["confidence"] = 0.7
            
            # Check if date column exists
            date_cols = [col for col in df.columns if any(date_word in col.lower() for date_word in ["date", "time", "created", "updated"])]
            if not date_cols:
                validation_result["issues"].append("No date column found in the data")
                validation_result["suggestions"].append("Your data might not have date information for filtering")
        
        elif any(word in query_lower for word in ["first", "rows", "head", "top"]):
            validation_result["query_type"] = "basic_info"
            validation_result["confidence"] = 0.9
        
        elif any(word in query_lower for word in ["correlation", "relationship"]):
            validation_result["query_type"] = "analysis"
            validation_result["confidence"] = 0.8
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                validation_result["issues"].append("Need at least 2 numeric columns for correlation analysis")
        
        return validation_result
    
    def get_query_suggestions(self, df: pd.DataFrame) -> List[str]:
        """
        Generate contextual query suggestions based on data
        """
        suggestions = []
        
        if df is None or df.empty:
            return ["Upload a file to get personalized suggestions"]
        
        # Basic suggestions
        suggestions.extend([
            "Show me the first 5 rows",
            "What are the column names?",
            "Give me summary statistics"
        ])
        
        # Column-specific suggestions
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:2]:  # Limit to first 2 numeric columns
            suggestions.extend([
                f"What is the average {col}?",
                f"What is the maximum {col}?",
                f"What is the total {col}?"
            ])
        
        # Date-based suggestions if date column exists
        date_cols = [col for col in df.columns if any(date_word in col.lower() for date_word in ["date", "time", "created"])]
        if date_cols and len(numeric_cols) > 0:
            suggestions.extend([
                f"Show me {numeric_cols[0]} for 2015",
                f"What is the monthly trend for {numeric_cols[0]}?"
            ])
        
        # Analysis suggestions
        if len(numeric_cols) >= 2:
            suggestions.append("Show me correlations between columns")
        
        return suggestions[:8]  # Limit to 8 suggestions

class DataProcessor:
    """
    Helper class for data processing operations
    """
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataframe for display in Streamlit
        """
        df_clean = df.copy()
        
        # Convert object columns to string to avoid PyArrow issues
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str)
        
        return df_clean
    
    @staticmethod
    def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect and categorize column types
        """
        column_types = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                column_types[col] = 'numeric'
            elif df[col].dtype == 'datetime64[ns]':
                column_types[col] = 'datetime'
            elif df[col].dtype == 'bool':
                column_types[col] = 'boolean'
            else:
                # Try to detect if it's a date column
                try:
                    pd.to_datetime(df[col].dropna().head(10), errors='raise')
                    column_types[col] = 'datetime_string'
                except:
                    column_types[col] = 'categorical'
        
        return column_types
    
    @staticmethod
    def format_number(value: float, precision: int = 2) -> str:
        """
        Format numbers for display
        """
        if pd.isna(value):
            return "N/A"
        
        if abs(value) >= 1e6:
            return f"{value/1e6:.{precision}f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.{precision}f}K"
        else:
            return f"{value:.{precision}f}" 