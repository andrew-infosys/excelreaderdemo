"""
Improved Query Processor with enhanced NLP and date parsing capabilities
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from config import QUERY_CONFIG

class EnhancedQueryProcessor:
    def __init__(self):
        self.date_formats = QUERY_CONFIG["date_formats"]
        self.column_mappings = QUERY_CONFIG["common_column_mappings"]
        self.aggregation_functions = QUERY_CONFIG["aggregation_functions"]
        
    def process_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process natural language query with enhanced understanding
        """
        if df is None or df.empty:
            return {
                "status": "error",
                "response": "No data available. Please upload a file first.",
                "data": None,
                "credits_used": 0
            }
        
        query_lower = query.lower().strip()
        
        try:
            # Parse the query to extract intent, column, date, and aggregation
            parsed_query = self._parse_query(query_lower, df)
            
            # Route to appropriate handler based on intent
            if parsed_query["intent"] == "basic_info":
                return self._handle_basic_info(parsed_query, df)
            elif parsed_query["intent"] == "aggregation":
                return self._handle_aggregation(parsed_query, df)
            elif parsed_query["intent"] == "filter_date":
                return self._handle_date_filter(parsed_query, df)
            elif parsed_query["intent"] == "correlation":
                return self._handle_correlation(parsed_query, df)
            elif parsed_query["intent"] == "comparison":
                return self._handle_comparison(parsed_query, df)
            else:
                return self._handle_fallback(query, df)
                
        except Exception as e:
            return {
                "status": "error",
                "response": f"Error processing query: {str(e)}. Please try rephrasing your question.",
                "data": None,
                "credits_used": 1
            }
    
    def _parse_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse the query to extract intent, columns, dates, and functions
        """
        parsed = {
            "intent": "unknown",
            "column": None,
            "date_filter": None,
            "aggregation": None,
            "original_query": query
        }
        
        # Detect basic info queries
        if any(phrase in query for phrase in ["first", "rows", "head", "top"]):
            parsed["intent"] = "basic_info"
            parsed["action"] = "head"
        elif any(phrase in query for phrase in ["summary", "statistics", "describe"]):
            parsed["intent"] = "basic_info"
            parsed["action"] = "summary"
        elif any(phrase in query for phrase in ["columns", "column names"]):
            parsed["intent"] = "basic_info"
            parsed["action"] = "columns"
        
        # Detect aggregation queries
        elif any(func in query for func_list in self.aggregation_functions.values() for func in func_list):
            parsed["intent"] = "aggregation"
            parsed["aggregation"] = self._extract_aggregation_function(query)
            parsed["column"] = self._extract_column_reference(query, df)
        
        # Detect date filtering
        elif self._has_date_reference(query):
            parsed["intent"] = "filter_date"
            parsed["date_filter"] = self._extract_date_filter(query)
            parsed["column"] = self._extract_column_reference(query, df)
            parsed["aggregation"] = self._extract_aggregation_function(query)
        
        # Detect correlation queries
        elif any(phrase in query for phrase in ["correlation", "correlate", "relationship"]):
            parsed["intent"] = "correlation"
        
        # Detect comparison queries
        elif any(phrase in query for phrase in ["compare", "comparison", "between", "vs", "versus"]):
            parsed["intent"] = "comparison"
        
        return parsed
    
    def _extract_aggregation_function(self, query: str) -> Optional[str]:
        """Extract the aggregation function from the query"""
        for func_type, keywords in self.aggregation_functions.items():
            if any(keyword in query for keyword in keywords):
                return func_type
        return None
    
    def _extract_column_reference(self, query: str, df: pd.DataFrame) -> Optional[str]:
        """Extract column reference from query"""
        # Direct column name match
        for col in df.columns:
            if col.lower() in query:
                return col
        
        # Semantic column mapping
        for concept, variations in self.column_mappings.items():
            if any(var in query for var in variations):
                # Find best matching column
                for col in df.columns:
                    if any(var in col.lower() for var in variations):
                        return col
        
        return None
    
    def _has_date_reference(self, query: str) -> bool:
        """Check if query contains date references"""
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
            r'\b\d{1,2}(st|nd|rd|th)?\b',
            r'\b\d{4}\b',
            r'\b(q1|q2|q3|q4|quarter)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        ]
        
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in date_patterns)
    
    def _extract_date_filter(self, query: str) -> Dict[str, Any]:
        """Extract date filter information from query"""
        date_filter = {}
        
        # Extract year
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', query)
        if year_match:
            date_filter["year"] = int(year_match.group(1))
        
        # Extract month
        month_patterns = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        for month_name, month_num in month_patterns.items():
            if month_name in query:
                date_filter["month"] = month_num
                break
        
        # Extract day
        day_match = re.search(r'\b(\d{1,2})(st|nd|rd|th)?\b', query)
        if day_match:
            date_filter["day"] = int(day_match.group(1))
        
        # Extract quarter
        quarter_match = re.search(r'\b(q[1-4]|quarter\s*[1-4])\b', query, re.IGNORECASE)
        if quarter_match:
            quarter_text = quarter_match.group(1).lower()
            if 'q1' in quarter_text or 'quarter 1' in quarter_text:
                date_filter["quarter"] = 1
            elif 'q2' in quarter_text or 'quarter 2' in quarter_text:
                date_filter["quarter"] = 2
            elif 'q3' in quarter_text or 'quarter 3' in quarter_text:
                date_filter["quarter"] = 3
            elif 'q4' in quarter_text or 'quarter 4' in quarter_text:
                date_filter["quarter"] = 4
        
        return date_filter
    
    def _handle_basic_info(self, parsed_query: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle basic information queries"""
        action = parsed_query.get("action", "head")
        
        if action == "head":
            return {
                "status": "success",
                "response": "First 5 rows:",
                "data": df.head().to_dict('records'),
                "credits_used": 2
            }
        elif action == "summary":
            return {
                "status": "success",
                "response": "Summary statistics:",
                "data": df.describe().to_dict(),
                "credits_used": 3
            }
        elif action == "columns":
            return {
                "status": "success",
                "response": ', '.join(df.columns),
                "data": list(df.columns),
                "credits_used": 1
            }
        
        return self._handle_fallback(parsed_query["original_query"], df)
    
    def _handle_aggregation(self, parsed_query: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle aggregation queries with optional date filtering"""
        aggregation = parsed_query.get("aggregation")
        column = parsed_query.get("column")
        date_filter = parsed_query.get("date_filter")
        
        # Apply date filter if present
        filtered_df = df
        filter_description = ""
        
        if date_filter:
            filtered_df, filter_description = self._apply_date_filter(df, date_filter)
            if filtered_df.empty:
                return {
                    "status": "error",
                    "response": f"No records found for the specified date criteria: {filter_description}",
                    "data": None,
                    "credits_used": 2
                }
        
        # If no specific column, try to infer from query context
        if not column:
            # Look for numeric columns that might be relevant
            numeric_cols = filtered_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                column = numeric_cols[0]  # Take the first numeric column
        
        if not column:
            return {
                "status": "error",
                "response": "Could not identify which column to analyze. Please specify a column name.",
                "data": None,
                "credits_used": 1
            }
        
        # Perform aggregation
        try:
            if aggregation == "average":
                result = filtered_df[column].mean()
                response = f"{result:.2f}"
            elif aggregation == "sum":
                result = filtered_df[column].sum()
                response = f"{result:.2f}"
            elif aggregation == "maximum":
                result = filtered_df[column].max()
                response = f"{result:.2f}"
            elif aggregation == "minimum":
                result = filtered_df[column].min()
                response = f"{result:.2f}"
            elif aggregation == "count":
                result = filtered_df[column].count()
                response = f"{result}"
            else:
                return self._handle_fallback(parsed_query["original_query"], df)
            
            return {
                "status": "success",
                "response": response,
                "data": {
                    "column": column,
                    "aggregation": aggregation,
                    "result": result,
                    "filter": filter_description,
                    "records_found": len(filtered_df)
                },
                "credits_used": 3 + (2 if date_filter else 0)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "response": f"Error calculating {aggregation} for {column}: {str(e)}",
                "data": None,
                "credits_used": 2
            }
    
    def _handle_date_filter(self, parsed_query: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle queries with date filtering"""
        date_filter = parsed_query.get("date_filter", {})
        column = parsed_query.get("column")
        aggregation = parsed_query.get("aggregation")
        
        # Find date column
        date_column = self._find_date_column(df)
        if not date_column:
            return {
                "status": "error",
                "response": "No date column found in the data. Available columns: " + ", ".join(df.columns),
                "data": None,
                "credits_used": 2
            }
        
        # Apply filter and get aggregation if specified
        filtered_df, filter_description = self._apply_date_filter(df, date_filter, date_column)
        
        if filtered_df.empty:
            return {
                "status": "error",
                "response": f"No records found for {filter_description}",
                "data": None,
                "credits_used": 2
            }
        
        if aggregation and column:
            # Perform aggregation on filtered data
            return self._handle_aggregation({
                "aggregation": aggregation,
                "column": column,
                "date_filter": date_filter,
                "original_query": parsed_query["original_query"]
            }, df)
        else:
            # Return filtered data
            return {
                "status": "success",
                "response": f"Found {len(filtered_df)} records for {filter_description}",
                "data": filtered_df.head(10).to_dict('records'),
                "credits_used": 4
            }
    
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the most likely date column in the dataframe"""
        # Check for explicit date column names
        date_keywords = ['date', 'time', 'timestamp', 'created', 'updated']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in date_keywords):
                return col
        
        # Check for columns that can be parsed as dates
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(100), errors='raise')
                return col
            except:
                continue
        
        return None
    
    def _apply_date_filter(self, df: pd.DataFrame, date_filter: Dict, date_column: str = None) -> Tuple[pd.DataFrame, str]:
        """Apply date filtering to dataframe"""
        if not date_column:
            date_column = self._find_date_column(df)
        
        if not date_column:
            return df, ""
        
        # Convert date column to datetime
        df_filtered = df.copy()
        try:
            df_filtered[date_column] = pd.to_datetime(df_filtered[date_column], errors='coerce')
        except:
            return df, ""
        
        # Remove rows with invalid dates
        df_filtered = df_filtered.dropna(subset=[date_column])
        
        filter_conditions = []
        filter_description_parts = []
        
        # Apply year filter
        if "year" in date_filter:
            year = date_filter["year"]
            filter_conditions.append(df_filtered[date_column].dt.year == year)
            filter_description_parts.append(str(year))
        
        # Apply month filter
        if "month" in date_filter:
            month = date_filter["month"]
            filter_conditions.append(df_filtered[date_column].dt.month == month)
            month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            filter_description_parts.append(month_names[month])
        
        # Apply day filter
        if "day" in date_filter:
            day = date_filter["day"]
            filter_conditions.append(df_filtered[date_column].dt.day == day)
            filter_description_parts.append(f"{day}")
        
        # Apply quarter filter
        if "quarter" in date_filter:
            quarter = date_filter["quarter"]
            filter_conditions.append(df_filtered[date_column].dt.quarter == quarter)
            filter_description_parts.append(f"Q{quarter}")
        
        # Combine all conditions
        if filter_conditions:
            combined_condition = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_condition &= condition
            
            df_filtered = df_filtered[combined_condition]
        
        filter_description = " ".join(filter_description_parts)
        if filter_description:
            filter_description = f" for {filter_description}"
        
        return df_filtered, filter_description
    
    def _handle_correlation(self, parsed_query: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle correlation queries"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            return {
                "status": "error",
                "response": "Need at least 2 numeric columns to calculate correlations.",
                "data": None,
                "credits_used": 1
            }
        
        correlations = df[numeric_cols].corr().to_dict()
        return {
            "status": "success",
            "response": "Here are the correlations between numeric columns:",
            "data": correlations,
            "credits_used": 4
        }
    
    def _handle_comparison(self, parsed_query: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle comparison queries"""
        return {
            "status": "success",
            "response": "Comparison analysis is coming soon! Try asking for specific aggregations or filters.",
            "data": "Feature in development",
            "credits_used": 1
        }
    
    def _handle_fallback(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle queries that don't match specific patterns"""
        return {
            "status": "success",
            "response": f"I understand you want to analyze: '{query}'. Try being more specific with queries like 'what is the average revenue?' or 'show me expenses for January 2015'.",
            "data": {
                "suggestions": [
                    "Use specific column names from your data",
                    "Include aggregation words like: average, sum, maximum, minimum",
                    "Add date filters like: 'for March 2015', 'in Q1 2020'",
                    "Ask for basic info: 'show first 5 rows', 'column names', 'summary statistics'"
                ],
                "available_columns": list(df.columns) if df is not None else []
            },
            "credits_used": 2
        } 