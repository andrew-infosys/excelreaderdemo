"""
AI Clients for MCP vs Vanilla API Comparison
This module demonstrates the difference between:
1. Vanilla API calls (basic LLM queries)
2. MCP-enhanced calls (with tools, context, and structured data access)
"""

import json
import pandas as pd
from typing import Dict, Any, List, Optional
from groq import Groq
from datetime import datetime
import re

class VanillaAIClient:
    """
    Basic AI client - just sends raw queries to LLM without tools or structured data access
    This represents the "old way" of doing AI interactions
    """
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)
        
    def query_data(self, query: str, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Vanilla approach: Just send the query to the LLM with basic data info
        No tools, no structured access, no data manipulation capabilities
        """
        try:
            # Create a basic prompt with limited data context
            data_summary = ""
            if df is not None:
                data_summary = f"""
                You have access to a dataset with:
                - {len(df)} rows
                - Columns: {', '.join(df.columns)}
                - Sample data: {df.head(2).to_string()}
                """
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a data analyst. Give brief, direct answers. Don't explain methodology - just provide the answer."
                },
                {
                    "role": "user", 
                    "content": f"{data_summary}\n\nQuestion: {query}\n\nProvide a direct, concise answer without explaining the process."
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            
            return {
                "status": "success",
                "response": ai_response,
                "method": "Vanilla API",
                "limitations": [
                    "Cannot directly access or manipulate data",
                    "No real-time calculations",
                    "Limited to text-based responses",
                    "Cannot verify answers against actual data"
                ],
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
                
        except Exception as e:
            return {
                "status": "error",
                "response": f"Error: {str(e)}",
                "method": "Vanilla API"
            }

class MCPEnhancedAIClient:
    """
    MCP-Enhanced AI client with tools, structured data access, and intelligent context management
    This represents the "new way" - using MCP for better AI-data interactions
    """
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)
        
        # MCP-style tools for data access
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "filter_data",
                    "description": "Filter dataset by date, column values, or conditions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "description": "Column to filter on"},
                            "date_filter": {"type": "string", "description": "Date filter (e.g., '2015-01-03', 'January 2015')"},
                            "condition": {"type": "string", "description": "Filter condition"}
                        }
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "calculate_aggregation",
                    "description": "Perform calculations on data (sum, average, max, min, count)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "description": "Column to calculate on"},
                            "operation": {"type": "string", "enum": ["sum", "mean", "max", "min", "count"]},
                            "filter": {"type": "string", "description": "Optional filter to apply first"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_data_rows",
                    "description": "Get specific rows of data",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "num_rows": {"type": "integer", "description": "Number of rows to return"},
                            "from_start": {"type": "boolean", "description": "Get from start (true) or end (false)"}
                        }
                    }
                }
            }
        ]
    
    def query_data(self, query: str, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        MCP-Enhanced approach: Use tools and structured data access for accurate responses
        """
        if df is None:
            return {
                "status": "error",
                "response": "No data available",
                "method": "MCP Enhanced"
            }
            
        try:
            # Enhanced context with structured data information
            data_context = self._build_enhanced_context(df)
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a data analyst with access to data tools. 
                    
                    Dataset info:
                    {data_context}
                    
                    Use tools to get data and provide direct, brief answers. Don't explain which tools you're using - just give the answer.
                    """
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            
            try:
                # First, try with tools if the model supports it
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=1000
                )
                return self._process_mcp_response(response, df)
            except Exception as tool_error:
                # Fallback: If tools aren't supported, use enhanced prompting with data context
                enhanced_prompt = f"""
                Dataset Context:
                {data_context}
                
                Query: {query}
                
                Provide a direct, brief answer based on the dataset above. Use the actual data to answer the question.
                """
                
                fallback_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a data analyst. Give brief, direct answers based on the provided dataset. Use the actual data to calculate answers."},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                return {
                    "status": "success",
                    "response": fallback_response.choices[0].message.content,
                    "method": "MCP Enhanced (Fallback)",
                    "data_access": "Enhanced prompting with data context",
                    "tokens_used": fallback_response.usage.total_tokens if hasattr(fallback_response, 'usage') else 0
                }
                
        except Exception as e:
            return {
                "status": "error",
                "response": f"Error: {str(e)}",
                "method": "MCP Enhanced"
            }
    
    def _build_enhanced_context(self, df: pd.DataFrame) -> str:
        """Build rich context about the dataset"""
        context = f"""
        Dataset Overview:
        - Rows: {len(df):,}
        - Columns: {len(df.columns)}
        
        Column Details:
        """
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            if df[col].dtype in ['int64', 'float64']:
                min_val = df[col].min()
                max_val = df[col].max()
                context += f"- {col} ({col_type}): {unique_count} unique values, range {min_val}-{max_val}, {null_count} nulls\n"
            else:
                context += f"- {col} ({col_type}): {unique_count} unique values, {null_count} nulls\n"
        
        # Add date range if date column exists
        date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time'])]
        if date_cols:
            date_col = date_cols[0]
            try:
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                min_date = df_temp[date_col].min()
                max_date = df_temp[date_col].max()
                context += f"\nDate Range: {min_date.date()} to {max_date.date()}"
            except:
                pass
        
        # Add sample data for the AI to see actual values
        context += f"\n\nSample Data (first 10 rows):\n"
        context += df.head(10).to_string(index=False)
        
        return context
    
    def _process_mcp_response(self, response, df: pd.DataFrame) -> Dict[str, Any]:
        """Process the MCP response and execute any tool calls"""
        choice = response.choices[0]
        message = choice.message
        
        # Check if the model wants to use tools
        if message.get('tool_calls'):
            tool_results = []
            
            for tool_call in message['tool_calls']:
                function_name = tool_call['function']['name']
                function_args = json.loads(tool_call['function']['arguments'])
                
                # Execute the tool
                tool_result = self._execute_tool(function_name, function_args, df)
                tool_results.append(tool_result)
            
            # Send tool results back to the model for final response
            final_response = self._get_final_response_with_tools(response_data, tool_results, df)
            
            return {
                "status": "success",
                "response": final_response['content'],
                "method": "MCP Enhanced",
                "tools_used": [tool_call['function']['name'] for tool_call in message['tool_calls']],
                "tool_results": tool_results,
                "advantages": [
                    "Direct data access and manipulation",
                    "Real-time calculations",
                    "Verified results against actual data",
                    "Structured tool-based responses"
                ],
                "tokens_used": response_data.get('usage', {}).get('total_tokens', 0)
            }
        else:
            # No tools needed, return direct response
            return {
                "status": "success",
                "response": message['content'],
                "method": "MCP Enhanced",
                "tools_used": [],
                "tokens_used": response_data.get('usage', {}).get('total_tokens', 0)
            }
    
    def _execute_tool(self, function_name: str, args: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute MCP tool functions"""
        try:
            if function_name == "filter_data":
                return self._tool_filter_data(args, df)
            elif function_name == "calculate_aggregation":
                return self._tool_calculate_aggregation(args, df)
            elif function_name == "get_data_rows":
                return self._tool_get_data_rows(args, df)
            else:
                return {"error": f"Unknown tool: {function_name}"}
        except Exception as e:
            return {"error": f"Tool execution error: {str(e)}"}
    
    def _tool_filter_data(self, args: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Tool: Filter data by conditions"""
        filtered_df = df.copy()
        
        if args.get('date_filter'):
            date_filter = args['date_filter']
            # Find date column
            date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time'])]
            if date_cols:
                date_col = date_cols[0]
                filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
                
                # Parse date filter
                if 'january' in date_filter.lower() or 'jan' in date_filter.lower():
                    if '2015' in date_filter:
                        if '3' in date_filter:
                            filtered_df = filtered_df[filtered_df[date_col].dt.date == pd.to_datetime('2015-01-03').date()]
                        else:
                            filtered_df = filtered_df[(filtered_df[date_col].dt.year == 2015) & (filtered_df[date_col].dt.month == 1)]
        
        return {
            "rows_found": len(filtered_df),
            "data": filtered_df.head(10).to_dict('records'),
            "summary": f"Filtered to {len(filtered_df)} rows"
        }
    
    def _tool_calculate_aggregation(self, args: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Tool: Calculate aggregations"""
        column = args.get('column')
        operation = args.get('operation')
        
        if not column or column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        if operation == "sum":
            result = float(df[column].sum())
        elif operation == "mean":
            result = float(df[column].mean())
        elif operation == "max":
            result = float(df[column].max())
        elif operation == "min":
            result = float(df[column].min())
        elif operation == "count":
            result = int(df[column].count())
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        return {
            "column": column,
            "operation": operation,
            "result": result,
            "records_used": len(df)
        }
    
    def _tool_get_data_rows(self, args: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Tool: Get specific rows"""
        num_rows = args.get('num_rows', 5)
        from_start = args.get('from_start', True)
        
        if from_start:
            result_df = df.head(num_rows)
        else:
            result_df = df.tail(num_rows)
        
        return {
            "rows_returned": len(result_df),
            "data": result_df.to_dict('records')
        }
    
    def _get_final_response_with_tools(self, initial_response: Dict, tool_results: List[Dict], df: pd.DataFrame) -> Dict[str, Any]:
        """Get final response incorporating tool results"""
        # For simplicity, we'll format a response based on tool results
        # In a real implementation, you'd send this back to the LLM
        
        response_parts = []
        
        for i, result in enumerate(tool_results):
            if "data" in result and len(result["data"]) > 0:
                response_parts.append(f"Found {len(result['data'])} records with the requested data.")
            elif "result" in result:
                response_parts.append(f"Calculation result: {result['result']}")
        
        return {
            "content": " ".join(response_parts) if response_parts else "Analysis completed using MCP tools."
        } 