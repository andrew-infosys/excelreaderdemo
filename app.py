"""
Excel Reading Agent - AI-Powered with MCP vs Vanilla Comparison
"""

import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path

# Import our modular components
from ui_components import UIComponents
from api_client import ExcelReaderAPI
from ai_clients import VanillaAIClient, MCPEnhancedAIClient
from env_config import load_env_config, get_api_key

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'api_key' not in st.session_state:
        # Try to load API key from .env file first
        env_key = get_api_key('groq')
        st.session_state.api_key = env_key if env_key else ""
    if 'ai_mode' not in st.session_state:
        st.session_state.ai_mode = "Local Processing"

def load_data_file(uploaded_file) -> pd.DataFrame:
    """Load and process uploaded data file"""
    try:
        # Check file size (50MB limit)
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("File too large. Maximum size is 50MB.")
            return None
        
        # Read the file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        # Basic data validation
        if df.empty:
            st.error("The uploaded file is empty.")
            return None
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def render_ai_mode_selector():
    """Render AI mode selection"""
    st.sidebar.subheader("🤖 AI Processing Mode")
    
    mode = st.sidebar.radio(
        "Choose processing method:",
        options=["Local Processing", "Vanilla AI", "MCP Enhanced AI"],
        help="Compare different approaches to data analysis"
    )
    
    if mode in ["Vanilla AI", "MCP Enhanced AI"]:
        # Check if we have a key from .env file
        env_key = get_api_key('groq')
        
        if env_key:
            st.sidebar.success("✅ API key loaded from .env file")
            # Allow override if needed
            override_key = st.sidebar.text_input(
                "Override API Key (optional)",
                type="password",
                placeholder="Leave empty to use .env key",
                help="Override the .env file API key if needed"
            )
            api_key = override_key if override_key else env_key
        else:
            api_key = st.sidebar.text_input(
                "Groq API Key (FREE)",
                type="password",
                placeholder="gsk_...",
                help="Required for AI-powered analysis"
            )
        
        st.session_state.api_key = api_key
        
        if not api_key:
            st.sidebar.warning("⚠️ API key required for AI modes")
            st.sidebar.info("🆓 Get a **FREE** Groq API key at: https://console.groq.com/keys")
            st.sidebar.info("💡 **Tip:** Create a `.env` file to auto-load your API key")
            return "Local Processing"
    
    st.session_state.ai_mode = mode
    return mode

def render_comparison_info():
    """Render information about the different modes"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Method Comparison")
    
    with st.sidebar.expander("🔍 See Differences"):
        st.markdown("""
        **🔧 Local Processing:**
        - Rule-based pattern matching
        - Fast, offline processing
        - No API costs
        - Limited understanding
        
        **🤖 Vanilla AI:**
        - Basic LLM queries
        - Text-only responses
        - No data manipulation
        - Cannot verify answers
        
        **✨ MCP Enhanced AI:**
        - Tool-assisted analysis
        - Direct data access
        - Real-time calculations  
        - Verified results
        """)

def render_ai_comparison_results(query: str, df: pd.DataFrame):
    """Render side-by-side comparison of AI approaches"""
    if not st.session_state.api_key:
        st.error("Please enter an OpenAI API key to use AI features")
        return
    
    st.subheader("🔬 AI Method Comparison")
    st.markdown("**Query:** " + query)
    
    # Create two columns for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Vanilla AI Response")
        with st.spinner("Getting vanilla AI response..."):
            vanilla_client = VanillaAIClient(st.session_state.api_key)
            vanilla_result = vanilla_client.query_data(query, df)
            
            if vanilla_result['status'] == 'success':
                st.success("✅ " + vanilla_result['response'])
                
                # Show limitations
                if 'limitations' in vanilla_result:
                    with st.expander("⚠️ Limitations"):
                        for limitation in vanilla_result['limitations']:
                            st.warning(f"• {limitation}")
                
                st.caption(f"Tokens used: {vanilla_result.get('tokens_used', 0)}")
            else:
                st.error("❌ " + vanilla_result['response'])
    
    with col2:
        st.markdown("### ✨ MCP Enhanced AI Response")
        with st.spinner("Getting MCP-enhanced response..."):
            mcp_client = MCPEnhancedAIClient(st.session_state.api_key)
            mcp_result = mcp_client.query_data(query, df)
            
            if mcp_result['status'] == 'success':
                st.success("✅ " + mcp_result['response'])
                
                # Show tools used
                if 'tools_used' in mcp_result and mcp_result['tools_used']:
                    with st.expander("🛠️ Tools Used"):
                        for tool in mcp_result['tools_used']:
                            st.info(f"• {tool}")
                
                # Show advantages
                if 'advantages' in mcp_result:
                    with st.expander("✨ Advantages"):
                        for advantage in mcp_result['advantages']:
                            st.success(f"• {advantage}")
                
                # Show tool results
                if 'tool_results' in mcp_result and mcp_result['tool_results']:
                    with st.expander("📊 Tool Results"):
                        for i, result in enumerate(mcp_result['tool_results']):
                            st.json(result)
                
                st.caption(f"Tokens used: {mcp_result.get('tokens_used', 0)}")
            else:
                st.error("❌ " + mcp_result['response'])
    
    # Summary comparison
    st.markdown("---")
    st.subheader("📈 Performance Comparison")
    
    comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
    
    with comparison_col1:
        st.metric(
            "Accuracy", 
            "High" if mcp_result.get('status') == 'success' and 'tools_used' in mcp_result else "Medium",
            "MCP uses tools for verified results"
        )
    
    with comparison_col2:
        vanilla_tokens = vanilla_result.get('tokens_used', 0)
        mcp_tokens = mcp_result.get('tokens_used', 0)
        st.metric(
            "Token Efficiency",
            f"{mcp_tokens}/{vanilla_tokens}" if vanilla_tokens > 0 else "N/A",
            "MCP may use more tokens but provides better results"
        )
    
    with comparison_col3:
        st.metric(
            "Capabilities",
            "Enhanced" if mcp_result.get('tools_used') else "Basic",
            "MCP can perform actual data operations"
        )

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Setup page and styling
    UIComponents.setup_page_config()
    UIComponents.apply_custom_css()
    
    # Render header
    st.markdown('<h1 class="main-header">🤖 Excel Reading Agent - AI Comparison</h1>', unsafe_allow_html=True)
    st.markdown("**Compare Local Processing vs Vanilla AI vs MCP-Enhanced AI**")
    st.markdown("---")
    
    # AI mode selector
    ai_mode = render_ai_mode_selector()
    render_comparison_info()
    
    # File upload section
    uploaded_file = UIComponents.render_file_upload()
    
    # Process uploaded file
    if uploaded_file is not None:
        # Load data
        df = load_data_file(uploaded_file)
        
        if df is not None:
            # Store in session state
            st.session_state.df = df
            
            # Show success message
            UIComponents.render_success_message(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Render data preview
            UIComponents.render_data_preview(df)
            
            # Query interface based on mode
            st.markdown("---")
            
            if ai_mode == "Local Processing":
                st.subheader("💬 Local Processing (Pattern Matching)")
                api_client = ExcelReaderAPI()
                query = UIComponents.render_chat_interface(api_client, df)
                
            elif ai_mode == "Vanilla AI":
                st.subheader("🤖 Vanilla AI Mode")
                if st.session_state.api_key:
                    query = st.text_input(
                        "Ask your data question:",
                        placeholder="e.g., what is expenses on jan 3rd of 2015?"
                    )
                    if query:
                        vanilla_client = VanillaAIClient(st.session_state.api_key)
                        result = vanilla_client.query_data(query, df)
                        
                        if result['status'] == 'success':
                            st.success("✅ " + result['response'])
                            if 'limitations' in result:
                                with st.expander("⚠️ Limitations of Vanilla AI"):
                                    for limitation in result['limitations']:
                                        st.warning(f"• {limitation}")
                        else:
                            st.error("❌ " + result['response'])
                else:
                    st.info("Please enter an OpenAI API key in the sidebar")
                    
            elif ai_mode == "MCP Enhanced AI":
                st.subheader("✨ MCP Enhanced AI Mode")
                if st.session_state.api_key:
                    query = st.text_input(
                        "Ask your data question:",
                        placeholder="e.g., what is expenses on jan 3rd of 2015?"
                    )
                    if query:
                        mcp_client = MCPEnhancedAIClient(st.session_state.api_key)
                        result = mcp_client.query_data(query, df)
                        
                        if result['status'] == 'success':
                            st.success("✅ " + result['response'])
                            
                            if 'tools_used' in result and result['tools_used']:
                                with st.expander("🛠️ MCP Tools Used"):
                                    for tool in result['tools_used']:
                                        st.info(f"• {tool}")
                            
                            if 'tool_results' in result and result['tool_results']:
                                with st.expander("📊 Tool Results"):
                                    for i, tool_result in enumerate(result['tool_results']):
                                        if 'data' in tool_result and tool_result['data']:
                                            st.dataframe(pd.DataFrame(tool_result['data']))
                                        else:
                                            st.json(tool_result)
                        else:
                            st.error("❌ " + result['response'])
                else:
                    st.info("Please enter an OpenAI API key in the sidebar")
            
            # Comparison mode
            st.markdown("---")
            st.subheader("🔬 AI Method Comparison")
            comparison_query = st.text_input(
                "Query to compare:", 
                value="what is expenses on jan 3rd of 2015",
                help="Enter any query to compare AI approaches"
            )
            
            if st.button("Run AI Comparison", help="Compare Vanilla AI vs MCP Enhanced AI"):
                if st.session_state.api_key and comparison_query:
                    render_ai_comparison_results(comparison_query, df)
                elif not st.session_state.api_key:
                    st.error("Please enter a Groq API key to run the comparison")
                else:
                    st.error("Please enter a query to compare")
            
            # Simple visualizations
            st.markdown("---")
            UIComponents.render_simple_visualizations(df)
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 **To get started:** Upload an Excel (.xlsx) or CSV (.csv) file using the file uploader above.")
        
        st.markdown("### 🤔 What's the difference?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔧 Local Processing**
            - Pattern matching & rules
            - Fast, offline processing  
            - No API costs
            - Limited understanding
            
            *Good for: Simple queries, privacy*
            """)
        
        with col2:
            st.markdown("""
            **🤖 Vanilla AI**
            - Basic LLM queries
            - Text-only responses
            - No data manipulation
            - Cannot verify answers
            
            *Issues: Hallucination, no tools*
            """)
        
        with col3:
            st.markdown("""
            **✨ MCP Enhanced AI**
            - Tool-assisted analysis
            - Direct data access
            - Real-time calculations
            - Verified results
            
            *Best: Accurate, powerful, reliable*
            """)

if __name__ == "__main__":
    main() 