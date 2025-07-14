"""
Configuration settings and constants for the Excel Reading Agent
"""

# App Configuration
APP_CONFIG = {
    "page_title": "Excel Reading Agent - Data Analysis",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# API Configuration
API_CONFIG = {
    "base_url": "https://api.pandas-agi.com/v1",
    "timeout": 30,
    "max_retries": 3
}

# File Processing
SUPPORTED_FILE_TYPES = ['csv', 'xlsx', 'xls']
MAX_FILE_SIZE_MB = 50

# Model Options
AI_MODELS = [
    "gpt-4-turbo",
    "claude-3-sonnet", 
    "gemini-pro",
    "gpt-3.5-turbo"
]

# Sample Queries
SAMPLE_QUERIES = [
    "What are the first 5 rows?",
    "Show me summary statistics",
    "What are the column names?", 
    "What is the average revenue?",
    "What is the maximum revenue?",
    "What is the total revenue?",
    "What is the average revenue for march 2015?",
    "Show me correlations between columns",
    "How many records are there?",
    "What are the unique values in each column?",
    "What expenses were on January 3rd, 2015?",
    "Show me all data for Q1 2015",
    "What is the monthly trend for revenue?",
    "Compare expenses between different quarters"
]

# CSS Styling
CSS_STYLES = """
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
    
    .query-tip {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
"""

# Query Processing Configuration
QUERY_CONFIG = {
    "date_formats": [
        "%Y-%m-%d",
        "%m/%d/%Y", 
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S"
    ],
    "common_column_mappings": {
        "revenue": ["revenue", "sales", "income", "earnings"],
        "expenses": ["expenses", "costs", "spending", "expenditure"],
        "profit": ["profit", "net income", "earnings"],
        "date": ["date", "time", "timestamp", "created_at", "updated_at"],
        "amount": ["amount", "value", "total", "sum"],
        "quantity": ["quantity", "qty", "count", "number"]
    },
    "aggregation_functions": {
        "average": ["average", "avg", "mean"],
        "sum": ["sum", "total", "add"],
        "maximum": ["max", "maximum", "highest", "largest"],
        "minimum": ["min", "minimum", "lowest", "smallest"],
        "count": ["count", "number of", "how many"]
    }
}

# Chart Configuration
CHART_CONFIG = {
    "color_schemes": {
        "Default": None,
        "Viridis": "viridis", 
        "Plasma": "plasma",
        "Inferno": "inferno",
        "Cividis": "cividis",
        "Blues": "blues",
        "Reds": "reds", 
        "Greens": "greens",
        "Pastel": "pastel"
    },
    "numeric_chart_types": [
        "Histogram", "Box Plot", "Violin Plot", "Strip Plot",
        "Density Plot", "Q-Q Plot", "Line Chart"
    ],
    "categorical_chart_types": [
        "Bar Chart", "Pie Chart", "Donut Chart", "Treemap",
        "Sunburst", "Count Plot"
    ]
} 