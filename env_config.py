"""
Environment Configuration
Handles loading API keys from .env file or environment variables
"""

import os
from typing import Optional

def load_env_config():
    """
    Load environment configuration from .env file or environment variables
    Create a .env file in this directory with your API keys
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed. Using system environment variables only.")
    
    return {
        'groq_api_key': os.getenv('GROQ_API_KEY', ''),
        'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
        'pandasai_api_key': os.getenv('PANDASAI_API_KEY', ''),
        'default_model': os.getenv('DEFAULT_MODEL', 'llama-3.1-8b-instant')
    }

def get_api_key(service: str) -> Optional[str]:
    """Get API key for a specific service"""
    config = load_env_config()
    return config.get(f'{service}_api_key', '')

# Example .env file content (create this file in the same directory):
"""
# Copy the lines below to a file named .env (no extension)

# Groq API Key (FREE) - Get from: https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_api_key_here

# OpenAI API Key (if you prefer OpenAI instead)
OPENAI_API_KEY=sk_your_openai_api_key_here

# PandasAI API Key (if you want to try PandasAI)
PANDASAI_API_KEY=your_pandasai_api_key_here

# Default model settings
DEFAULT_MODEL=llama-3.1-8b-instant
""" 