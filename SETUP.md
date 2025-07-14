# 🚀 Quick Setup Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Set Up API Keys (Optional but Recommended)

Create a `.env` file in this directory with your API keys:

```bash
# Create .env file
touch .env
```

Add your API keys to the `.env` file:

```env
# Groq API Key (FREE) - Get from: https://console.groq.com/keys
GROQ_API_KEY=gsk_your_actual_groq_api_key_here

# OpenAI API Key (if you prefer OpenAI instead)
OPENAI_API_KEY=sk_your_actual_openai_api_key_here

# Default model settings
DEFAULT_MODEL=llama-3.1-8b-instant
```

## 3. Get Free API Keys

### Groq (Recommended - FREE)
1. Visit: https://console.groq.com/keys
2. Sign up (no credit card required)
3. Create an API key
4. Copy the key (starts with `gsk_`) to your `.env` file

### OpenAI (Alternative)
1. Visit: https://platform.openai.com/api-keys
2. Sign up and add billing info
3. Create an API key
4. Copy the key (starts with `sk-`) to your `.env` file

## 4. Run the Application

```bash
streamlit run app.py
```

## 5. Using the App

1. **Upload Data**: Upload your Excel/CSV file
2. **Choose Mode**: 
   - **Local Processing**: No API needed, pattern matching
   - **Vanilla AI**: Basic AI without tools
   - **MCP Enhanced AI**: AI with data access tools (best results)
3. **Ask Questions**: Try queries like:
   - "What is expenses on jan 3rd of 2015?"
   - "Show me the average revenue for Q1 2015"
   - "What are the top 5 expenses?"

## 🎯 Why Use .env File?

- ✅ **Automatic loading** - No need to enter API keys each time
- ✅ **Security** - Keys are not visible in the UI
- ✅ **Version control safe** - .env files are ignored by git
- ✅ **Easy switching** - Change keys without modifying code

## 📝 Notes

- The `.env` file is optional - you can still enter API keys in the UI
- If you have a `.env` file, the app will show "✅ API key loaded from .env file"
- You can override .env keys by entering a different key in the UI
- The app defaults to Groq's free tier for best experience 