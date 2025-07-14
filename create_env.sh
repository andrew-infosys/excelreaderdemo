#!/bin/bash
echo "🚀 Creating .env file..."

cat > .env << 'ENV'
# API Keys Configuration
# Add your actual API keys below (remove the # to uncomment)

# Groq API Key (FREE) - Get from: https://console.groq.com/keys
# GROQ_API_KEY=gsk_your_groq_api_key_here

# OpenAI API Key (if you prefer OpenAI instead)
# OPENAI_API_KEY=sk_your_openai_api_key_here

# Default model settings
DEFAULT_MODEL=llama-3.1-8b-instant
ENV

echo "✅ .env file created!"
echo "📝 Edit the .env file and add your API keys"
echo "💡 Uncomment the lines by removing the # symbol"
echo ""
echo "To get a FREE Groq API key:"
echo "1. Visit: https://console.groq.com/keys"
echo "2. Sign up (no credit card required)"
echo "3. Create an API key"
echo "4. Replace 'gsk_your_groq_api_key_here' with your actual key"
