!pip install -q -U google-generativeai

# Import the Python SDK
import google.generativeai as genai
from google.colab import userdata
userdata.get('GOOGLE_API_KEY')

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("Escreva um markdown com uma introdução sobre os fundamentos do Gemini API")
print(response.text)
