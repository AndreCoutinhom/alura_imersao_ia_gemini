# Instalando o SDK do Google
!pip install -q -U google-generativeai

# Importando a API key
import google.generativeai as genai
from google.colab import userdata
userdata.get('GOOGLE_API_KEY')

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Listar os modelos disponíveis
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
# models/gemini-1.0-pro
# models/gemini-1.0-pro-001
# models/gemini-1.0-pro-latest
# models/gemini-1.0-pro-vision-latest
# models/gemini-1.5-pro-latest
# models/gemini-pro
# models/gemini-pro-vision

generation_config = {
    "candidate_count": 1,
    "temperature": 0.5,
}

safety_settings = {
    "HARASSMENT": "BLOCK_NONE",
    "HATE": "BLOCK_NONE",
    "SEXUAL": "BLOCK_NONE",
    "DANGEROUS": "BLOCK_NONE",
}

# Inicializando o modelo
model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

response = model.generate_content("Escreva um markdown com uma introdução sobre os fundamentos do Gemini API.")
print(response.text)

# Chat

chat = model.start_chat(history=[])

prompt = input("Esperando prompt: ")

while prompt != "fim": 
  response = chat.send_message(prompt)
  print("Resposta: ", response.text, "\n")
  prompt = input("Esperando prompt: ")
