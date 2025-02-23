import gradio as gr
import requests
import os

# Defina sua chave da API do Groq aqui ou use uma variável de ambiente
GROQ_API_KEY = "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"  # Substitua pela sua chave real

# Endpoint da API do Groq (verifique se está atualizado)
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"

# Função para interagir com a API do Groq
def chat_with_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "mixtral-8x7b-32768",  # Substitua pelo modelo adequado, se necessário
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,  # Controle de criatividade
        "max_tokens": 200,   # Limite de resposta
    }

    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        response_data = response.json()

        if "choices" in response_data:
            return response_data["choices"][0]["message"]["content"]
        else:
            return "Erro: Resposta inesperada da API."

    except Exception as e:
        return f"Erro ao acessar a API: {str(e)}"

# Interface com Gradio
iface = gr.Interface(
    fn=chat_with_groq,
    inputs="text",
    outputs="text",
    title="Chat com Groq API",
    description="Digite um prompt e receba uma resposta do modelo da Groq."
)

# Iniciar a interface
iface.launch()


