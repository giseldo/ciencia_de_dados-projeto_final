import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import gradio as gr

# Carregar a base de dados
file_path = "data.csv"
df = pd.read_csv(file_path)
# Remover ou substituir valores ausentes (exemplo: substituir pela média)
df.fillna(df.mean(), inplace=True)

# Renomear colunas para português
df.columns = ["Indice_Socioeconomico", "Horas_Estudo", "Horas_Sono", "Frequencia", "Nota"]

# Definir variáveis independentes (X) e variável dependente (y)
X = df[["Indice_Socioeconomico", "Horas_Estudo", "Horas_Sono", "Frequencia"]]
y = df["Nota"]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo RandomForest
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Previsão e cálculo do erro médio absoluto
y_pred = modelo.predict(X_test)
erro_medio = mean_

# Função para previsão
def prever(indice_socio, horas_estudo, horas_sono, frequencia):
    entrada = np.array([[float(indice_socio), float(horas_estudo), float(horas_sono), float(frequencia)]])
    previsao = modelo.predict(entrada)
    return f"Previsão da Nota: {float(previsao[0]):.2f} | Erro Médio: {erro_medio:.2f}"

# Criar a interface Gradio
iface = gr.Interface(
    fn=prever,
    inputs=[
        gr.Number(label="Índice Socioeconômico"),
        gr.Number(label="Horas de Estudo"),
        gr.Number(label="Horas de Sono"),
        gr.Number(label="Frequência (%)")
    ],
    outputs="text",
    title="Previsão de Notas com RandomForest",
    description="Insira os dados para prever a nota do aluno."
)

# Executar a aplicação
iface.launch()
