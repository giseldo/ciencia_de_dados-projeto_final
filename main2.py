# Carregar as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import gradio as gr

# Carregar a base de dados
file_path = "data.csv"
df = pd.read_csv(file_path)

# Remover valores ausentes substituindo pela média
df.fillna(df.mean(), inplace=True)

# Remover duplicatas, se existirem
df.drop_duplicates(inplace=True)

# Renomear colunas para português
df.columns = ["Indice_Socioeconomico", "Horas_Estudo", "Horas_Sono", "Frequencia", "Nota"]

# Criar nova variável de interação entre estudo e frequência
df["Interacao_Estudo_Frequencia"] = df["Horas_Estudo"] * df["Frequencia"]

# Definir variáveis independentes (X) e variável dependente (y)
X = df[["Indice_Socioeconomico", "Horas_Estudo", "Horas_Sono", "Frequencia", "Interacao_Estudo_Frequencia"]]
y = df["Nota"]

# Aplicar normalização Z-Score APÓS a criação da nova variável
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinar o modelo RandomForest
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Previsão e cálculo do erro médio absoluto
y_pred = modelo.predict(X_test)
erro_medio = mean_absolute_error(y_test, y_pred)

# Função para previsão
def prever(indice_socio, horas_estudo, horas_sono, frequencia):
    # Criar a nova variável de interação
    interacao = float(horas_estudo) * float(frequencia)
    
    # Criar a entrada para previsão
    entrada = np.array([[float(indice_socio), float(horas_estudo), float(horas_sono), float(frequencia), interacao]])
    
    # Aplicar a normalização
    entrada_scaled = scaler.transform(entrada)
    
    # Fazer a previsão
    previsao = modelo.predict(entrada_scaled)
    
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
