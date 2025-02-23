import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv("data.csv")

# Gerar estatísticas descritivas transpostas
df_describe = df.describe().T

# Salvar como HTML e abrir no navegador
df_describe.to_html("resumo_estatistico.html")

import webbrowser
webbrowser.open("resumo_estatistico.html")
