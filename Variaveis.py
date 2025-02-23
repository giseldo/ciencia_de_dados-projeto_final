import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
df = pd.read_csv("data.csv")  # Substitua pelo caminho correto do arquivo

# Selecionar as colunas numéricas
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Configurar subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

# Gerar histogramas para cada variável numérica
for i, column in enumerate(numeric_columns):
    df[column].hist(bins=10, ax=axes[i], edgecolor='black', color='orange')
    axes[i].set_title(column)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Frequência")

# Remover gráficos extras (caso existam mais subplots do que colunas)
for j in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
