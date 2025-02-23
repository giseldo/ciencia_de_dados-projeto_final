import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar o arquivo CSV
df = pd.read_csv("data.csv")  # Substitua pelo caminho correto do arquivo

# Selecionar as colunas numéricas
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

def plot_scatter(x_column, y_column):
    """Gera um gráfico de dispersão para as variáveis escolhidas."""
    x = df[x_column]
    y = df[y_column]

    # Criar o gráfico de dispersão
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, color='gray', edgecolors='black')

    # Adicionar linha de tendência (regressão linear)
    if len(x.dropna()) > 1 and len(y.dropna()) > 1:
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * x + b, color='black', linewidth=2, label=f"y = {m:.2f}x + {b:.2f}")
    
    # Configurar os eixos e o título
    plt.title(f"{y_column} vs {x_column}", fontsize=14)
    plt.xlabel(x_column, fontsize=12)
    plt.ylabel(y_column, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Exibir opções para o usuário escolher
print("Variáveis disponíveis para gerar gráficos de dispersão:")
for i, col in enumerate(numeric_columns, start=1):
    print(f"{i}: {col}")

# Pedir ao usuário para selecionar as variáveis
try:
    x_idx = int(input("\nDigite o número correspondente à variável no eixo X: ")) - 1
    y_idx = int(input("Digite o número correspondente à variável no eixo Y: ")) - 1

    if x_idx in range(len(numeric_columns)) and y_idx in range(len(numeric_columns)):
        x_column = numeric_columns[x_idx]
        y_column = numeric_columns[y_idx]

        # Gerar o gráfico de dispersão
        plot_scatter(x_column, y_column)
    else:
        print("Índices inválidos. Por favor, tente novamente.")
except ValueError:
    print("Entrada inválida. Por favor, insira números inteiros correspondentes às variáveis.")
