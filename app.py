import matplotlib
matplotlib.use('Agg')  # Usar backend não interativo para evitar conflitos com tkinter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier
import gradio as gr
import os
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')

# Função para carregar e pré-processar os dados
def carregar_e_preprocessar_dados(caminho_arquivo):
    print("Iniciando carregamento e pré-processamento dos dados...")
    df = pd.read_csv(caminho_arquivo)
    
    # Remover linhas com valores nulos
    print("  Removendo linhas com valores nulos...")
    df = df.dropna()
    
    # Criar uma cópia do DataFrame para visualizações
    print("  Criando cópia do DataFrame para visualizações...")
    df_viz = df.copy()
    
    # Codificar variáveis categóricas
    print("  Codificando variáveis categóricas...")
    df = pd.get_dummies(df, columns=['Gender', 'Preferred_Learning_Style', 'Participation_in_Discussions', 
                                     'Use_of_Educational_Tech', 'Self_Reported_Stress_Level'], drop_first=True)
    
    # Mapear Final_Grade para valores numéricos
    print("  Mapeando Final_Grade para valores numéricos...")
    mapeamento_grades = {'A': 3, 'B': 2, 'C': 1, 'D': 0}  # Sem 'F'
    df['Final_Grade'] = df['Final_Grade'].map(mapeamento_grades)
    
    # Verificar valores únicos de Final_Grade
    print("Valores únicos de Final_Grade:", df['Final_Grade'].unique())
    
    # Normalizar variáveis numéricas
    print("  Normalizando variáveis numéricas...")
    colunas_numericas = ['Age', 'Study_Hours_per_Week', 'Online_Courses_Completed', 
                         'Assignment_Completion_Rate (%)', 'Exam_Score (%)', 
                         'Attendance_Rate (%)', 'Time_Spent_on_Social_Media (hours/week)', 
                         'Sleep_Hours_per_Night']
    scaler = MinMaxScaler()
    df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])
    
    print("Carregamento e pré-processamento concluídos.")
    return df, df_viz, scaler, colunas_numericas, mapeamento_grades

# Função para gerar estatísticas descritivas
def estatisticas_descritivas(df):
    print("Gerando estatísticas descritivas...")
    estatisticas = df.describe()
    print("Estatísticas descritivas geradas.")
    return estatisticas

# Função para gerar visualizações
def gerar_visualizacoes(df_viz, caminho_salvar='graficos/'):
    print("Iniciando geração de visualizações...")
    # Criar o diretório 'graficos' se não existir
    os.makedirs(caminho_salvar, exist_ok=True)
    print(f"  Diretório '{caminho_salvar}' criado ou já existente.")
    
    # Lista de variáveis numéricas para histogramas
    colunas_numericas = ['Age', 'Study_Hours_per_Week', 'Online_Courses_Completed', 
                         'Assignment_Completion_Rate (%)', 'Exam_Score (%)', 
                         'Attendance_Rate (%)', 'Time_Spent_on_Social_Media (hours/week)', 
                         'Sleep_Hours_per_Night']
    
    # Gerar histogramas para todas as variáveis numéricas
    for coluna in colunas_numericas:
        print(f"  Gerando histograma para {coluna}...")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_viz[coluna], kde=True)
        plt.title(f'Distribuição de {coluna}')
        # Substituir caracteres inválidos no nome do arquivo
        nome_arquivo = coluna.replace(' (%)', '').replace(' (hours/week)', '').replace('/', '_').lower()
        plt.savefig(f'{caminho_salvar}hist_{nome_arquivo}.png')
        plt.close()
        print(f"  Histograma salvo como 'hist_{nome_arquivo}.png'.")
    
    # Countplot de Preferred_Learning_Style por Final_Grade
    print("  Gerando countplot de Preferred_Learning_Style por Final_Grade...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Preferred_Learning_Style', hue='Final_Grade', data=df_viz)
    plt.title('Nota Final por Estilo de Aprendizado')
    plt.savefig(f'{caminho_salvar}bar_learning_style.png')
    plt.close()
    print("  Countplot salvo como 'bar_learning_style.png'.")
    
    # Boxplot de Exam_Score por Final_Grade
    print("  Gerando boxplot de Exam_Score por Final_Grade...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Final_Grade', y='Exam_Score (%)', data=df_viz)
    plt.title('Pontuação em Exames por Nota Final')
    plt.savefig(f'{caminho_salvar}box_exam_score_grade.png')
    plt.close()
    print("  Boxplot salvo como 'box_exam_score_grade.png'.")
    
    # Heatmap de correlação
    print("  Gerando heatmap de correlação...")
    plt.figure(figsize=(12, 8))
    corr = df_viz[['Age', 'Study_Hours_per_Week', 'Exam_Score (%)', 'Attendance_Rate (%)']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.savefig(f'{caminho_salvar}heatmap_corr.png')
    plt.close()
    print("  Heatmap salvo como 'heatmap_corr.png'.")
    
    # Gráfico de dispersão: Exam_Score vs Study_Hours_per_Week
    print("  Gerando scatterplot de Exam_Score vs Study_Hours_per_Week...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Study_Hours_per_Week', y='Exam_Score (%)', hue='Final_Grade', data=df_viz)
    plt.title('Pontuação em Exames vs. Horas de Estudo por Semana')
    plt.savefig(f'{caminho_salvar}scatter_exam_score_study_hours.png')
    plt.close()
    print("  Scatterplot salvo como 'scatter_exam_score_study_hours.png'.")
    
    # Fechar todas as figuras abertas para liberar memória
    plt.close('all')
    print("Geração de visualizações concluída.")

# Função para treinar e avaliar modelos
def treinar_e_avaliar_modelos(X, y, caminho_salvar='graficos/'):
    
    print("Iniciando treinamento e avaliação dos modelos...")
    # Verificar valores únicos de y
    unique_classes = np.unique(y)
    print("Classes encontradas em y:", unique_classes)
    
    print("  Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    modelos = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss', num_class=len(unique_classes)),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        print(f"  Treinando modelo {nome}...")
        # Ajuste de hiperparâmetros
        if nome == 'Random Forest':
            param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
        elif nome == 'XGBoost':
            param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
        else:
            param_grid = {'C': [1, 10], 'gamma': ['scale', 'auto']}
        
        grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Melhor modelo
        melhor_modelo = grid_search.best_estimator_
        
        # Previsões
        print(f"  Fazendo previsões com {nome}...")
        y_pred = melhor_modelo.predict(X_test)
        
        # Métricas
        print(f"  Calculando métricas para {nome}...")
        acuracia = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, melhor_modelo.predict_proba(X_test), multi_class='ovr')
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        cm = confusion_matrix(y_test, y_pred)
        
        # Validação cruzada
        print(f"  Executando validação cruzada para {nome}...")
        scores = cross_val_score(melhor_modelo, X, y, cv=5, scoring='accuracy')
        
        resultados[nome] = {
            'modelo': melhor_modelo,
            'acuracia': acuracia,
            'f1_score': f1,
            'auc': auc,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'matriz_confusao': cm,
            'cv_scores': scores,
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores)
        }
        
        # Visualizar matriz de confusão
        print(f"  Gerando matriz de confusão para {nome}...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {nome}')
        plt.savefig(f'{caminho_salvar}matriz_confusao_{nome.lower().replace(" ", "_")}.png')
        plt.close()
        print(f"  Matriz de confusão salva como 'matriz_confusao_{nome.lower().replace(' ', '_')}.png'.")
    
    # Gerar tabela de métricas
    print("  Gerando tabela de métricas...")
    tabela_metricas = pd.DataFrame({
        'Modelo': [nome for nome in resultados.keys()],
        'Acurácia': [resultados[nome]['acuracia'] for nome in resultados.keys()],
        'F1-Score': [resultados[nome]['f1_score'] for nome in resultados.keys()],
        'AUC': [resultados[nome]['auc'] for nome in resultados.keys()],
        'MAE': [resultados[nome]['mae'] for nome in resultados.keys()],
        'MSE': [resultados[nome]['mse'] for nome in resultados.keys()],
        'RMSE': [resultados[nome]['rmse'] for nome in resultados.keys()],
        'CV Média': [resultados[nome]['cv_mean'] for nome in resultados.keys()],
        'CV Desvio Padrão': [resultados[nome]['cv_std'] for nome in resultados.keys()]
    })
    
    # Arredondar valores para 4 casas decimais
    tabela_metricas = tabela_metricas.round(4)
    
    # Exibir tabela no terminal
    print("\nTabela de Métricas dos Modelos:")
    print(tabela_metricas.to_string(index=False))
    
    # Salvar tabela como CSV
    tabela_metricas.to_csv('metricas_modelos.csv', index=False)
    print("Tabela de métricas salva em 'metricas_modelos.csv'.")
    
    print("Treinamento e avaliação dos modelos concluídos.")
    return resultados, X_train, X_test, y_train, y_test

# Função para interface Gradio com dropdowns traduzidos
def prever_nota_final(age, study_hours, online_courses, assignment_completion, exam_score, 
                     attendance_rate, social_media, sleep_hours, genero, estilo_aprendizado, 
                     participacao_discussoes, uso_tecnologia_educacional, nivel_estresse, 
                     mapeamento_grades):
    print("Iniciando previsão de nota final...")
    
    # Mapeamentos para traduzir opções do português para inglês
    mapeamento_genero = {
        'Feminino': 'Female',
        'Masculino': 'Male',
        'Outro': 'Other'
    }
    mapeamento_estilo_aprendizado = {
        'Auditivo': 'Auditory',
        'Cinesio': 'Kinesthetic',
        'Leitura/Escrita': 'Reading/Writing',
        'Visual': 'Visual'
    }
    mapeamento_sim_nao = {
        'Sim': 'Yes',
        'Não': 'No'
    }
    mapeamento_estresse = {
        'Baixo': 'Low',
        'Médio': 'Medium',
        'Alto': 'High'
    }
    
    # Converter opções em português para inglês
    gender = mapeamento_genero[genero]
    learning_style = mapeamento_estilo_aprendizado[estilo_aprendizado]
    participation = mapeamento_sim_nao[participacao_discussoes]
    ed_tech = mapeamento_sim_nao[uso_tecnologia_educacional]
    stress_level = mapeamento_estresse[nivel_estresse]
    
    dados_entrada = {
        'Age': age,
        'Study_Hours_per_Week': study_hours,
        'Online_Courses_Completed': online_courses,
        'Assignment_Completion_Rate (%)': assignment_completion,
        'Exam_Score (%)': exam_score,
        'Attendance_Rate (%)': attendance_rate,
        'Time_Spent_on_Social_Media (hours/week)': social_media,
        'Sleep_Hours_per_Night': sleep_hours,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Other': 1 if gender == 'Other' else 0,
        'Preferred_Learning_Style_Kinesthetic': 1 if learning_style == 'Kinesthetic' else 0,
        'Preferred_Learning_Style_Reading/Writing': 1 if learning_style == 'Reading/Writing' else 0,
        'Preferred_Learning_Style_Visual': 1 if learning_style == 'Visual' else 0,
        'Participation_in_Discussions_Yes': 1 if participation == 'Yes' else 0,
        'Use_of_Educational_Tech_Yes': 1 if ed_tech == 'Yes' else 0,
        'Self_Reported_Stress_Level_Low': 1 if stress_level == 'Low' else 0,
        'Self_Reported_Stress_Level_Medium': 1 if stress_level == 'Medium' else 0
    }
    
    df_entrada = pd.DataFrame([dados_entrada])
    df_entrada[colunas_numericas] = scaler.transform(df_entrada[colunas_numericas])
    
    modelo = resultados['Random Forest']['modelo']
    predicao = modelo.predict(df_entrada)[0]
    probabilidades = modelo.predict_proba(df_entrada)[0]
    
    # Mapeamento inverso para exibir a nota
    mapeamento_inverso = {v: k for k, v in mapeamento_grades.items()}
    nota = mapeamento_inverso[predicao]
    
    # Ajustar probabilidades para refletir apenas as classes disponíveis
    prob_dict = {mapeamento_inverso[i]: probabilidades[i] for i in range(len(probabilidades))}
    
    print("Previsão de nota final concluída.")
    return (f"Nota Final Prevista: {nota}\n"
            f"Probabilidades:\n"
            f"A: {prob_dict.get('A', 0):.2%}\n"
            f"B: {prob_dict.get('B', 0):.2%}\n"
            f"C: {prob_dict.get('C', 0):.2%}\n"
            f"D: {prob_dict.get('D', 0):.2%}\n")

# Pipeline principal
def main():
    global scaler, colunas_numericas, resultados  # Tornar variáveis acessíveis globalmente
    print("=== Iniciando Pipeline de Ciência de Dados ===")
    
    # Carregar e pré-processar dados
    caminho_arquivo = 'student_performance_large_dataset.csv'
    df, df_viz, scaler, colunas_numericas, mapeamento_grades = carregar_e_preprocessar_dados(caminho_arquivo)
    
    # Gerar estatísticas descritivas
    estatisticas = estatisticas_descritivas(df)
    print("Salvando estatísticas descritivas...")
    estatisticas.to_csv('estatisticas_descritivas.csv')
    print("Estatísticas descritivas salvas em 'estatisticas_descritivas.csv'.")
    
    # Gerar visualizações
    gerar_visualizacoes(df_viz)
    
    # Preparar dados para modelagem
    print("Preparando dados para modelagem...")
    X = df.drop(['Student_ID', 'Final_Grade'], axis=1)
    y = df['Final_Grade']
    print("Dados preparados.")
    
    # Treinar e avaliar modelos
    resultados, X_train, X_test, y_train, y_test = treinar_e_avaliar_modelos(X, y)
    
    # Teste de hipóteses
    print("Executando testes de hipóteses para comparar modelos...")
    rf_scores = resultados['Random Forest']['cv_scores']
    xgb_scores = resultados['XGBoost']['cv_scores']
    svm_scores = resultados['SVM']['cv_scores']
    
    t_stat_rf_xgb, p_val_rf_xgb = ttest_rel(rf_scores, xgb_scores)
    t_stat_rf_svm, p_val_rf_svm = ttest_rel(rf_scores, svm_scores)
    print(f"  Teste t para Random Forest vs XGBoost: t={t_stat_rf_xgb:.4f}, p-valor={p_val_rf_xgb:.4f}")
    print(f"  Teste t para Random Forest vs SVM: t={t_stat_rf_svm:.4f}, p-valor={p_val_rf_svm:.4f}")
    print("Testes de hipóteses concluídos.")
    
    # Configurar interface Gradio com dropdowns traduzidos
    print("Configurando interface Gradio...")
    interface = gr.Interface(
        fn=lambda *args: prever_nota_final(*args, mapeamento_grades=mapeamento_grades),
        inputs=[
            gr.Slider(18, 29, step=1, label="Idade"),
            gr.Slider(0, 50, step=1, label="Horas de Estudo por Semana"),
            gr.Slider(0, 20, step=1, label="Cursos Online Concluídos"),
            gr.Slider(0, 100, step=1, label="Taxa de Conclusão de Tarefas (%)"),
            gr.Slider(0, 100, step=1, label="Pontuação em Exames (%)"),
            gr.Slider(0, 100, step=1, label="Taxa de Presença (%)"),
            gr.Slider(0, 30, step=1, label="Tempo em Redes Sociais (horas/semana)"),
            gr.Slider(0, 10, step=1, label="Horas de Sono por Noite"),
            gr.Dropdown(['Feminino', 'Masculino', 'Outro'], label="Gênero"),
            gr.Dropdown(['Auditivo', 'Cinesio', 'Leitura/Escrita', 'Visual'], label="Estilo de Aprendizado Preferido"),
            gr.Dropdown(['Sim', 'Não'], label="Participação em Discussões"),
            gr.Dropdown(['Sim', 'Não'], label="Uso de Tecnologia Educacional"),
            gr.Dropdown(['Baixo', 'Médio', 'Alto'], label="Nível de Estresse Autorreportado")
        ],
        outputs="text",
        title="Previsão de Desempenho Acadêmico"
    )
    print("Interface Gradio configurada.")
    
    print("=== Pipeline de Ciência de Dados Concluído ===")
    # Lançar interface (comentar para submissão)
    # interface.launch()

if __name__ == "__main__":
    main()