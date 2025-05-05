# 📦 Manipulação de Dados
import pandas as pd
import numpy as np

# 📈 Visualização
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

# 🧠 Machine Learning
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, recall_score, mean_absolute_error,
    mean_squared_error, r2_score
)

# 🖥️ Interface Web
import streamlit as st

# ⚙️ Outros
import os
import joblib
import pickle

# Configurações da página
st.set_page_config(
    page_title="Análise de Dados Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar dados
@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/atlantico-academy/datasets/main/diabetes.csv"
    return pd.read_csv(url)

# Carregar os dados
diabetes = carregar_dados()

# Treinar e salvar os LabelEncoders, se ainda não existirem
if not os.path.exists('le_gender.pkl') or not os.path.exists('le_smoking.pkl'):
    le_gender = LabelEncoder()
    le_gender.fit(diabetes['gender'])
    joblib.dump(le_gender, 'le_gender.pkl')

    le_smoking = LabelEncoder()
    le_smoking.fit(diabetes['smoking_history'])
    joblib.dump(le_smoking, 'le_smoking.pkl')

# Carregar os LabelEncoders
le_gender = joblib.load('le_gender.pkl')
le_smoking = joblib.load('le_smoking.pkl')

# Aplicar Label Encoding nos dados
diabetes['gender'] = le_gender.transform(diabetes['gender'])
diabetes['smoking_history'] = le_smoking.transform(diabetes['smoking_history'])
diabetes_numerico = diabetes.select_dtypes(include=['int64', 'float64'])

# Definir as colunas de características
feature_columns = ['gender', 'age', 'hypertension', 'heart_disease',
                   'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Salvar as colunas de características, se ainda não existirem
if not os.path.exists('feature_columns.pkl'):
    joblib.dump(feature_columns, 'feature_columns.pkl')

# Carregar o modelo final
with open('le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)

# Título do aplicativo
st.title("🔎 Projeto Final - Bootcamp Ciência de Dados Avanti")
st.caption("Análise Exploratória e Comparativa utilizando o dataset de Diabetes")

# Criar abas - AGORA COM 4 ABAS EXPLÍCITAS
tab1, tab2, tab3, tab4 = st.tabs([ 
    "🎯 Sobre o Projeto", 
    "📚 Análise Exploratória", 
    "🧠 Análise Comparativa de dados",
    "🧪 Avaliação Pessoal de Risco"
    
])

# Função para gerar download de gráficos
def gerar_download(fig):
    # Gerar imagem do gráfico
    img = pio.to_image(fig, format='png', width=800, height=600)
    st.download_button(
        label="Baixar Gráfico",
        data=img,
        file_name="grafico.png",
        mime="image/png"
    )

# Tab de Apresentação do Projeto
with tab1:
    # Cabeçalho com emoji
    #st.header("🔎 Projeto de Previsão de Diabetes")
    
    # Introdução em container destacado
    with st.container(border=True):
        st.write("""
        ** 🖥️  Aplicativo educativo** desenvolvido para o Bootcamp de Ciência de Dados da Avanti, 
        com objetivo de prever diabetes através de modelos preditivos.
        """)
    
    # Divisão em colunas para organização
    col1, col2 = st.columns(2)
    
    with col1:
        # Seção 1: Partes do Projeto
        st.subheader("📋 Estrutura do Projeto")
        st.markdown("""
        - **🔍 Parte 1:** Análise Exploratória de Dados
        - **📊 Parte 2:** Análise Comparativa de Modelos
        """)
        
        # Seção 2: Equipe
        st.subheader("👥 Equipe")
        st.markdown("""
        - Time inicial: 6 pessoas
        - Time final: 4 colaboradores
        - Encontros semanais às terças, 18h
        """)
        
        # Seção 5: Aprendizados 
        st.subheader("🔑 Aprendizados Chave:")
    st.markdown("""
<div style='text-align: justify'>
<ul>
  <li><b>Domínio do Ciclo de Dados:</b> A experiência prática em análise exploratória e comparativa evidenciou o impacto da preparação e interpretação de dados em decisões concretas, aproximando-me da atuação profissional.</li>
  <li><b>Rigidez Metodológica:</b> A implementação de pipelines e validação cruzada consolidou a importância da consistência e confiabilidade no processo de modelagem.</li>
  <li><b>Seleção Estratégica de Métricas:</b> Desenvolvi a capacidade de escolher métricas adequadas para contextos sensíveis, como a área da saúde, garantindo avaliações mais precisas e éticas.</li>
  <li><b>Comparação Robusta de Modelos:</b> Aprimorei a segurança na comparação de modelos, fundamentando as escolhas em critérios técnicos sólidos.</li>
  <li><b>Valor da Colaboração:</b> O trabalho em equipe demonstrou ser fundamental para a superação de desafios complexos.</li>
  <li><b>Narrativa dos Dados:</b> Concluí que dados possuem o poder de contar histórias significativas, e a habilidade de interpretá-las constitui um diferencial crucial.</li>
</ul>
</div>
""", unsafe_allow_html=True)
    
    with col2:
        # Seção 4: Metodologia
        st.subheader("📌 Metodologia CRISP-DM")
        st.markdown("""
        1. **Entendimento dos dados**  
           ↳ Coleta + Análise Exploratória  
        2. **Preparação**  
           ↳ Tratamento de dados faltantes  
           ↳ Codificação/Normalização  
        3. **Modelagem**  
           ↳ Treinamento + Comparação de modelos
        """)
            
    # Seção 3: Tecnologias
    st.subheader("🛠 Tecnologias")
    st.markdown("""
        ```python
        Python (Pandas, Scikit-learn)
        Jupyter Notebook
        Streamlit
        ```
        """)
    
    # Seção de agradecimento
    st.divider()
    st.markdown("""
    *Agradeço à [Avanti](https://atlanticoavanti.com.br) pela oportunidade e a todos os colegas que colaboraram neste projeto!*  
    *Desenvolvido por Alexsandra Tavares* 🚀
    """)
    
    # Imagem com ajuste de largura
    st.image("imagens/tipos-de-analise-de-dados.jpg",
             caption="Exemplo de Tipos de Análise de Dados",
             use_container_width=True)  # Ajuste o valor conforme necessidade ou adaptar a responsividade de tela

# Tab de Análise Exploratória
with tab2:
    st.header("Análise Exploratória")
    
    # Estatísticas básicas
    st.subheader("Estatísticas Descritivas")
    st.write(diabetes.describe())
    
    # Visualizações
    st.subheader("Distribuição das Variáveis")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_col = st.selectbox("Selecione a variável:", diabetes.columns)
        fig, ax = plt.subplots()
        sns.histplot(diabetes[selected_col], kde=True, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Correlação entre Variáveis")
        corr = diabetes_numerico.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Matriz de Correlação entre Variáveis Numéricas")
        plt.tight_layout()
        st.pyplot(plt)

# Tab Análise Comparativa de dados (Modelagem e Avaliação)    
with tab3:
    st.header("🧠 Modelagem e Avaliação")
    st.write("Utilizamos o algoritmo **XGBoost Classifier** para prever a presença de diabetes com base nas características fornecidas.")
    
    st.markdown("""
    **Principais variáveis utilizadas no modelo:**
    - Gênero
    - Idade
    - Histórico de Tabagismo
    - Hipertensão
    - Doença Cardíaca
    - IMC (Índice de Massa Corporal)
    - Hemoglobina Glicada (HbA1c)
    - Nível de Glicose no Sangue

    O modelo foi treinado com uma divisão de 80% dos dados para treino e 20% para teste, garantindo uma avaliação justa e precisa.
    """)

    st.success("🔍 Observação: O XGBoost demonstrou ótimo desempenho em problemas de classificação tabular, como este.")

# Tab Avaliação Pessoal de Risco
with tab4:
    st.subheader("🎯 Predição Personalizada de Risco de Diabetes")
    
    # Carregamentos
    modelo_final = joblib.load("modelo_final.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    le_gender = joblib.load("le_gender.pkl")
    le_smoking = joblib.load("le_smoking.pkl")

    st.title("Analisador de Risco de Diabetes")

    # Interface de entrada de dados
    st.header("Preencha os dados abaixo:")

    # Entradas do usuário
    col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gênero:", le_gender.classes_)
    age = st.number_input("Idade:", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Hipertensão:", ["0", "1"])
    heart_disease = st.selectbox("Doença Cardíaca:", ["0", "1"])

with col2:
    smoking_history = st.selectbox("Histórico de Tabagismo:", le_smoking.classes_)
    bmi = st.number_input("IMC:", min_value=10.0, max_value=60.0, value=25.0)
    hba1c_level = st.number_input("HbA1c (%):", min_value=3.0, max_value=15.0, value=5.5)
    blood_glucose_level = st.number_input("Glicose no sangue (mg/dL):", min_value=50, max_value=500, value=100)

        # Botão de predição
if st.button("Analisar Risco"):
    try:
        # Transforma os dados categóricos com LabelEncoder
        gender_encoded = le_gender.transform([gender])[0]
        smoking_encoded = le_smoking.transform([smoking_history])[0]

        # Monta o DataFrame de entrada
        input_data = {
            'gender': gender_encoded,
            'age': age,
            'hypertension': int(hypertension),
            'heart_disease': int(heart_disease),
            'smoking_history': smoking_encoded,
            'bmi': bmi,
            'HbA1c_level': hba1c_level,
            'blood_glucose_level': blood_glucose_level
        }

        input_df = pd.DataFrame([input_data], columns=feature_columns)

        # Predição
        prediction = modelo_final.predict(input_df)[0]
        prob = modelo_final.predict_proba(input_df)[0][1]

        # Exibe resultado
        st.subheader("Resultado da Análise:")
        if prediction == 1:
            st.error(f"⚠️ Alto risco de diabetes. Probabilidade: {prob:.2%}")
        else:
            st.success(f"✅ Baixo risco de diabetes. Probabilidade: {prob:.2%}")

    except Exception as e:
        st.error(f"Ocorreu um erro na análise: {str(e)}")
