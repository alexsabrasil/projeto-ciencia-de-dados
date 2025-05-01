# 🧪 Projeto de Ciência de Dados Avanti 2025: Predição de Diabetes

Este projeto tem como objetivo desenvolver uma aplicação interativa que utiliza um modelo de Machine Learning para prever a probabilidade de uma pessoa desenvolver diabetes, com base em variáveis clínicas. A aplicação foi construída utilizando Python, Pandas, Scikit-learn e Streamlit, com um modelo treinado em um conjunto de dados de diabetes.

O projeto envolve a exploração dos dados, pré-processamento, treinamento do modelo, e a criação de uma interface interativa utilizando o Streamlit. O modelo preditivo foi otimizado e avaliado utilizando métricas como acurácia, precisão, recall, f1-score, entre outras.

---

## 📊 Tecnologias Utilizadas

- Python 3.x
- Pandas (para manipulação e análise de dados)
- NumPy (para operações numéricas)
- Scikit-learn (para treinamento e avaliação do modelo de Machine Learning)
- Matplotlib / Seaborn (para visualização de dados)
- Pickle (para salvar o modelo treinado)
- Streamlit (para criar a aplicação interativa)
- XGBoost
- Plotly
- Statmodels

---

## 🔍 O que foi feito

Etapas do Projeto:

1. Análise Exploratória dos Dados (EDA):
   
    - Carregamento e visualização dos dados.
    - Análise de variáveis, distribuição e correlações.
  
2. Pré-processamento dos Dados:
   
    - Tratamento de valores ausentes.
    - Codificação de variáveis categóricas.
    - Escalonamento de variáveis.
  
3. Construção do Modelo de Machine Learning:
   
    - Seleção do modelo (ex: regressão logística, árvore de decisão, etc.).
    - Treinamento e validação do modelo.
    - Avaliação do desempenho utilizando métricas adequadas.

4. Criação da Aplicação Interativa:

    - Interface em Streamlit para permitir a interação do usuário.
    - Carregamento do modelo treinado e uso para fazer previsões com base nas entradas do usuário.

---

## 💡 O que eu aprendi

Durante o desenvolvimento deste projeto, aprendi diversas técnicas importantes em ciência de dados e Machine Learning, como:

  - Pré-processamento de dados: como limpar, preparar e transformar dados brutos em um formato adequado para o modelo.
  - Construção de modelos preditivos: como escolher, treinar e avaliar modelos de machine learning.
  - Implantação de soluções: como criar uma aplicação interativa com o Streamlit, integrando o modelo treinado para realizar previsões em tempo real.
  - Validação de modelos: como usar métricas de desempenho (precisão, recall, F1-score) para avaliar a qualidade do modelo.

---

## 🌐 Link da Aplicação Web

👉 [Acesse o aplicativo no Streamlit](https://alexsandratss.streamlit.app/)

Experimente o modelo de predição de diabetes!  
Aqui você pode inserir dados clínicos e verificar a probabilidade de desenvolvimento de diabetes.

---

## 🚀 Como Executar Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/alexsabrasil/projeto-ciencia-de-dados.git

# 2. Acesse o diretório do projeto
cd projeto-ciencia-de-dados

# 3. Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

# 4. Instale as dependências
pip install -r requirements.txt

# 5. Rode a aplicação
streamlit run app.py


--- 

## 📚 Dataset Utilizado

O dataset utilizado é de domínio público e contém atributos clínicos relacionados à saúde dos pacientes, como idade, IMC, glicose, entre outros.

--- 

🙏 Agradecimentos

Agradeço à [Avanti](https://atlanticoavanti.com.br) pela oportunidade e a todos os colegas que colaboraram neste projeto!*  
Desenvolvido por Alexsandra Tavares 🚀

---

📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
