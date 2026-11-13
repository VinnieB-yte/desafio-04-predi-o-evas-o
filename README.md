# Projeto de Machine Learning — Previsão de Risco de Evasão Escolar

## 1. Sobre o projeto

Este projeto foi desenvolvido como parte do Desafio 04 e tem como objetivo utilizar técnicas de Machine Learning para identificar estudantes com possível risco de evasão escolar.

A solução utiliza um modelo de classificação treinado com dados do conjunto **Student Dropout and Academic Success**, disponibilizado no UCI Machine Learning Repository.

O modelo estima a probabilidade de evasão e apresenta uma classificação de risco, auxiliando no acompanhamento dos estudantes. A previsão representa uma estimativa, não uma certeza de que o estudante abandonará o curso.

## 2. Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Google Colab
- Google Drive
- Git e GitHub

## 3. Modelo de Machine Learning

Foram avaliados três algoritmos de classificação:

- Regressão Logística;
- Árvore de Decisão;
- Random Forest.

Após a comparação dos resultados e a otimização dos parâmetros, a **Regressão Logística** foi selecionada como modelo principal.

O modelo treinado foi salvo no formato `.joblib`, permitindo sua utilização na aplicação sem a necessidade de executar novamente o treinamento.

## 4. Aplicação

A interface foi desenvolvida com Streamlit. O usuário informa características acadêmicas e socioeconômicas do estudante, e a aplicação utiliza o modelo salvo para calcular a probabilidade estimada de evasão.

A aplicação também possui um mecanismo de monitoramento que registra informações das previsões realizadas, como:

- Data e horário;
- Probabilidade estimada de evasão;
- Classificação apresentada;
- Tempo de execução da previsão.

## 5. Estrutura do projeto

```text
desafio-04-predicao-evasao/
│
├── app.py
├── modelo_regressao_logistica.joblib
├── requirements.txt
├── README.md
│
├── dados/
├── documentacao/
├── evidencias/
├── modelos/
├── notebooks/
└── resultados/
```

As pastas organizam os dados, documentos, evidências, modelos, notebooks e resultados do projeto. Nem todas precisam estar presentes na versão publicada no GitHub.

## 6. Como executar a aplicação

Instale as dependências:

```bash
pip install -r requirements.txt
```

Em seguida, execute:

```bash
streamlit run app.py
```

O arquivo `modelo_regressao_logistica.joblib` deve estar disponível no local esperado pelo código da aplicação.

## 7. Monitoramento

A aplicação possui um mecanismo básico de monitoramento que registra as previsões realizadas em um arquivo CSV.

Esses registros permitem acompanhar a quantidade de previsões, as classificações apresentadas, as probabilidades estimadas e o tempo de execução.

## 8. Objetivo acadêmico

O projeto demonstra a aplicação de conceitos de Machine Learning e serviços de desenvolvimento e armazenamento em nuvem em um cenário educacional, integrando treinamento de modelos, disponibilização de inferência e monitoramento básico.
