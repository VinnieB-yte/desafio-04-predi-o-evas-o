# Desafio 04 — Predição de Risco de Evasão Escolar

## 1. Sobre o projeto

Este projeto utiliza técnicas de Machine Learning para estimar o risco de evasão escolar no ensino superior, com base em informações acadêmicas, pessoais e socioeconômicas dos estudantes.

O desenvolvimento utiliza a base de dados Student Dropout and Academic Success, disponibilizada pelo UCI Machine Learning Repository.

O modelo selecionado foi a Regressão Logística, treinada e avaliada para identificar estudantes que apresentam características associadas à evasão.

A previsão representa uma estimativa estatística e não uma certeza sobre o futuro do estudante.

## 2. Objetivo

Desenvolver uma solução capaz de:

- Receber informações de estudantes por meio de uma interface.
- Processar os dados utilizando um modelo de Machine Learning já treinado.
- Estimar a probabilidade de evasão.
- Apresentar a classificação do risco.
- Registrar automaticamente as previsões realizadas.

## 3. Tecnologias utilizadas

- Python: linguagem de programação.
- Google Colab: ambiente de desenvolvimento.
- Pandas e NumPy: manipulação e preparação dos dados.
- Scikit-learn: treinamento, preparação e avaliação do modelo.
- Matplotlib e Seaborn: visualização de dados.
- Joblib: armazenamento e carregamento do modelo.
- Streamlit: desenvolvimento da interface.
- FastAPI: criação da API de inferência.
- Uvicorn: servidor responsável por executar a API.
- Requests: comunicação HTTP entre Streamlit e API.
- Pydantic: validação dos dados recebidos pela API.
- Google Drive: armazenamento dos arquivos do projeto.
- GitHub: controle de versão e disponibilização do código.

## 4. Estrutura do projeto

```text
Desafio_04_Machine_Learning_Evasao/
│
├── api.py
├── app.py
├── app_backup_antes_api.py
├── modelo_regressao_logistica.joblib
├── requirements.txt
├── README.md
│
└── resultados/
    └── monitoramento.csv
```

O arquivo do modelo contém o pipeline de Regressão Logística treinado e os componentes necessários para preparar os dados antes da previsão.

O arquivo `api.py` disponibiliza os serviços de inferência e registra as previsões realizadas.

O arquivo `app.py` implementa a interface utilizada para inserir os dados dos estudantes e visualizar os resultados.

O arquivo `monitoramento.csv` armazena o histórico das previsões registradas.

## 5. Arquitetura da aplicação

O funcionamento ocorre por meio da integração entre os seguintes componentes:

1. Streamlit: recebe os dados preenchidos pelo usuário.
2. FastAPI: recebe a solicitação enviada pela interface.
3. Modelo de Machine Learning: processa os dados e calcula a probabilidade de evasão.
4. FastAPI: devolve a classificação e a probabilidade à interface.
5. Monitoramento: registra a previsão, sua data e hora e o tempo de execução.

## 6. Instalação das dependências

Com o Python instalado e o terminal aberto na pasta do projeto, execute:

```bash
pip install -r requirements.txt
```

## 7. Como executar a aplicação

A API e o Streamlit devem ser iniciados em dois terminais separados.

### 7.1. Iniciar a API

No primeiro terminal, execute:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

A API ficará disponível em:

- Serviço principal: http://127.0.0.1:8000
- Documentação interativa: http://127.0.0.1:8000/docs
- Verificação de saúde: http://127.0.0.1:8000/saude

### 7.2. Iniciar o Streamlit

No segundo terminal, execute:

```bash
streamlit run app.py
```

A interface poderá ser acessada pelo endereço local apresentado pelo Streamlit, normalmente:

http://localhost:8501

É necessário manter a API em execução para que a interface consiga solicitar as previsões.

## 8. Endpoints da API

| Método | Endpoint | Função |
|---|---|---|
| GET | / | Apresenta informações básicas da API |
| GET | /saude | Verifica se a API e o modelo estão disponíveis |
| GET | /opcoes | Retorna as opções dos campos categóricos |
| POST | /prever | Recebe os dados e realiza a previsão |

A documentação interativa dos endpoints está disponível em `/docs`.

## 9. Monitoramento

Cada previsão realizada pela API é registrada automaticamente no arquivo:

`resultados/monitoramento.csv`

São armazenadas informações como:

- Data e hora da previsão.
- Probabilidade estimada de evasão.
- Classificação do resultado.
- Tempo de execução do modelo.

O monitoramento permite acompanhar o histórico de utilização do serviço.

## 10. Infraestrutura e limitações

O desenvolvimento e os testes foram realizados no Google Colab, utilizando o Google Drive para armazenar os arquivos do projeto.

A interface Streamlit e a API FastAPI foram executadas simultaneamente no ambiente de desenvolvimento. O Cloudflare Tunnel foi utilizado para permitir o acesso externo temporário à interface.

A aplicação depende da disponibilidade do ambiente de execução. Portanto, a interrupção da sessão do Google Colab encerra os serviços executados nela.

O projeto não representa uma implantação permanente em nuvem nem utiliza, nesta versão, um serviço gerenciado de inferência.

## 11. Considerações finais

O projeto demonstra a aplicação de técnicas de Machine Learning em um cenário educacional, integrando um modelo de classificação a uma interface e a uma API de inferência.

A solução permite estimar o risco de evasão e registrar as previsões realizadas, podendo servir como ferramenta de apoio ao acompanhamento acadêmico e à tomada de decisões pelas instituições de ensino.
