import streamlit as st
import pandas as pd
import joblib
import csv
import time
from pathlib import Path
from datetime import datetime

# Configuração da página

CAMINHO_MONITORAMENTO = (
    Path(__file__).resolve().parent
    / "resultados"
    / "monitoramento.csv"
)


def registrar_monitoramento(probabilidade, classificacao, tempo_execucao):
    """Registra informações básicas de cada previsão."""

    CAMINHO_MONITORAMENTO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    arquivo_existe = CAMINHO_MONITORAMENTO.exists()

    with open(
        CAMINHO_MONITORAMENTO,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(arquivo)

        if not arquivo_existe:
            escritor.writerow([
                "data_hora",
                "probabilidade_evasao",
                "classificacao",
                "tempo_execucao_segundos"
            ])

        escritor.writerow([
            datetime.now().astimezone().isoformat(timespec="seconds"),
            round(float(probabilidade), 4),
            classificacao,
            round(float(tempo_execucao), 4)
        ])


st.set_page_config(
    page_title="Predição de Evasão Escolar",
    page_icon="🎓",
    layout="wide"
)

# Carregamento do modelo
CAMINHO_MODELO = (
    Path(__file__).resolve().parent
    / "modelo_regressao_logistica.joblib"
)

@st.cache_resource
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)

try:
    modelo = carregar_modelo()
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo: {erro}")
    st.stop()

# Recuperação das categorias aprendidas pelo modelo
preparador = modelo.named_steps["preparacao"]
codificador = preparador.named_transformers_["categoricas"]

colunas_categoricas = [
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Gender",
    "International"
]

colunas_numericas = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)"
]

# Interface
st.title("🎓 Predição de Risco de Evasão Escolar")

st.write(
    "Ferramenta demonstrativa de Machine Learning "
    "para estimar o risco de evasão de estudantes "
    "do ensino superior."
)

st.info(
    "Preencha os dados acadêmicos e cadastrais "
    "para obter uma estimativa estatística."
)

st.divider()

st.subheader("1. Informações do estudante")

dados = {}

with st.form("formulario_predicao"):

    st.markdown("### Dados categóricos")

    col1, col2, col3 = st.columns(3)

    for i, coluna in enumerate(colunas_categoricas):
        opcoes = codificador.categories_[i].tolist()

        with [col1, col2, col3][i % 3]:
            dados[coluna] = st.selectbox(
                coluna,
                options=opcoes,
                key=f"cat_{i}"
            )

    st.divider()
    st.markdown("### Dados numéricos")

    configuracoes = {
        "Previous qualification (grade)": (95.0, 190.0, 133.1, 0.1),
        "Admission grade": (95.0, 190.0, 126.0, 0.1),
        "Age at enrollment": (17, 62, 20, 1),
        "Curricular units 1st sem (credited)": (0, 20, 0, 1),
        "Curricular units 1st sem (enrolled)": (0, 26, 6, 1),
        "Curricular units 1st sem (evaluations)": (0, 45, 8, 1),
        "Curricular units 1st sem (approved)": (0, 26, 5, 1),
        "Curricular units 1st sem (grade)": (0.0, 18.0, 12.33, 0.1),
        "Curricular units 1st sem (without evaluations)": (0, 12, 0, 1)
    }

    col1, col2, col3 = st.columns(3)

    for i, coluna in enumerate(colunas_numericas):
        minimo, maximo, padrao, passo = configuracoes[coluna]

        with [col1, col2, col3][i % 3]:
            dados[coluna] = st.number_input(
                coluna,
                min_value=minimo,
                max_value=maximo,
                value=padrao,
                step=passo,
                key=f"num_{i}"
            )

    st.divider()

    enviar = st.form_submit_button(
        "Calcular risco de evasão",
        use_container_width=True
    )

# Predição
if enviar:

    entrada = pd.DataFrame(
        [dados],
        columns=colunas_categoricas + colunas_numericas
    )

    try:
        inicio_execucao = time.perf_counter()
        predicao = modelo.predict(entrada)[0]
        probabilidades = modelo.predict_proba(entrada)[0]
        tempo_execucao = time.perf_counter() - inicio_execucao

        indice_dropout = list(modelo.classes_).index(1)
        probabilidade = probabilidades[indice_dropout]

        classificacao = (
            "Risco de evasão"
            if int(predicao) == 1
            else "Sem classificação de risco de evasão"
        )

        registrar_monitoramento(
            probabilidade,
            classificacao,
            tempo_execucao
        )


        st.divider()
        st.subheader("2. Resultado da análise")

        st.metric(
            "Probabilidade estimada de evasão",
            f"{probabilidade * 100:.2f}%"
        )

        st.progress(float(probabilidade))

        if predicao == 1:
            st.warning(
                "O modelo identificou sinais associados "
                "a um maior risco de evasão."
            )
        else:
            st.success(
                "O modelo não classificou o estudante "
                "como provável caso de evasão."
            )

        st.caption(
            "A classificação representa uma estimativa "
            "estatística, não uma certeza sobre o futuro "
            "do estudante."
        )

    except Exception as erro:
        st.error(f"Erro durante a predição: {erro}")

