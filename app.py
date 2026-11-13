import streamlit as st
import requests
import time

# =====================================================
# 1. CONFIGURAÇÕES
# =====================================================

URL_API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Predição de Evasão Escolar",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# 2. COMUNICAÇÃO COM A API
# =====================================================

@st.cache_data
def carregar_opcoes():
    resposta = requests.get(
        f"{URL_API}/opcoes",
        timeout=10
    )
    resposta.raise_for_status()
    return resposta.json()


try:
    opcoes_api = carregar_opcoes()

except requests.exceptions.RequestException as erro:
    st.error(
        "Não foi possível acessar a API. "
        "Verifique se o servidor está em execução."
    )
    st.code(str(erro))
    st.stop()


# =====================================================
# 3. CAMPOS DO FORMULÁRIO
# =====================================================

campos_categoricos = {
    "Marital Status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance": "daytime_evening_attendance",
    "Previous qualification": "previous_qualification",
    "Nacionality": "nacionality",
    "Mother's qualification": "mother_qualification",
    "Father's qualification": "father_qualification",
    "Mother's occupation": "mother_occupation",
    "Father's occupation": "father_occupation",
    "Displaced": "displaced",
    "Educational special needs": "educational_special_needs",
    "Gender": "gender",
    "International": "international"
}


configuracoes_numericas = {
    "Previous qualification (grade)": (
        "previous_qualification_grade", 95.0, 190.0, 133.1, 0.1
    ),
    "Admission grade": (
        "admission_grade", 95.0, 190.0, 126.0, 0.1
    ),
    "Age at enrollment": (
        "age_at_enrollment", 17, 62, 20, 1
    ),
    "Curricular units 1st sem (credited)": (
        "curricular_units_1st_sem_credited", 0, 20, 0, 1
    ),
    "Curricular units 1st sem (enrolled)": (
        "curricular_units_1st_sem_enrolled", 0, 26, 6, 1
    ),
    "Curricular units 1st sem (evaluations)": (
        "curricular_units_1st_sem_evaluations", 0, 45, 8, 1
    ),
    "Curricular units 1st sem (approved)": (
        "curricular_units_1st_sem_approved", 0, 26, 5, 1
    ),
    "Curricular units 1st sem (grade)": (
        "curricular_units_1st_sem_grade", 0.0, 18.0, 12.33, 0.1
    ),
    "Curricular units 1st sem (without evaluations)": (
        "curricular_units_1st_sem_without_evaluations", 0, 12, 0, 1
    )
}


# =====================================================
# 4. INTERFACE
# =====================================================

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

    colunas = st.columns(3)

    for i, (nome, chave_api) in enumerate(
        campos_categoricos.items()
    ):
        with colunas[i % 3]:
            dados[chave_api] = st.selectbox(
                nome,
                options=opcoes_api[nome],
                key=f"cat_{i}"
            )

    st.divider()
    st.markdown("### Dados numéricos")

    colunas = st.columns(3)

    for i, (nome, configuracao) in enumerate(
        configuracoes_numericas.items()
    ):
        chave_api, minimo, maximo, padrao, passo = configuracao

        with colunas[i % 3]:
            dados[chave_api] = st.number_input(
                nome,
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


# =====================================================
# 5. ENVIO DOS DADOS PARA A API
# =====================================================

if enviar:

    try:
        inicio = time.perf_counter()

        resposta = requests.post(
            f"{URL_API}/prever",
            json=dados,
            timeout=30
        )

        resposta.raise_for_status()

        resultado = resposta.json()

        tempo_total = time.perf_counter() - inicio

        probabilidade = resultado["probabilidade_evasao"]
        classificacao = resultado["classificacao"]
        predicao = resultado["predicao"]

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

        st.caption(
            f"Tempo de execução do modelo: "
            f"{resultado['tempo_execucao_segundos']:.4f} segundos."
        )

        st.caption(
            f"Tempo total da comunicação: {tempo_total:.4f} segundos."
        )

    except requests.exceptions.RequestException as erro:
        st.error(f"Erro ao comunicar com a API: {erro}")

    except Exception as erro:
        st.error(f"Erro durante a previsão: {erro}")
