
import csv
import time
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# =====================================================
# 1. CAMINHOS DOS ARQUIVOS
# =====================================================

PASTA_PROJETO = Path(__file__).resolve().parent

CAMINHO_MODELO = (
    PASTA_PROJETO / "modelo_regressao_logistica.joblib"
)

CAMINHO_MONITORAMENTO = (
    PASTA_PROJETO / "resultados" / "monitoramento.csv"
)


# =====================================================
# 2. CARREGAMENTO DO MODELO
# =====================================================

if not CAMINHO_MODELO.exists():
    raise FileNotFoundError(
        f"Modelo não encontrado: {CAMINHO_MODELO}"
    )

modelo = joblib.load(CAMINHO_MODELO)


# =====================================================
# 3. CAMPOS RECEBIDOS PELA API
# =====================================================

class DadosEstudante(BaseModel):

    # Variáveis categóricas
    marital_status: int
    application_mode: int
    application_order: int
    course: int
    daytime_evening_attendance: int
    previous_qualification: int
    nacionality: int
    mother_qualification: int
    father_qualification: int
    mother_occupation: int
    father_occupation: int
    displaced: int
    educational_special_needs: int
    gender: int
    international: int

    # Variáveis numéricas
    previous_qualification_grade: float
    admission_grade: float
    age_at_enrollment: float
    curricular_units_1st_sem_credited: float
    curricular_units_1st_sem_enrolled: float
    curricular_units_1st_sem_evaluations: float
    curricular_units_1st_sem_approved: float
    curricular_units_1st_sem_grade: float
    curricular_units_1st_sem_without_evaluations: float


# =====================================================
# 4. ORDEM ORIGINAL DAS VARIÁVEIS DO MODELO
# =====================================================

MAPA_COLUNAS = {
    "marital_status": "Marital Status",
    "application_mode": "Application mode",
    "application_order": "Application order",
    "course": "Course",
    "daytime_evening_attendance": "Daytime/evening attendance",
    "previous_qualification": "Previous qualification",
    "nacionality": "Nacionality",
    "mother_qualification": "Mother's qualification",
    "father_qualification": "Father's qualification",
    "mother_occupation": "Mother's occupation",
    "father_occupation": "Father's occupation",
    "displaced": "Displaced",
    "educational_special_needs": "Educational special needs",
    "gender": "Gender",
    "international": "International",
    "previous_qualification_grade": "Previous qualification (grade)",
    "admission_grade": "Admission grade",
    "age_at_enrollment": "Age at enrollment",
    "curricular_units_1st_sem_credited": "Curricular units 1st sem (credited)",
    "curricular_units_1st_sem_enrolled": "Curricular units 1st sem (enrolled)",
    "curricular_units_1st_sem_evaluations": "Curricular units 1st sem (evaluations)",
    "curricular_units_1st_sem_approved": "Curricular units 1st sem (approved)",
    "curricular_units_1st_sem_grade": "Curricular units 1st sem (grade)",
    "curricular_units_1st_sem_without_evaluations": "Curricular units 1st sem (without evaluations)"
}


COLUNAS_MODELO = list(MAPA_COLUNAS.values())


# =====================================================
# 5. REGISTRO DO MONITORAMENTO
# =====================================================

def registrar_monitoramento(
    probabilidade,
    classificacao,
    tempo_execucao
):

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
            datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            round(float(probabilidade), 4),
            classificacao,
            round(float(tempo_execucao), 4)
        ])


# =====================================================
# 6. CRIAÇÃO DA API
# =====================================================

app = FastAPI(
    title="API de Predição de Evasão Escolar",
    description=(
        "Serviço de inferência utilizando "
        "um modelo de Regressão Logística."
    ),
    version="1.0.0"
)


@app.get("/")
def inicio():
    return {
        "mensagem": "API de Predição de Evasão Escolar funcionando!"
    }


@app.get("/saude")
def verificar_saude():
    return {
        "status": "online",
        "modelo_carregado": True
    }


# =====================================================
# 7. ENDPOINT DE PREVISÃO
# =====================================================

@app.post("/prever")
def prever(dados: DadosEstudante):

    try:

        inicio_execucao = time.perf_counter()

        # Converter os dados recebidos para DataFrame
        dados_dict = dados.model_dump()

        entrada = pd.DataFrame(
            [{
                MAPA_COLUNAS[chave]: valor
                for chave, valor in dados_dict.items()
            }],
            columns=COLUNAS_MODELO
        )

        # Realizar a previsão
        predicao = modelo.predict(entrada)[0]

        probabilidades = modelo.predict_proba(entrada)[0]

        indice_dropout = list(modelo.classes_).index(1)

        probabilidade = probabilidades[indice_dropout]

        tempo_execucao = (
            time.perf_counter() - inicio_execucao
        )

        classificacao = (
            "Risco de evasão"
            if int(predicao) == 1
            else "Sem classificação de risco de evasão"
        )

        # Registrar previsão
        registrar_monitoramento(
            probabilidade,
            classificacao,
            tempo_execucao
        )

        return {
            "predicao": int(predicao),
            "classificacao": classificacao,
            "probabilidade_evasao": round(
                float(probabilidade), 4
            ),
            "tempo_execucao_segundos": round(
                float(tempo_execucao), 4
            )
        }

    except Exception as erro:

        raise HTTPException(
            status_code=500,
            detail=f"Erro durante a previsão: {str(erro)}"
        )


# =====================================================
# 8. ENDPOINT PARA FORNECER AS OPÇÕES DO FORMULÁRIO
# =====================================================

@app.get("/opcoes")
def obter_opcoes():

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

    return {
        coluna: [
            valor.item() if hasattr(valor, "item") else valor
            for valor in codificador.categories_[i]
        ]
        for i, coluna in enumerate(colunas_categoricas)
    }
