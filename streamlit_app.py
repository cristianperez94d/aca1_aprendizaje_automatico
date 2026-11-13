"""Aplicación ACA 1: análisis y predicción de deserción estudiantil."""

import json
import re
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


# -----------------------------------------------------------------------------
# Configuración y constantes del proyecto
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de deserción estudiantil | ACA 1",
    page_icon="🎓",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DRIVE_FILE_ID = "1hibMQVZCIThMb-rM3A2J7zUBj1hwL6PH"
DRIVE_CSV_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
LOGISTIC_MODEL_PATH = APP_DIR / "modelo_logistico.pkl"
TREE_MODEL_PATH = APP_DIR / "modelo_arbol.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"
MLP_MODEL_PATH = APP_DIR / "modelo_mlp.keras"
MLP_RESULTS_PATH = APP_DIR / "mlp_resultados.json"
DB_FILENAME = "UCI_Predict_Students_Dropout.db"
DB_PATH = APP_DIR / DB_FILENAME
TABLE = "students_dropout_clean"
EDITOR_MAX_ROWS = 1000
QUERY_TIMEOUT_SECONDS = 5

FEATURE_COLUMNS = [
    "Age at enrollment",
    "Admission grade",
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

TARGET_LABELS = {0: "Dropout", 1: "Graduate"}
MODEL_FILES = {
    "Regresión Logística": LOGISTIC_MODEL_PATH,
    "Árbol de Decisión": TREE_MODEL_PATH,
}
MLP_NAME = "Red Neuronal (MLP)"
# El MLP va primero para que sea el algoritmo seleccionado por defecto
MODEL_OPTIONS = [MLP_NAME, *MODEL_FILES]
# Modelos entrenados con datos escalados: requieren scaler.pkl
SCALED_MODELS = {"Regresión Logística", MLP_NAME}

st.markdown(
    """
    <style>
      .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1rem;
        background: linear-gradient(120deg, #12304a 0%, #176b70 100%);
        color: white;
        margin-bottom: 1rem;
      }
      .hero p { margin-bottom: 0; color: #e8f4f4; }
      div[data-testid="stMetric"] {
        background: #f5f8fa;
        border: 1px solid #e3eaee;
        padding: 0.8rem 1rem;
        border-radius: 0.8rem;
                color: #12304a !important;
            }
            div[data-testid="stMetric"] label,
            div[data-testid="stMetric"] label *,
            div[data-testid="stMetricValue"],
            div[data-testid="stMetricValue"] *,
            div[data-testid="stMetricDelta"],
            div[data-testid="stMetricDelta"] *,
            div[data-testid="stMetricLabel"],
            div[data-testid="stMetricLabel"] * {
                color: #12304a !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Carga de datos y modelos: se cachean para no repetir lecturas en cada rerun
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Carga el CSV exclusivamente desde la URL configurada de Google Drive."""
    frame = pd.read_csv(DRIVE_CSV_URL, sep=";")

    frame.columns = frame.columns.astype(str).str.strip()
    if "Target" not in frame.columns:
        raise ValueError("El dataset no contiene la columna obligatoria 'Target'.")

    frame["Target"] = frame["Target"].astype("string").str.strip().str.title()
    frame = frame[frame["Target"].isin(["Dropout", "Graduate"])].copy()
    if frame.empty:
        raise ValueError(
            "No se encontraron registros Dropout o Graduate después de excluir Enrolled."
        )
    return frame


@st.cache_resource(show_spinner=False)
def load_artifacts(
    signature: tuple[tuple[str, int | None, int | None], ...],
) -> tuple[dict, object | None, dict[str, str]]:
    """Carga cada artefacto de forma independiente para informar errores precisos."""
    del signature  # Se usa como clave de caché para detectar archivos nuevos o actualizados.
    models = {}
    errors = {}

    for model_name, model_path in MODEL_FILES.items():
        try:
            models[model_name] = joblib.load(model_path)
        except Exception as exc:
            errors[model_name] = f"No se pudo cargar {model_path.name}: {exc}"

    try:
        # TensorFlow se importa aquí para que la app funcione aunque no esté instalado
        import keras

        models[MLP_NAME] = keras.models.load_model(MLP_MODEL_PATH, compile=False)
    except Exception as exc:
        errors[MLP_NAME] = f"No se pudo cargar {MLP_MODEL_PATH.name}: {exc}"

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as exc:
        scaler = None
        errors["scaler"] = f"No se pudo cargar {SCALER_PATH.name}: {exc}"

    return models, scaler, errors


@st.cache_data(show_spinner=False)
def read_mlp_results(modified_ns: int) -> dict:
    """Lee los resultados de la red; modified_ns invalida la caché si el archivo cambia."""
    del modified_ns
    return json.loads(MLP_RESULTS_PATH.read_text(encoding="utf-8"))


def load_mlp_results() -> dict | None:
    """Devuelve los resultados exportados por el notebook o None si no existen."""
    try:
        return read_mlp_results(MLP_RESULTS_PATH.stat().st_mtime_ns)
    except (OSError, ValueError):
        return None


def get_artifact_signature() -> tuple[tuple[str, int | None, int | None], ...]:
    """Genera una clave que cambia si aparece o se modifica algún artefacto."""
    paths = [*MODEL_FILES.values(), MLP_MODEL_PATH, SCALER_PATH]
    signature = []
    for path in paths:
        try:
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append((str(path), None, None))
    return tuple(signature)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Devuelve una columna numérica, ignorando valores que no se puedan convertir."""
    return pd.to_numeric(frame[column], errors="coerce")


def validate_feature_schema(estimator: object, artifact_name: str) -> None:
    """Comprueba que el artefacto corresponde al esquema de nueve entradas."""
    feature_count = getattr(estimator, "n_features_in_", None)
    if feature_count is None:
        # Los modelos de Keras exponen la forma de entrada en lugar de n_features_in_
        input_shape = getattr(estimator, "input_shape", None)
        feature_count = input_shape[-1] if input_shape else None
    if feature_count is not None and feature_count != len(FEATURE_COLUMNS):
        raise ValueError(
            f"{artifact_name} fue entrenado con {feature_count} variables; "
            f"se esperaban {len(FEATURE_COLUMNS)}. Reentrena y vuelve a exportar "
            "los artefactos con las nueve variables del formulario."
        )

    feature_names = getattr(estimator, "feature_names_in_", None)
    if feature_names is not None and list(feature_names) != FEATURE_COLUMNS:
        raise ValueError(
            f"El orden o los nombres de las variables de {artifact_name} no "
            "coinciden con el formulario. Reentrena y vuelve a exportar los artefactos."
        )


def binary_selector(label: str, key: str) -> int:
    """Presenta opciones binarias con etiquetas claras y devuelve 0 o 1."""
    return int(
        st.selectbox(
            label,
            options=[0, 1],
            format_func=lambda value: "No (0)" if value == 0 else "Sí (1)",
            key=key,
        )
    )


# -----------------------------------------------------------------------------
# SQLite: base limpia del notebook (o reconstruida desde el dataset si no existe)
# -----------------------------------------------------------------------------
def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Abre la base en modo solo lectura para que ninguna consulta pueda modificarla."""
    conn = sqlite3.connect(f"{Path(db_path).as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def database_is_valid(db_path: Path) -> bool:
    """Comprueba que el archivo tenga la tabla limpia con las columnas esperadas."""
    if not db_path.is_file():
        return False
    try:
        with closing(open_readonly(db_path)) as conn:
            columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{TABLE}")')}
            return {"Target_Binario", *FEATURE_COLUMNS} <= columns
    except sqlite3.Error:
        return False


@st.cache_resource(show_spinner=False)
def ensure_database(_frame: pd.DataFrame | None) -> tuple[str, str]:
    """Devuelve la ruta de la base SQLite y su origen; la reconstruye si falta el archivo."""
    if database_is_valid(DB_PATH):
        return str(DB_PATH), f"Archivo {DB_FILENAME} generado por el notebook"
    if _frame is None:
        raise RuntimeError(
            f"No existe {DB_FILENAME} y tampoco fue posible cargar el dataset para crearla."
        )

    clean = _frame.copy()
    clean["Target"] = clean["Target"].astype(str)
    clean["Target_Binario"] = (clean["Target"] == "Graduate").astype(int)
    rebuilt_path = Path(tempfile.gettempdir()) / DB_FILENAME
    with closing(sqlite3.connect(rebuilt_path)) as conn:
        clean.to_sql(TABLE, conn, if_exists="replace", index=False)
    return str(rebuilt_path), "Base reconstruida en tiempo de ejecución desde el dataset limpio"


@st.cache_data(show_spinner=False)
def run_query(
    db_path: str, sql: str, params: tuple = (), max_rows: int = 10_000
) -> tuple[pd.DataFrame, bool]:
    """Ejecuta una consulta de lectura con tiempo y filas limitados."""
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
    with closing(open_readonly(db_path)) as conn:
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 100_000)
        cursor = conn.execute(sql, params)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(max_rows + 1)
    return pd.DataFrame(rows[:max_rows], columns=columns), len(rows) > max_rows


def validate_read_only_sql(sql: str) -> str:
    """Acepta únicamente una sentencia SELECT o WITH."""
    statement = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", statement):
        raise ValueError("Solo se permiten consultas SELECT o WITH.")
    if ";" in statement:
        raise ValueError("Ejecuta una sola sentencia por consulta.")
    return statement


QUICK_METRICS_SQL = f"""
SELECT
    COUNT(*) AS total,
    SUM(Target_Binario) AS graduados,
    COUNT(*) - SUM(Target_Binario) AS desertores,
    ROUND(100.0 * (COUNT(*) - SUM(Target_Binario)) / COUNT(*), 2) AS tasa_desercion,
    ROUND(AVG("Age at enrollment"), 1) AS edad_promedio,
    ROUND(AVG("Curricular units 2nd sem (grade)"), 2) AS nota_promedio_2s,
    ROUND(100.0 * AVG("Scholarship holder"), 1) AS pct_becarios,
    ROUND(100.0 * AVG("Debtor"), 1) AS pct_deudores
FROM {TABLE}
""".strip()

RISK_FACTORS_SQL = f"""
SELECT 'Becario: ' || CASE WHEN "Scholarship holder" = 1 THEN 'Sí' ELSE 'No' END AS grupo,
       COUNT(*) AS estudiantes,
       ROUND(100.0 * SUM(1 - Target_Binario) / COUNT(*), 1) AS tasa_desercion
FROM {TABLE} GROUP BY 1
UNION ALL
SELECT 'Deudor: ' || CASE WHEN "Debtor" = 1 THEN 'Sí' ELSE 'No' END,
       COUNT(*),
       ROUND(100.0 * SUM(1 - Target_Binario) / COUNT(*), 1)
FROM {TABLE} GROUP BY 1
UNION ALL
SELECT 'Matrícula al día: ' || CASE WHEN "Tuition fees up to date" = 1 THEN 'Sí' ELSE 'No' END,
       COUNT(*),
       ROUND(100.0 * SUM(1 - Target_Binario) / COUNT(*), 1)
FROM {TABLE} GROUP BY 1
""".strip()

EXAMPLE_QUERIES = {
    "Deserción por grupo de edad": f"""
SELECT CASE
           WHEN "Age at enrollment" <= 20 THEN '18-20'
           WHEN "Age at enrollment" <= 23 THEN '21-23'
           WHEN "Age at enrollment" <= 26 THEN '24-26'
           WHEN "Age at enrollment" <= 30 THEN '27-30'
           ELSE '31+'
       END AS grupo_edad,
       COUNT(*) AS estudiantes,
       SUM(1 - Target_Binario) AS desertores,
       ROUND(100.0 * SUM(1 - Target_Binario) / COUNT(*), 1) AS tasa_desercion
FROM {TABLE}
GROUP BY grupo_edad
ORDER BY MIN("Age at enrollment")
""".strip(),
    "Cursos con mayor deserción (mín. 30 estudiantes)": f"""
SELECT Course AS curso,
       COUNT(*) AS estudiantes,
       ROUND(100.0 * SUM(1 - Target_Binario) / COUNT(*), 1) AS tasa_desercion,
       ROUND(AVG("Curricular units 2nd sem (grade)"), 2) AS nota_promedio_2s
FROM {TABLE}
GROUP BY Course
HAVING COUNT(*) >= 30
ORDER BY tasa_desercion DESC
LIMIT 10
""".strip(),
    "Perfil financiero y riesgo": f"""
SELECT "Debtor" AS deudor,
       "Tuition fees up to date" AS matricula_al_dia,
       "Scholarship holder" AS becario,
       COUNT(*) AS estudiantes,
       ROUND(100.0 * SUM(1 - Target_Binario) / COUNT(*), 1) AS tasa_desercion
FROM {TABLE}
GROUP BY 1, 2, 3
ORDER BY tasa_desercion DESC
""".strip(),
    "Rendimiento académico por estado final": f"""
SELECT Target AS estado_final,
       COUNT(*) AS estudiantes,
       ROUND(AVG("Curricular units 1st sem (approved)"), 2) AS aprobadas_1s,
       ROUND(AVG("Curricular units 2nd sem (approved)"), 2) AS aprobadas_2s,
       ROUND(AVG("Curricular units 2nd sem (grade)"), 2) AS nota_promedio_2s,
       ROUND(AVG("Admission grade"), 1) AS nota_admision
FROM {TABLE}
GROUP BY Target
""".strip(),
    "Nota frente al promedio de su curso (función de ventana)": f"""
SELECT Course AS curso,
       Target AS estado_final,
       "Curricular units 2nd sem (grade)" AS nota_2s,
       ROUND(AVG("Curricular units 2nd sem (grade)") OVER (PARTITION BY Course), 2) AS promedio_curso,
       RANK() OVER (
           PARTITION BY Course ORDER BY "Curricular units 2nd sem (grade)" ASC
       ) AS posicion_desde_la_nota_mas_baja
FROM {TABLE}
ORDER BY curso, posicion_desde_la_nota_mas_baja
LIMIT 100
""".strip(),
}


# -----------------------------------------------------------------------------
# Barra lateral: proyecto, fuente de datos y estado de los artefactos
# -----------------------------------------------------------------------------
st.sidebar.title("🎓 ACA Final | Aprendizaje Automático")
st.sidebar.markdown(
    "**Proyecto:** Predicción de deserción y éxito académico estudiantil  \n"
    "**Dataset:** UCI Predict Students Dropout"
)
st.sidebar.markdown(
    "**Equipo**  \n"
    "Cristian Alejandro Perez Dejoy  \n"
    "Jose Ricardo Osorio Caceres  \n"
    "Sandra Milena Gutiérrez Rodríguez"
)
st.sidebar.markdown("**Fuente de datos:** Google Drive")
st.sidebar.divider()

loaded_models, loaded_scaler, artifact_errors = load_artifacts(get_artifact_signature())
st.sidebar.subheader("Estado de artefactos")
for model_name in MODEL_OPTIONS:
    if model_name in loaded_models:
        st.sidebar.success(f"{model_name}: cargado")
    else:
        st.sidebar.warning(f"{model_name}: no disponible")
if loaded_scaler is not None:
    st.sidebar.success("scaler.pkl: cargado")
else:
    st.sidebar.warning("scaler.pkl: no disponible")

selected_model_name = st.sidebar.selectbox(
    "Algoritmo predictivo",
    options=MODEL_OPTIONS,
    index=0,
)

try:
    data = load_dataset()
    data_error = None
except Exception as exc:
    data = None
    data_error = str(exc)


# -----------------------------------------------------------------------------
# Encabezado
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🎓 Predicción de deserción y éxito académico</h1>
      <p>
        Proyecto ACA 1 — Aprendizaje Automático. Exploramos los datos del
        estudiantado y presentamos modelos de apoyo para identificar señales
        tempranas que puedan orientar acciones institucionales.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_eda, tab_metrics, tab_nn, tab_predict, tab_sql = st.tabs(
    [
        "📊 Análisis exploratorio",
        "📈 Evaluación de modelos",
        "🧠 Red neuronal",
        "🔮 Simulador predictivo",
        "🗄️ Consultas SQL",
    ]
)


# -----------------------------------------------------------------------------
# Pestaña 1: EDA con datos observados
# -----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Exploración del conjunto de datos")
    if data is None:
        st.error(
            "No fue posible cargar el dataset desde Google Drive. "
            "Verifica la conexión, el identificador del archivo y que esté "
            f"compartido para lectura. Detalle: {data_error}"
        )
    else:
        total_students = len(data)
        graduates = int((data["Target"] == "Graduate").sum())
        dropouts = int((data["Target"] == "Dropout").sum())
        graduate_pct = graduates / total_students * 100 if total_students else 0
        dropout_pct = dropouts / total_students * 100 if total_students else 0

        metric_cols = st.columns(3)
        metric_cols[0].metric("Estudiantes analizados", f"{total_students:,}")
        metric_cols[1].metric("Graduados", f"{graduate_pct:.1f}%", f"{graduates:,} estudiantes")
        metric_cols[2].metric("Desertores", f"{dropout_pct:.1f}%", f"{dropouts:,} estudiantes")
        st.caption(
            "Se excluyen los registros Enrolled, de acuerdo con el alcance del notebook. "
            "Las métricas se calculan sobre la fuente seleccionada."
        )

        st.markdown("#### Hallazgos del análisis exploratorio")
        st.markdown(
            "- **Edad de matrícula:** el notebook observa que la mayoría de los "
            "estudiantes se matricula entre los 18 y 21 años. En los grupos de mayor "
            "edad —aproximadamente desde los 23–25 años— la deserción adquiere mayor "
            "presencia relativa frente a la graduación.\n"
            "- **Nota del segundo semestre:** el análisis muestra una diferencia "
            "marcada entre los resultados finales. En el grupo de graduados, las "
            "calificaciones aparecen concentradas principalmente entre 12 y 14 puntos, "
            "mientras que en los desertores se observan notas más bajas y mayor presencia "
            "de valores cercanos a cero.\n\n"
            "Estos son patrones descriptivos observados en el conjunto analizado; no "
            "demuestran causalidad ni constituyen por sí solos una prueba de "
            "significancia estadística. Además, la nota del segundo semestre solo "
            "está disponible cuando ese periodo ya fue cursado, por lo que no representa "
            "una señal de alerta al inicio de la trayectoria académica."
        )

        age_column = "Age at enrollment"
        grade_column = "Curricular units 2nd sem (grade)"
        chart_cols = st.columns(2)

        if age_column in data.columns:
            age_data = data[[age_column, "Target"]].copy()
            age_data[age_column] = numeric_series(age_data, age_column)
            age_data = age_data.dropna(subset=[age_column])
            if not age_data.empty:
                with chart_cols[0]:
                    fig, ax = plt.subplots(figsize=(8, 4.8))
                    sns.histplot(
                        data=age_data,
                        x=age_column,
                        hue="Target",
                        multiple="stack",
                        bins=30,
                        palette="viridis",
                        ax=ax,
                    )
                    ax.set_title("Edad de matrícula según estado final")
                    ax.set_xlabel("Edad al matricularse")
                    ax.set_ylabel("Número de estudiantes")
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
        else:
            chart_cols[0].info(f"No se encontró la columna '{age_column}'.")

        if grade_column in data.columns:
            grade_data = data[[grade_column, "Target"]].copy()
            grade_data[grade_column] = numeric_series(grade_data, grade_column)
            grade_data = grade_data.dropna(subset=[grade_column])
            if not grade_data.empty:
                with chart_cols[1]:
                    fig, ax = plt.subplots(figsize=(8, 4.8))
                    sns.boxplot(
                        data=grade_data,
                        x="Target",
                        y=grade_column,
                        palette="Pastel1",
                        ax=ax,
                    )
                    ax.set_title("Nota del segundo semestre por estado final")
                    ax.set_xlabel("Estado final")
                    ax.set_ylabel("Nota promedio")
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
        else:
            chart_cols[1].info(f"No se encontró la columna '{grade_column}'.")


# -----------------------------------------------------------------------------
# Pestaña 2: métricas del notebook, sin recalcular ni inventar resultados
# -----------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Evaluación y comparación de modelos")
    st.info(
        "Estos resultados corresponden al reentrenamiento del notebook con las nueve "
        "variables predictoras usadas por el simulador, e incluyen la red neuronal (MLP). "
        "Se excluyeron los casos "
        "Enrolled y se conservó la partición estratificada 80/20 con `random_state=42`. "
        "Las métricas de esta pestaña son fijas y no dependen de las entradas del usuario."
    )

    def pct_es(value: float) -> str:
        """Formatea una proporción como porcentaje con coma decimal."""
        return f"{value * 100:.2f} %".replace(".", ",")

    model_results = {
        "Árbol de Decisión": {"train": 0.8991, "test": 0.8802, "matrix": [[228, 56], [31, 411]]},
        "Regresión Logística": {"train": 0.8860, "test": 0.8857, "matrix": [[238, 46], [37, 405]]},
    }
    mlp_eval = load_mlp_results()
    if mlp_eval is not None:
        model_results[MLP_NAME] = {
            "train": mlp_eval["exactitud"]["train"],
            "test": mlp_eval["exactitud"]["test"],
            "matrix": mlp_eval["matriz_confusion"],
        }
    else:
        st.warning(
            "No se encontró `mlp_resultados.json`; se comparan solo los modelos clásicos. "
            "Ejecuta la celda de exportación del notebook para incluir la red neuronal."
        )

    ranking = sorted(model_results.items(), key=lambda item: item[1]["test"], reverse=True)
    best_name, best_result = ranking[0]
    others_text = ", ".join(f"{name} **{pct_es(result['test'])}**" for name, result in ranking[1:])
    test_students = sum(sum(row) for row in model_results["Árbol de Decisión"]["matrix"])
    st.markdown(
        "#### Exactitud en el conjunto de prueba\n"
        f"Sobre {test_students} estudiantes, el mejor resultado lo obtuvo **{best_name}** con "
        f"**{pct_es(best_result['test'])}**, frente a {others_text}."
    )
    accuracy_cols = st.columns(len(model_results))
    for column, (name, result) in zip(accuracy_cols, model_results.items()):
        column.metric(f"{name} · Test", pct_es(result["test"]).replace(" %", "%"))

    st.markdown("#### Generalización: entrenamiento frente a prueba")
    # Gráfico equivalente al comparativo Train vs. Test del notebook.
    model_names = list(model_results)
    train_scores = [model_results[name]["train"] for name in model_names]
    test_scores = [model_results[name]["test"] for name in model_names]
    positions = range(len(model_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    train_bars = ax.bar(
        [position - bar_width / 2 for position in positions],
        train_scores,
        bar_width,
        label="Entrenamiento (Train)",
        color="#4c72b0",
    )
    test_bars = ax.bar(
        [position + bar_width / 2 for position in positions],
        test_scores,
        bar_width,
        label="Prueba (Test)",
        color="#55a868",
    )
    ax.set_ylabel("Exactitud (Accuracy)")
    ax.set_title("Exactitud de entrenamiento vs. prueba")
    ax.set_xticks(list(positions), model_names)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bars in (train_bars, test_bars):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.2%}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    generalization = pd.DataFrame(
        {
            "Modelo": model_names,
            "Entrenamiento": train_scores,
            "Prueba": test_scores,
            "Brecha (puntos porcentuales)": [
                (train - test) * 100 for train, test in zip(train_scores, test_scores)
            ],
        }
    )
    st.dataframe(
        generalization.style.format(
            {
                "Entrenamiento": "{:.2%}",
                "Prueba": "{:.2%}",
                "Brecha (puntos porcentuales)": "{:.2f} pp",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    gaps_text = ", ".join(
        f"{name} {(result['train'] - result['test']) * 100:.2f} pp".replace(".", ",")
        for name, result in model_results.items()
    )
    st.markdown(
        f"Brecha entre entrenamiento y prueba: {gaps_text}. Una brecha cercana a 0 "
        "indica que el modelo funciona casi igual con estudiantes que no vio al "
        "aprender; en esta partición ninguno muestra un sobreajuste marcado."
    )

    st.markdown("#### Matrices de confusión del conjunto de prueba")
    st.caption(
        "Visualizaciones basadas en las matrices calculadas en el notebook. "
        "Las filas representan la clase real y las columnas la predicción."
    )
    confusion_matrices = [(name, result["matrix"]) for name, result in model_results.items()]
    confusion_cols = st.columns(len(confusion_matrices))
    for column, (model_name, matrix) in zip(confusion_cols, confusion_matrices):
        with column:
            fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=140)
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Dropout (0)", "Graduate (1)"],
                yticklabels=["Dropout (0)", "Graduate (1)"],
                ax=ax,
            )
            ax.set_title(f"Matriz de confusión · {model_name}")
            ax.set_xlabel("Predicción del modelo")
            ax.set_ylabel("Clase real")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    st.markdown("#### Comparativa de detección por clase")
    comparison_rows = []
    for name, result in model_results.items():
        (dropout_ok, dropout_missed), (graduate_missed, graduate_ok) = result["matrix"]
        precision_dropout = dropout_ok / (dropout_ok + graduate_missed)
        recall_dropout = dropout_ok / (dropout_ok + dropout_missed)
        precision_graduate = graduate_ok / (graduate_ok + dropout_missed)
        recall_graduate = graduate_ok / (graduate_ok + graduate_missed)
        f1_dropout = 2 * precision_dropout * recall_dropout / (precision_dropout + recall_dropout)
        f1_graduate = 2 * precision_graduate * recall_graduate / (precision_graduate + recall_graduate)
        comparison_rows.append(
            {
                "Modelo": name,
                "Desertores detectados": f"{dropout_ok} de {dropout_ok + dropout_missed}",
                "Precisión (Dropout)": precision_dropout,
                "Sensibilidad (Dropout)": recall_dropout,
                "Sensibilidad (Graduate)": recall_graduate,
                "F1-Score (promedio)": (f1_dropout + f1_graduate) / 2,
            }
        )
    st.dataframe(
        pd.DataFrame(comparison_rows).style.format(
            {
                "Precisión (Dropout)": "{:.2f}",
                "Sensibilidad (Dropout)": "{:.2f}",
                "Sensibilidad (Graduate)": "{:.2f}",
                "F1-Score (promedio)": "{:.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "La sensibilidad de Dropout indica qué parte de los desertores reales detecta "
        "cada modelo; es la métrica más importante para ofrecer ayuda a tiempo."
    )

    st.markdown("#### Interpretación de los resultados del notebook")
    analysis_cols = st.columns(3 if mlp_eval is not None else 2)
    with analysis_cols[0]:
        st.markdown(
            "**Detección de deserción:** la Regresión Logística identificó "
            "correctamente a 238 de 284 estudiantes que desertaron (recall de 0,84); "
            "el Árbol identificó 228 de 284 (recall de 0,80). En este conjunto de "
            "prueba, la logística dejó menos casos de deserción clasificados como "
            "graduación: 46 frente a 56."
        )
    with analysis_cols[1]:
        st.markdown(
            "**Detección de graduación:** el Árbol identificó correctamente a "
            "411 de 442 graduados (recall de 0,93), frente a 405 de 442 (recall de "
            "0,92) con Regresión Logística. En este experimento, el árbol detectó seis "
            "graduados más, mientras que la logística detectó ocho casos de deserción "
            "adicionales y obtuvo una exactitud global ligeramente mayor."
        )
    if mlp_eval is not None:
        mlp_matrix = model_results[MLP_NAME]["matrix"]
        logistic_matrix = model_results["Regresión Logística"]["matrix"]
        with analysis_cols[2]:
            st.markdown(
                f"**Red neuronal:** su exactitud global fue "
                f"{pct_es(model_results[MLP_NAME]['test'])}. Detectó a {mlp_matrix[0][0]} de "
                f"{sum(mlp_matrix[0])} desertores (la Regresión Logística, {logistic_matrix[0][0]}) "
                f"y clasificó bien a {mlp_matrix[1][1]} de {sum(mlp_matrix[1])} graduados "
                f"(la Regresión Logística, {logistic_matrix[1][1]}). La comparación "
                "proviene de una sola partición de datos y puede variar al reentrenar."
            )


# -----------------------------------------------------------------------------
# Pestaña: gráficos y métricas de la red neuronal (resultados exportados por el notebook)
# -----------------------------------------------------------------------------
with tab_nn:
    st.subheader("Red neuronal (Perceptrón Multicapa)")
    st.write(
        "La red neuronal es un modelo que aprende patrones combinando las nueve "
        "variables del estudiante. Aquí se muestra cómo aprendió y qué tan bien "
        "acierta con estudiantes que no vio durante el entrenamiento."
    )
    with st.expander("¿Cómo está construida la red?"):
        st.markdown(
            "| Capa | Detalle |\n|---|---|\n"
            "| Entrada | 9 neuronas, una por variable del formulario |\n"
            "| Oculta 1 | 32 neuronas, activación ReLU y Dropout 0,2 |\n"
            "| Oculta 2 | 16 neuronas, activación ReLU y Dropout 0,2 |\n"
            "| Salida | 2 neuronas con Softmax: probabilidad de Dropout y de Graduate |\n\n"
            "El *Dropout* apaga al azar algunas neuronas durante el aprendizaje para "
            "que la red no memorice a los estudiantes."
        )

    mlp_results = load_mlp_results()
    if mlp_results is None:
        st.info(
            "No se encontró `mlp_resultados.json`. Ejecuta la celda de exportación del "
            "notebook para generarlo junto a `streamlit_app.py`."
        )
    else:
        try:
            history = mlp_results["historial"]
            best_epoch = int(mlp_results["epoca_convergencia"])
            matrix = mlp_results["matriz_confusion"]
            class_metrics = mlp_results["metricas"]
            macro = class_metrics["Promedio macro"]
            accuracy_test = mlp_results["exactitud"]["test"]
            accuracy_train = mlp_results["exactitud"]["train"]
            epochs = list(range(1, len(history["loss"]) + 1))

            summary_cols = st.columns(4)
            summary_cols[0].metric("Exactitud (prueba)", f"{accuracy_test:.2%}")
            summary_cols[1].metric("Precisión (promedio)", f"{macro['precision']:.2f}")
            summary_cols[2].metric("Sensibilidad (promedio)", f"{macro['recall']:.2f}")
            summary_cols[3].metric("F1-Score (promedio)", f"{macro['f1']:.2f}")

            st.markdown("#### 1. ¿Cómo aprendió la red?")
            curve_cols = st.columns(2)
            for column, (train_key, val_key, title, y_label) in zip(
                curve_cols,
                [
                    ("loss", "val_loss", "Pérdida por época", "Pérdida (error)"),
                    ("accuracy", "val_accuracy", "Exactitud por época", "Exactitud"),
                ],
            ):
                with column:
                    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=140)
                    ax.plot(epochs, history[train_key], label="Entrenamiento")
                    ax.plot(epochs, history[val_key], label="Validación")
                    ax.axvline(
                        best_epoch,
                        color="gray",
                        linestyle="--",
                        label=f"Convergencia (época {best_epoch})",
                    )
                    ax.set_title(title)
                    ax.set_xlabel("Época")
                    ax.set_ylabel(y_label)
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
            st.markdown(
                f"La pérdida mide cuánto se equivoca la red y debe bajar mientras aprende. "
                f"La línea punteada marca la **época {best_epoch}**, cuando dejó de mejorar con "
                f"estudiantes que no usa para aprender; el entrenamiento se detuvo solo en la "
                f"época {mlp_results['epocas_entrenadas']}. Como las líneas de entrenamiento y "
                f"validación van juntas, la red **aprendió patrones generales y no memorizó** "
                f"a los estudiantes (exactitud de entrenamiento {accuracy_train:.2%} frente "
                f"a {accuracy_test:.2%} en prueba)."
            )

            st.markdown("#### 2. ¿En qué acierta y en qué se equivoca?")
            result_cols = st.columns(2)
            with result_cols[0]:
                fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=140)
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=["Dropout (0)", "Graduate (1)"],
                    yticklabels=["Dropout (0)", "Graduate (1)"],
                    ax=ax,
                )
                ax.set_title("Matriz de confusión · Red neuronal")
                ax.set_xlabel("Predicción de la red")
                ax.set_ylabel("Lo que realmente pasó")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            with result_cols[1]:
                metrics_table = pd.DataFrame(class_metrics).T.rename(
                    columns={
                        "precision": "Precisión",
                        "recall": "Sensibilidad (Recall)",
                        "f1": "F1-Score",
                        "soporte": "Estudiantes",
                    }
                )
                st.dataframe(
                    metrics_table.style.format(
                        {
                            "Precisión": "{:.2f}",
                            "Sensibilidad (Recall)": "{:.2f}",
                            "F1-Score": "{:.2f}",
                            "Estudiantes": "{:.0f}",
                        }
                    ),
                    width="stretch",
                )
                st.markdown(
                    "- **Exactitud:** de cada 100 estudiantes, cuántos clasifica bien.\n"
                    "- **Precisión:** cuando la red dice «desertará», qué tan seguido acierta.\n"
                    "- **Sensibilidad:** de los que realmente desertaron, cuántos detecta. "
                    "Es la clave, porque un desertor no detectado no recibe ayuda.\n"
                    "- **F1-Score:** resume precisión y sensibilidad en un solo número."
                )

            dropouts_total = sum(matrix[0])
            graduates_total = sum(matrix[1])
            st.markdown("#### 3. Conclusión")
            st.markdown(
                f"De {dropouts_total} estudiantes que desertaron, la red **detectó a "
                f"{matrix[0][0]} y no detectó a {matrix[0][1]}**. De {graduates_total} que "
                f"se graduaron, clasificó bien a {matrix[1][1]} y marcó en riesgo a "
                f"{matrix[1][0]} por error. Sirve como apoyo para decidir a quién acompañar de "
                "forma temprana, no como veredicto definitivo sobre un estudiante."
            )
        except (KeyError, TypeError, ValueError) as exc:
            st.error(
                "El archivo `mlp_resultados.json` no tiene el formato esperado. "
                f"Vuelve a ejecutar la celda de exportación del notebook. Detalle: {exc}"
            )


# -----------------------------------------------------------------------------
# Pestaña 3: formulario y predicción con los artefactos entrenados
# -----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Simulador predictivo")
    st.write(
        "Ingresa las nueve variables utilizadas durante el entrenamiento. "
        "El simulador aplica el modelo seleccionado sin volver a entrenarlo."
    )
    st.caption(
        "Rangos permitidos: edad 16–100 años; nota de admisión 0–200; "
        "asignaturas aprobadas por semestre 0–30; nota del segundo semestre 0–20. "
        "En los campos binarios, No = 0 y Sí = 1."
    )

    missing_features = []
    if data is not None:
        missing_features = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing_features:
        st.warning(
            "El dataset seleccionado no contiene todas las variables requeridas: "
            + ", ".join(missing_features)
            + ". El formulario sigue disponible, pero confirma que los artefactos "
            "se entrenaron con este esquema exacto."
        )

    if selected_model_name not in loaded_models:
        st.error(artifact_errors.get(selected_model_name, "El modelo seleccionado no está disponible."))
        st.info(
            "Ejecuta la celda de exportación del notebook para guardar "
            "`modelo_logistico.pkl`, `modelo_arbol.pkl` y `modelo_mlp.keras` junto a "
            "`streamlit_app.py`. La red neuronal también requiere `tensorflow`."
        )
    if selected_model_name in SCALED_MODELS and loaded_scaler is None:
        st.error(artifact_errors.get("scaler", "El escalador no está disponible."))
        st.info("Guarda el StandardScaler entrenado en `scaler.pkl` desde el notebook.")

    with st.form("student_prediction_form"):
        left_col, right_col = st.columns(2)
        with left_col:
            age = st.number_input(
                "Edad al matricularse (16–100 años)",
                min_value=16,
                max_value=100,
                value=20,
                step=1,
                key="age_at_enrollment",
            )
            admission_grade = st.number_input(
                "Nota de admisión (0–200)",
                min_value=0.0,
                max_value=200.0,
                value=120.0,
                step=0.1,
                format="%.2f",
                key="admission_grade",
            )
            displaced = binary_selector("¿Es una persona desplazada?", "displaced")
            debtor = binary_selector("¿Tiene deuda de matrícula?", "debtor")
            tuition_current = binary_selector(
                "¿Tiene la matrícula al día?", "tuition_fees_up_to_date"
            )
        with right_col:
            scholarship = binary_selector("¿Es becado/a?", "scholarship_holder")
            approved_first = st.number_input(
                "Asignaturas aprobadas · primer semestre (0–30)",
                min_value=0,
                max_value=30,
                value=4,
                step=1,
                key="approved_first_semester",
            )
            approved_second = st.number_input(
                "Asignaturas aprobadas · segundo semestre (0–30)",
                min_value=0,
                max_value=30,
                value=4,
                step=1,
                key="approved_second_semester",
            )
            second_semester_grade = st.number_input(
                "Nota promedio · segundo semestre (0–20)",
                min_value=0.0,
                max_value=20.0,
                value=12.0,
                step=0.1,
                format="%.2f",
                key="second_semester_grade",
            )

        submitted = st.form_submit_button("🚀 Predecir estado", type="primary")

    if submitted:
        input_values = {
            "Age at enrollment": int(age),
            "Admission grade": float(admission_grade),
            "Displaced": displaced,
            "Debtor": debtor,
            "Tuition fees up to date": tuition_current,
            "Scholarship holder": scholarship,
            "Curricular units 1st sem (approved)": int(approved_first),
            "Curricular units 2nd sem (approved)": int(approved_second),
            "Curricular units 2nd sem (grade)": float(second_semester_grade),
        }
        input_frame = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)

        model = loaded_models.get(selected_model_name)
        if model is None:
            st.error("No se puede predecir: el modelo seleccionado no se cargó.")
        elif selected_model_name in SCALED_MODELS and loaded_scaler is None:
            st.error("No se puede predecir: falta el escalador del modelo seleccionado.")
        else:
            try:
                validate_feature_schema(model, selected_model_name)
                if selected_model_name in SCALED_MODELS:
                    validate_feature_schema(loaded_scaler, "scaler.pkl")
                    prediction_input = loaded_scaler.transform(input_frame)
                else:
                    prediction_input = input_frame

                graduate_probability = None
                if selected_model_name == MLP_NAME:
                    # Softmax: la posición 0 es Dropout y la 1 es Graduate
                    probabilities = model.predict(prediction_input, verbose=0)[0]
                    predicted_class = int(probabilities.argmax())
                    graduate_probability = float(probabilities[1])
                else:
                    predicted_class = int(model.predict(prediction_input)[0])
                predicted_label = TARGET_LABELS.get(predicted_class)
                if predicted_label is None:
                    raise ValueError(f"El modelo devolvió una clase inesperada: {predicted_class}.")

                if predicted_class == 0:
                    st.error("⚠️ Estado estimado: riesgo de deserción (Dropout).")
                    st.markdown(
                        "**Posibles acciones institucionales:** ofrecer orientación "
                        "académica, revisar barreras financieras y contactar al "
                        "estudiante de forma temprana y respetuosa."
                    )
                else:
                    st.success("🎓 Estado estimado: graduación (Graduate).")
                    st.markdown(
                        "**Posibles acciones institucionales:** mantener el "
                        "acompañamiento, reconocer avances y facilitar la continuidad "
                        "académica."
                    )

                if graduate_probability is None and hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(prediction_input)[0]
                    classes = list(model.classes_)
                    graduate_probability = next(
                        (
                            float(probability)
                            for class_value, probability in zip(classes, probabilities)
                            if int(class_value) == 1
                        ),
                        None,
                    )
                if graduate_probability is not None:
                    st.metric(
                        "Probabilidad estimada de graduación",
                        f"{graduate_probability:.1%}",
                    )
                    st.caption(
                        "Probabilidad estimada por el modelo; no es una garantía "
                        "del resultado individual."
                    )
            except Exception as exc:
                st.error(
                    "No fue posible generar la predicción. Verifica que los modelos "
                    "y el escalador se hayan entrenado con las nueve variables, "
                    f"en el orden indicado. Detalle: {exc}"
                )


# -----------------------------------------------------------------------------
# Pestaña 4: métricas y consultas SQL sobre la base SQLite limpia
# -----------------------------------------------------------------------------
with tab_sql:
    st.subheader("Consultas SQL sobre la base de datos limpia")

    try:
        db_path, db_source = ensure_database(data)
    except Exception as exc:
        db_path = None
        st.error(f"No fue posible preparar la base SQLite. Detalle: {exc}")

    if db_path is not None:
        st.caption(f"{db_source} · tabla `{TABLE}` · acceso de solo lectura.")
        sql_metrics_tab, sql_advanced_tab = st.tabs(
            ["⚡ Métricas rápidas", "🔎 Consultas avanzadas e interactivas"]
        )

        # --- Métricas rápidas calculadas con SQL ---------------------------------
        with sql_metrics_tab:
            try:
                summary, _ = run_query(db_path, QUICK_METRICS_SQL)
                row = summary.iloc[0]
                top_cols = st.columns(4)
                top_cols[0].metric("Estudiantes", f"{int(row['total']):,}")
                top_cols[1].metric("Graduados", f"{int(row['graduados']):,}")
                top_cols[2].metric("Desertores", f"{int(row['desertores']):,}")
                top_cols[3].metric("Tasa de deserción", f"{row['tasa_desercion']:.2f}%")
                bottom_cols = st.columns(4)
                bottom_cols[0].metric("Edad promedio", f"{row['edad_promedio']:.1f}")
                bottom_cols[1].metric("Nota promedio 2.º sem.", f"{row['nota_promedio_2s']:.2f}")
                bottom_cols[2].metric("Becarios", f"{row['pct_becarios']:.1f}%")
                bottom_cols[3].metric("Con deuda", f"{row['pct_deudores']:.1f}%")
                with st.expander("Ver consulta SQL de las métricas"):
                    st.code(QUICK_METRICS_SQL, language="sql")

                st.markdown("#### Tasa de deserción por factor")
                factors, _ = run_query(db_path, RISK_FACTORS_SQL)
                st.bar_chart(factors.set_index("grupo")["tasa_desercion"], horizontal=True)
                st.dataframe(
                    factors.rename(
                        columns={
                            "grupo": "Grupo",
                            "estudiantes": "Estudiantes",
                            "tasa_desercion": "Tasa de deserción (%)",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
                with st.expander("Ver consulta SQL de los factores"):
                    st.code(RISK_FACTORS_SQL, language="sql")
            except Exception as exc:
                st.error(f"No fue posible calcular las métricas con SQL. Detalle: {exc}")

        # --- Consultas avanzadas e interactivas ----------------------------------
        with sql_advanced_tab:
            st.markdown("#### 1. Explorador con filtros")
            st.caption("Los filtros se envían como parámetros enlazados a la consulta SQL.")
            try:
                bounds, _ = run_query(
                    db_path,
                    'SELECT MIN("Age at enrollment") AS min_age, '
                    f'MAX("Age at enrollment") AS max_age FROM {TABLE}',
                )
                min_age, max_age = int(bounds.at[0, "min_age"]), int(bounds.at[0, "max_age"])

                filter_cols = st.columns(3)
                with filter_cols[0]:
                    age_range = st.slider("Edad al matricularse", min_age, max_age, (min_age, max_age))
                    grade_range = st.slider("Nota 2.º semestre", 0.0, 20.0, (0.0, 20.0), step=0.5)
                with filter_cols[1]:
                    status_filter = st.selectbox("Estado final", ["Todos", "Dropout", "Graduate"])
                    scholarship_filter = st.selectbox("Becario", ["Todos", "Sí", "No"])
                with filter_cols[2]:
                    debtor_filter = st.selectbox("Deudor", ["Todos", "Sí", "No"])
                    tuition_filter = st.selectbox("Matrícula al día", ["Todos", "Sí", "No"])

                conditions = [
                    '"Age at enrollment" BETWEEN ? AND ?',
                    '"Curricular units 2nd sem (grade)" BETWEEN ? AND ?',
                ]
                params: list = [age_range[0], age_range[1], grade_range[0], grade_range[1]]
                if status_filter != "Todos":
                    conditions.append("Target_Binario = ?")
                    params.append(1 if status_filter == "Graduate" else 0)
                for choice, column in (
                    (scholarship_filter, "Scholarship holder"),
                    (debtor_filter, "Debtor"),
                    (tuition_filter, "Tuition fees up to date"),
                ):
                    if choice != "Todos":
                        conditions.append(f'"{column}" = ?')
                        params.append(1 if choice == "Sí" else 0)
                where_sql = " AND ".join(conditions)

                count_sql = (
                    "SELECT COUNT(*) AS total, "
                    "ROUND(100.0 * SUM(1 - Target_Binario) / COUNT(*), 1) AS tasa_desercion, "
                    'ROUND(AVG("Admission grade"), 1) AS nota_admision '
                    f"FROM {TABLE} WHERE {where_sql}"
                )
                counts, _ = run_query(db_path, count_sql, tuple(params))
                total_found = int(counts.at[0, "total"])
                result_cols = st.columns(3)
                result_cols[0].metric("Estudiantes encontrados", f"{total_found:,}")
                if total_found:
                    result_cols[1].metric("Tasa de deserción", f"{counts.at[0, 'tasa_desercion']:.1f}%")
                    result_cols[2].metric("Nota de admisión promedio", f"{counts.at[0, 'nota_admision']:.1f}")

                select_columns = ", ".join(f'"{column}"' for column in FEATURE_COLUMNS)
                detail_sql = (
                    f"SELECT {select_columns}, Target FROM {TABLE} "
                    f"WHERE {where_sql} LIMIT {EDITOR_MAX_ROWS}"
                )
                detail, _ = run_query(db_path, detail_sql, tuple(params))
                with st.expander("Ver consulta SQL generada"):
                    st.code(detail_sql, language="sql")
                    st.caption(f"Parámetros: {params}")
                st.dataframe(detail, hide_index=True, width="stretch")
                st.download_button(
                    "Descargar resultado (CSV)",
                    detail.to_csv(index=False).encode("utf-8"),
                    file_name="consulta_estudiantes.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"No fue posible ejecutar el explorador. Detalle: {exc}")

            st.divider()
            st.markdown("#### 2. Editor SQL")
            example_name = st.selectbox("Consulta de ejemplo", list(EXAMPLE_QUERIES))
            sql_text = st.text_area(
                "Consulta (solo SELECT o WITH)",
                value=EXAMPLE_QUERIES[example_name],
                height=220,
                key=f"sql_editor_{example_name}",
            )
            with st.expander("Columnas disponibles"):
                try:
                    schema, _ = run_query(
                        db_path,
                        f"SELECT name AS columna, type AS tipo FROM pragma_table_info('{TABLE}')",
                    )
                    st.dataframe(schema, hide_index=True, width="stretch")
                except Exception as exc:
                    st.error(f"No se pudo leer el esquema. Detalle: {exc}")

            if st.button("▶️ Ejecutar consulta", type="primary"):
                try:
                    statement = validate_read_only_sql(sql_text)
                    result, truncated = run_query(db_path, statement, (), EDITOR_MAX_ROWS)
                    st.success(f"{len(result):,} filas devueltas.")
                    if truncated:
                        st.warning(f"Resultado limitado a las primeras {EDITOR_MAX_ROWS:,} filas.")
                    st.dataframe(result, hide_index=True, width="stretch")
                except Exception as exc:
                    st.error(f"No se pudo ejecutar la consulta: {exc}")

st.sidebar.divider()