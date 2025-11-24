import os
import warnings

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dash_table

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
import xgboost as xgb

# ============================================================
# 0. CONFIGURACIÓN GENERAL Y MATHJAX
# ============================================================

warnings.filterwarnings("ignore")

# MathJax v3 (lo maneja internamente dcc.Markdown con mathjax=True)
external_scripts = [
    {
        "src": "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
    }
]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=external_scripts,
    suppress_callback_exceptions=True,
)

app.title = "Dashboard del Proyecto Final"
server = app.server

# ============================================================
# 1. CARGA DE DATOS 
# ============================================================

import os

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Conexión Postgres Render
    engine = DATABASE_URL
else:
    # Conexión Postgres local
    USER = "postgres"
    PASSWORD = "MinimishaAlaska_845" 
    HOST = "localhost"
    PORT = "5432"
    DB = "proyecto_final"

    engine = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"

query = "SELECT * FROM credit_data ORDER BY id;"
df = pd.read_sql(query, engine)
df.drop(columns=["id"], inplace=True)

# ============================================================
# 2. CONFIG GENERALES
# ============================================================

def card_estadistica(titulo, valor, subtitulo="", color="#0d6efd"):
    """
    Card de KPI con borde de color y fondo suave.
    """
    return dbc.Card(
        [
            dbc.CardHeader(titulo, className="fw-bold"),
            dbc.CardBody(
                [
                    html.H4(f"{valor}", className="card-title"),
                    html.P(subtitulo, className="card-text"),
                ]
            ),
        ],
        className="mb-3 shadow-sm",
        style={
            "borderLeft": f"6px solid {color}",
            "backgroundColor": "#f8f9fa",
        },
    )


target_col = "credit_score"
if target_col not in df.columns:
    raise ValueError(f"No se encontró la columna objetivo '{target_col}' en el PKL.")

df[target_col] = df[target_col].astype(str)

# Columnas para EDA
num_cols_all = [
    "age", "annual_income", "monthly_inhand_salary", "num_bank_accounts",
    "num_credit_card", "interest_rate", "delay_from_due_date",
    "num_of_delayed_payment", "changed_credit_limit", "num_credit_inquiries",
    "outstanding_debt", "credit_utilization_ratio", "credit_history_age",
    "total_emi_per_month", "amount_invested_monthly", "monthly_balance"
]

cat_cols_all = [
    "occupation", "credit_mix", "payment_of_min_amount", "payment_behaviour",
    "not_specified", "credit_builder_loan", "personal_loan",
    "debt_consolidation_loan", "student_loan", "payday_loan", "mortgage_loan",
    "auto_loan", "home_equity_loan"
]

def _has_numeric_values(frame: pd.DataFrame, c: str) -> bool:
    if c not in frame.columns:
        return False
    s = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.notna().sum() > 0

num_cols = [c for c in num_cols_all if c in df.columns and _has_numeric_values(df, c)]
cat_cols = [c for c in cat_cols_all if c in df.columns]

# ============================================================
# 3. CONFIG PARA EL MODELO XGBOOST (ENTRENADO EN VIVO)
# ============================================================

# columnas que usó el modelo optimizado (sin annual_income)
numerical_cols_model = [
    "age",
    "monthly_inhand_salary",
    "num_bank_accounts",
    "num_credit_card",
    "interest_rate",
    "delay_from_due_date",
    "num_of_delayed_payment",
    "changed_credit_limit",
    "num_credit_inquiries",
    "outstanding_debt",
    "credit_utilization_ratio",
    "credit_history_age",
    "total_emi_per_month",
    "amount_invested_monthly",
    "monthly_balance",
]

categorical_cols_model = [
    "occupation",
    "credit_mix",
    "payment_of_min_amount",
    "payment_behaviour",
]

loan_cols = [
    "not_specified",
    "credit_builder_loan",
    "personal_loan",
    "debt_consolidation_loan",
    "student_loan",
    "payday_loan",
    "mortgage_loan",
    "auto_loan",
    "home_equity_loan",
]

X = df.drop(columns=[target_col])
y = df[target_col]

# 3.1 Encoding del target y split
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train_enc, y_test_enc = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

# 3.2 Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols_model),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols_model),
    ],
    remainder="passthrough",  # aquí pasan los loan_cols
)

preprocessor.fit(X_train)

ohe = preprocessor.named_transformers_["cat"]
ohe_feature_names = list(ohe.get_feature_names_out(categorical_cols_model))

feature_names = numerical_cols_model + ohe_feature_names + loan_cols

# 3.3 XGBoost con los hiperparámetros del Random Search
xgb_clf = xgb.XGBClassifier(
    colsample_bytree=0.9933692563579372,
    learning_rate=0.12964733273336593,
    max_depth=9,
    min_child_weight=1,
    n_estimators=173,
    reg_alpha=0.5081987767407187,
    reg_lambda=2.3916256135817635,
    subsample=0.943343521925488,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="mlogloss",
)

best_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("xgb", xgb_clf),
    ]
)

best_model.fit(X_train, y_train_enc)

# 3.4 Predicciones, métricas, matriz de confusión
y_pred_enc = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

class_labels = le.classes_

cm = confusion_matrix(y_test_enc, y_pred_enc)
accuracy = accuracy_score(y_test_enc, y_pred_enc)

report_dict = classification_report(
    y_test_enc,
    y_pred_enc,
    target_names=class_labels,
    output_dict=True,
)

metrics_rows = []
for label in class_labels:
    row = {
        "Clase": label,
        "Precision": report_dict[label]["precision"],
        "Recall": report_dict[label]["recall"],
        "F1": report_dict[label]["f1-score"],
        "Soporte": int(report_dict[label]["support"]),
    }
    metrics_rows.append(row)

metrics_rows.append(
    {
        "Clase": "macro avg",
        "Precision": report_dict["macro avg"]["precision"],
        "Recall": report_dict["macro avg"]["recall"],
        "F1": report_dict["macro avg"]["f1-score"],
        "Soporte": int(report_dict["macro avg"]["support"]),
    }
)
metrics_rows.append(
    {
        "Clase": "weighted avg",
        "Precision": report_dict["weighted avg"]["precision"],
        "Recall": report_dict["weighted avg"]["recall"],
        "F1": report_dict["weighted avg"]["f1-score"],
        "Soporte": int(report_dict["weighted avg"]["support"]),
    }
)

df_metrics = pd.DataFrame(metrics_rows)

importances = best_model.named_steps["xgb"].feature_importances_
fi_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(20)
)
top_features_text = ", ".join(fi_df["feature"].head(5))

# ============================================================
# 4. AYUDAS GRÁFICAS PARA EDA
# ============================================================

def _empty_fig(msg="Sin datos con los filtros actuales"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig

def _sanitize_numeric(df_in: pd.DataFrame, column: str):
    if column not in df_in.columns:
        return df_in.iloc[0:0]
    d = df_in[[column, target_col]].copy()
    d[column] = pd.to_numeric(d[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[column, target_col])
    return d

def numeric_histogram(data, column):
    d = _sanitize_numeric(data, column)
    if d.empty:
        return _empty_fig()
    fig = px.histogram(d, x=column, color=target_col, barmode="overlay", nbins=40)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig

def numeric_box(data, column):
    d = _sanitize_numeric(data, column)
    if d.empty:
        return _empty_fig()
    fig = px.box(d, y=column, color=target_col, points="outliers")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig

def categorical_bars(data, column):
    if column not in data.columns:
        return _empty_fig("Columna no encontrada")
    vc = data.groupby([column, target_col], dropna=False).size().reset_index(name="n")
    if vc.empty:
        return _empty_fig()
    fig = px.bar(vc, y=column, x="n", color=target_col, orientation="h", barmode="stack")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig

def corr_heatmap(data, columns):
    cols = [c for c in (columns or []) if c in data.columns]
    if len(cols) < 2:
        return _empty_fig("Selecciona ≥ 2 variables")
    d = data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if d.dropna(how="all").empty:
        return _empty_fig()
    corr = d.corr(numeric_only=True)
    if corr.isnull().all().all():
        return _empty_fig()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig

def apply_base_filters(data, clase_sel, age_range, income_range):
    dfv = data.copy()
    if clase_sel:
        dfv = dfv[dfv[target_col].isin(clase_sel)]
    if "age" in dfv.columns and age_range:
        a = pd.to_numeric(dfv["age"], errors="coerce")
        dfv = dfv[(a >= age_range[0]) & (a <= age_range[1])]
    if "annual_income" in dfv.columns and income_range:
        inc = pd.to_numeric(dfv["annual_income"], errors="coerce")
        dfv = dfv[(inc >= income_range[0]) & (inc <= income_range[1])]
    return dfv

def cards_overview(data: pd.DataFrame):
    total = len(data)

    vc = (
        data[target_col]
        .astype(str)
        .value_counts(normalize=True, dropna=False)
        .sort_index()
    )
    df_bars = pd.DataFrame({target_col: vc.index.astype(str), "pct": vc.values})
    bars = px.bar(df_bars, x=target_col, y="pct", text="pct")
    bars.update_layout(margin=dict(l=10, r=10, t=30, b=10), yaxis_tickformat=".0%")

    estructura = pd.DataFrame({
        "columna": data.columns,
        "tipo": [str(data[c].dtype) for c in data.columns],
    })
    tabla = html.Table(
        [
            html.Thead(html.Tr([html.Th("columna"), html.Th("tipo")])),
            html.Tbody(
                [
                    html.Tr([html.Td(row["columna"]), html.Td(row["tipo"])])
                    for _, row in estructura.head(200).iterrows()
                ]
            ),
        ],
        style={"width": "100%", "display": "block", "overflowX": "auto"},
    )

    return html.Div(
        [
            html.Div(
                [
                    html.H3("Total registros"),
                    html.H1(f"{total:,}"),
                ],
                style={
                    "background": "#1113",
                    "padding": "12px",
                    "borderRadius": "12px",
                    "marginBottom": "10px",
                },
            ),
            html.Div(
                [
                    html.H3("Distribución por credit_score"),
                    dcc.Graph(figure=bars),
                ],
                style={
                    "background": "#1113",
                    "padding": "12px",
                    "borderRadius": "12px",
                    "marginBottom": "10px",
                },
            ),
            html.Div(
                [
                    html.H3("Estructura del dataset"),
                    tabla,
                ],
                style={
                    "background": "#1113",
                    "padding": "12px",
                    "borderRadius": "12px",
                    "marginBottom": "10px",
                },
            ),
        ]
    )

# Figuras iniciales
_default_num = num_cols[0] if num_cols else None
_default_cat = cat_cols[0] if cat_cols else None
_initial_num_hist = numeric_histogram(df, _default_num) if _default_num else _empty_fig("Sin variables numéricas")
_initial_num_box = numeric_box(df, _default_num) if _default_num else _empty_fig("Sin variables numéricas")
_initial_cat_bar = categorical_bars(df, _default_cat) if _default_cat else _empty_fig("Sin variables categóricas")
_initial_corr = corr_heatmap(df, num_cols[:10]) if len(num_cols) >= 2 else _empty_fig("Selecciona ≥ 2 variables")

# ============================================================
# 5. FIGURAS DEL MODELO (MATRIZ + IMPORTANCIA)
# ============================================================

def make_confusion_heatmap():
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    fig = px.imshow(
        cm_df,
        text_auto=".0f",
        color_continuous_scale="Blues",
        labels={"x": "Predicho", "y": "Real", "color": "Recuentos"},
        title="Matriz de Confusión (conjunto de prueba)",
    )
    fig.update_layout(height=450, margin=dict(l=40, r=20, t=60, b=40))
    return fig

def make_feature_importance_fig():
    fig = px.bar(
        fi_df.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        title="Importancia de Variables (Top 20) - XGBoost",
    )
    fig.update_layout(height=500, margin=dict(l=120, r=40, t=60, b=40))
    return fig

fig_confusion = make_confusion_heatmap()
fig_feature_importance = make_feature_importance_fig()

# ============================================================
# 7. SUB-TABS METODOLOGÍA (CON FÓRMULAS LATEX + MATHJAX)
# ============================================================

subtabs_metodologia = dcc.Tabs(
    [
        # --------------------------------------------------
        # a) Definición formal del problema
        # --------------------------------------------------
        dcc.Tab(
            label="a. Definición del Problema",
            children=[
                html.Div(
                    [
                        html.H4("a. Definición del Problema a Resolver", className="mb-3"),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Formalización del problema", className="fw-bold"),
                                            dbc.CardBody(
                                                dcc.Markdown(
                                                    r"""
Se plantea un **problema de clasificación multiclase**.  
Dado un vector de características $\mathbf{x}_i \in \mathbb{R}^p$ de cada cliente, se busca
una función de decisión

$$
f: \mathbb{R}^p \rightarrow \{0,1,2\}
$$

tal que la predicción para el cliente $i$ sea

$$
\hat{y}_i = f(\mathbf{x}_i)
$$

donde los valores $\{0,1,2\}$ representan las clases **Good, Standard, Poor**.
                                                    """,
                                                    mathjax=True,
                                                )
                                            ),
                                        ],
                                        className="shadow-sm mb-4",
                                        style={"borderLeft": "6px solid #0d6efd"},
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Variable objetivo y características", className="fw-bold"),
                                            dbc.CardBody(
                                                dcc.Markdown(
                                                    r"""
- **Variable objetivo:** `credit_score`  
  $y_i \in \{\text{Good}, \text{Standard}, \text{Poor}\}$

- **Vector de características** $\mathbf{x}_i$: incluye variables financieras,
  demográficas y de comportamiento, por ejemplo:

  - número de productos financieros  
  - historial de pagos y mora  
  - endeudamiento y utilización de crédito  
  - ingresos y saldos mensuales
                                                    """,
                                                    mathjax=True,
                                                )
                                            ),
                                        ],
                                        className="shadow-sm mb-4",
                                        style={"borderLeft": "6px solid #6610f2"},
                                    ),
                                    md=6,
                                ),
                            ]
                        ),
                    ],
                    className="p-3",
                )
            ],
        ),

        # --------------------------------------------------
        # b) Preparación de datos
        # --------------------------------------------------
        dcc.Tab(
            label="b. Preparación de Datos",
            children=[
                html.Div(
                    [
                        html.H4("b. Preparación de los Datos", className="mb-3"),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Limpieza y transformación", className="fw-bold"),
                                            dbc.CardBody(
                                                dcc.Markdown(
                                                    r"""
La preparación de datos incluye:

1. **Tratamiento de valores faltantes** y registros atípicos.
2. **Conversión de tipos** y consistencia de categorías.
3. **Estandarización** de variables numéricas mediante

$$
x' = \frac{x - \mu}{\sigma}
$$

donde $\mu$ es la media y $\sigma$ la desviación estándar de cada variable.
                                                    """,
                                                    mathjax=True,
                                                )
                                            ),
                                        ],
                                        className="shadow-sm mb-4",
                                        style={"borderLeft": "6px solid #20c997"},
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Codificación y partición del dataset", className="fw-bold"),
                                            dbc.CardBody(
                                                dcc.Markdown(
                                                    r"""
- Las variables categóricas se representan mediante **One-Hot Encoding**,
  produciendo vectores binarios para cada categoría.

- El dataset se divide en conjuntos de **entrenamiento** y **prueba**:

$$
D = D_{\text{train}} \cup D_{\text{test}}, \quad
D_{\text{train}} \cap D_{\text{test}} = \varnothing
$$

con una proporción aproximada de 70% para entrenamiento y 30% para prueba,
usando muestreo **estratificado** según la variable `credit_score`.
                                                    """,
                                                    mathjax=True,
                                                )
                                            ),
                                        ],
                                        className="shadow-sm mb-4",
                                        style={"borderLeft": "6px solid #ffc107"},
                                    ),
                                    md=6,
                                ),
                            ]
                        ),
                    ],
                    className="p-3",
                )
            ],
        ),

        # --------------------------------------------------
        # c) Selección del modelo (XGBoost)
        # --------------------------------------------------
        dcc.Tab(
            label="c. Selección del Modelo",
            children=[
                html.Div(
                    [
                        html.H4("c. Selección del Modelo o Algoritmo", className="mb-3"),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Modelo base: XGBoost", className="fw-bold"),
                                            dbc.CardBody(
                                                dcc.Markdown(
                                                    r"""
El modelo seleccionado para la clasificación es **XGBoost**, un método
de **gradient boosting** basado en árboles de decisión. Este modelo en particular fue escogido
ya que fue el que obtuvo el mejor desempeño bajo la métrica f1-score macro avg (métrica principal), 
comparado con otros modelos commo regresión logística, Naive Bayes, Random Forest y SVM. 
Para más información sobre el resultado y selección de los modelos, puede consultar el siguiente enlace
en el que se encuentra el desarrollo del proyecto final de la clase Machine Learning, el cual utiliza
el mismo dataset que este proyecto: https://johand-lopez.github.io/proyecto_final_score/intro.html

El modelo ensamble puede escribirse como

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i),
$$

donde cada $f_k$ es un árbol de decisión perteneciente al conjunto
$\mathcal{F}$ de funciones posibles.
                                                    """,
                                                    mathjax=True,
                                                )
                                            ),
                                        ],
                                        className="shadow-sm mb-4",
                                        style={"borderLeft": "6px solid #198754"},
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Función objetivo y regularización", className="fw-bold"),
                                            dbc.CardBody(
                                                dcc.Markdown(
                                                    r"""
XGBoost minimiza una función objetivo compuesta por el error de predicción
y un término de **regularización**:

$$
\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) +
\sum_{k=1}^{K} \Omega(f_k),
$$

donde $l$ es la pérdida (por ejemplo, log-loss) y

$$
\Omega(f_k) = \gamma T_k + \frac{1}{2}\lambda \sum_j w_{jk}^2
$$

controla la complejidad de cada árbol $f_k$ mediante el número de hojas
$T_k$ y los pesos $w_{jk}$.

Los hiperparámetros (profundidad máxima, tasa de aprendizaje, número de árboles,
etc.) se ajustaron mediante un esquema de **búsqueda aleatoria** con
validación cruzada.
                                                    """,
                                                    mathjax=True,
                                                )
                                            ),
                                        ],
                                        className="shadow-sm mb-4",
                                        style={"borderLeft": "6px solid #fd7e14"},
                                    ),
                                    md=6,
                                ),
                            ]
                        ),
                    ],
                    className="p-3",
                )
            ],
        ),

        # --------------------------------------------------
        # d) Entrenamiento y evaluación del modelo
        # --------------------------------------------------
        dcc.Tab(
            label="d. Evaluación del Modelo",
            children=[
                html.Div(
                    [
                        html.H4("d. Entrenamiento y Evaluación del Modelo", className="mb-3"),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Esquema de entrenamiento", className="fw-bold"),
                                            dbc.CardBody(
                                                dcc.Markdown(
                                                    r"""
El modelo se entrena sobre $D_{\text{train}}$ y se valida sobre
$D_{\text{test}}$. Durante la búsqueda de hiperparámetros se utiliza
validación cruzada estratificada para mantener el balance de clases.
                                                    """,
                                                    mathjax=True,
                                                )
                                            ),
                                        ],
                                        className="shadow-sm mb-4",
                                        style={"borderLeft": "6px solid #0dcaf0"},
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Métricas de desempeño", className="fw-bold"),
                                            dbc.CardBody(
                                                dcc.Markdown(
                                                    r"""
Sea $TP$ verdaderos positivos, $FP$ falsos positivos,
$TN$ verdaderos negativos y $FN$ falsos negativos:

- **Accuracy**

$$
Accuracy = \frac{TP + TN}{TP + FP + TN + FN}
$$

- **Precision**

$$
Precision = \frac{TP}{TP + FP}
$$

- **Recall**

$$
Recall = \frac{TP}{TP + FN}
$$

- **F1-score**

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

Estas métricas se calculan para cada clase (`Good`, `Standard`, `Poor`)
y se resumen mediante promedios **macro** y **ponderado**.
                                                    """,
                                                    mathjax=True,
                                                )
                                            ),
                                        ],
                                        className="shadow-sm mb-4",
                                        style={"borderLeft": "6px solid #dc3545"},
                                    ),
                                    md=8,
                                ),
                            ]
                        ),
                    ],
                    className="p-3",
                )
            ],
        ),
    ]
)

# ============================================================
# 8. SUB-TABS RESULTADOS (EDA + MODELO)
# ============================================================

eda_layout = html.Div(
    [
        html.H4("a. Análisis Exploratorio de Datos (EDA)"),

        # --------- FILTROS GLOBALES ----------
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Filtrar por credit_score"),
                        dcc.Dropdown(
                            id="f_clase",
                            options=[
                                {"label": c, "value": c}
                                for c in sorted(df[target_col].unique())
                            ],
                            value=[],
                            multi=True,
                            placeholder="(Opcional)",
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Rango de edad"),
                        dcc.RangeSlider(
                            id="f_age",
                            min=float(
                                pd.to_numeric(df["age"], errors="coerce").min()
                            )
                            if "age" in df.columns
                            else 0.0,
                            max=float(
                                pd.to_numeric(df["age"], errors="coerce").max()
                            )
                            if "age" in df.columns
                            else 1.0,
                            value=[
                                float(
                                    pd.to_numeric(df["age"], errors="coerce").min()
                                )
                                if "age" in df.columns
                                else 0.0,
                                float(
                                    pd.to_numeric(df["age"], errors="coerce").max()
                                )
                                if "age" in df.columns
                                else 1.0,
                            ],
                            tooltip={"placement": "bottom"},
                            allowCross=False,
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Rango de ingresos anuales"),
                        dcc.RangeSlider(
                            id="f_income",
                            min=float(
                                pd.to_numeric(
                                    df.get("annual_income", pd.Series([0])),
                                    errors="coerce",
                                ).min()
                            ),
                            max=float(
                                pd.to_numeric(
                                    df.get("annual_income", pd.Series([1])),
                                    errors="coerce",
                                ).max()
                            ),
                            value=[
                                float(
                                    pd.to_numeric(
                                        df.get("annual_income", pd.Series([0])),
                                        errors="coerce",
                                    ).min()
                                ),
                                float(
                                    pd.to_numeric(
                                        df.get("annual_income", pd.Series([1])),
                                        errors="coerce",
                                    ).max()
                                ),
                            ],
                            tooltip={"placement": "bottom"},
                            allowCross=False,
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                "gap": "12px",
            },
        ),

        html.Hr(),

        # ---------- SUB-TABS DEL EDA ----------
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Overview",
                    children=[html.Div(id="overview_cards", children=cards_overview(df))],
                ),
                dcc.Tab(
                    label="Numéricas",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Variable numérica"),
                                        dcc.Dropdown(
                                            id="num_var",
                                            options=[
                                                {"label": c, "value": c}
                                                for c in num_cols
                                            ],
                                            value=_default_num,
                                        ),
                                    ],
                                    style={"marginBottom": "10px"},
                                ),
                                dcc.Graph(id="num_hist", figure=_initial_num_hist),
                                dcc.Graph(id="num_box", figure=_initial_num_box),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Categóricas",
                    children=[
                        html.Div(
                            [
                                html.Label("Variable categórica"),
                                dcc.Dropdown(
                                    id="cat_var",
                                    options=[
                                        {"label": c, "value": c}
                                        for c in cat_cols
                                    ],
                                    value=_default_cat,
                                ),
                                dcc.Graph(id="cat_bar", figure=_initial_cat_bar),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Correlaciones",
                    children=[
                        html.Div(
                            [
                                html.Label("Variables (numéricas)"),
                                dcc.Dropdown(
                                    id="corr_vars",
                                    options=[
                                        {"label": c, "value": c}
                                        for c in num_cols
                                    ],
                                    value=num_cols[:10],
                                    multi=True,
                                ),
                                dcc.Graph(id="corr_heat", figure=_initial_corr),
                            ]
                        )
                    ],
                ),
            ]
        ),
    ]
)

subtabs_resultados = dcc.Tabs(
    [
        dcc.Tab(label="a. EDA", children=[eda_layout]),

        dcc.Tab(
            label="b. Visualización del Modelo",
            children=[
                html.H4("b. Visualización del Modelo XGBoost"),
                html.P(
                    "Se muestran la matriz de confusión sobre el conjunto de prueba "
                    "y la importancia de variables del modelo XGBoost optimizado."
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="grafico-matriz-confusion",
                                figure=fig_confusion,
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="grafico-importancia",
                                figure=fig_feature_importance,
                            ),
                            md=6,
                        ),
                    ]
                ),
            ],
        ),

        dcc.Tab(
            label="c. Indicadores del Modelo",
            children=[
                html.H4("c. Indicadores de Evaluación del Modelo"),
                html.P(
                    f"Accuracy global del modelo XGBoost en el conjunto de prueba: "
                    f"{accuracy:.4f} ({accuracy*100:.2f}%)."
                ),
                html.H5("Métricas por clase y promedios"),
                dash_table.DataTable(
                    id="tabla-metricas",
                    columns=[
                        {"name": col, "id": col} for col in df_metrics.columns
                    ],
                    data=df_metrics.round(4).to_dict("records"),
                    style_table={
                        "overflowX": "auto",
                        "maxHeight": "400px",
                        "overflowY": "auto",
                    },
                ),
            ],
        ),
    ]
)

# ============================================================
# 9. TABS PRINCIPALES (INTRO, CONTEXTO, PROBLEMA, ETC.)
# ============================================================

tabs_principales = [
    # 1. INTRODUCCIÓN
    dcc.Tab(
        label="1. Introducción",
        children=[
            html.Div(
                [
                    html.Div(
                        html.Img(
                            src="/assets/LogoUN.png",
                            style={"height": "90px", "display": "block", "margin": "0 auto"},
                        ),
                        className="mb-4",
                    ),
                    html.H2("Introducción", className="text-center mb-4"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Panorama general", className="fw-bold"),
                                        dbc.CardBody(
                                            [
                                                html.P(
                                                    "El análisis de datos aplicado al ámbito crediticio permite "
                                                    "comprender los factores que influyen en el riesgo financiero "
                                                    "y optimizar la toma de decisiones. Este proyecto integra una "
                                                    "base de datos alojada en PostgreSQL, análisis exploratorio, "
                                                    "modelado predictivo y visualización interactiva en un dashboard "
                                                    "desarrollado con Dash."
                                                ),
                                                html.P(
                                                    "El objetivo principal es analizar variables financieras y "
                                                    "socioeconómicas para clasificar el credit score en las categorías "
                                                    "Good, Standard y Poor, utilizando técnicas modernas de aprendizaje "
                                                    "automático como XGBoost."
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #0d6efd"},
                                ),
                                md=6,
                                className="mb-4",
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Importancia del estudio", className="fw-bold"),
                                        dbc.CardBody(
                                            [
                                                html.P(
                                                    "La evaluación crediticia es un proceso crítico en el sector "
                                                    "financiero. Contar con modelos sólidos y visualizaciones claras "
                                                    "permite mejorar la comprensión de patrones de riesgo, identificar "
                                                    "perfiles relevantes y fortalecer la toma de decisiones basadas "
                                                    "en datos."
                                                ),
                                                html.P(
                                                    "El dashboard final permite explorar dinámicamente las variables, "
                                                    "visualizar métricas del modelo y entender cómo distintos factores "
                                                    "impactan el puntaje crediticio."
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #198754"},
                                ),
                                md=6,
                                className="mb-4",
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                card_estadistica(
                                    "Registros en el dataset",
                                    f"{len(df):,}".replace(",", "."),
                                    "Observaciones totales disponibles para análisis."
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                card_estadistica(
                                    "Número de variables",
                                    f"{df.shape[1]}",
                                    "Atributos financieros, demográficos y comportamentales."
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                card_estadistica(
                                    "Clases objetivo",
                                    ", ".join(y.unique()),
                                    "Categorías del credit score."
                                ),
                                md=4,
                            ),
                        ],
                        className="mt-4 mb-4"
                    ),
                    html.Div(
                        html.Img(
                            src="/assets/Score.png",
                            style={"height": "150px", "display": "block", "margin": "0 auto"},
                        ),
                        className="mt-3"
                    )
                ],
                className="p-4",
            )
        ],
    ),

    # 2. CONTEXTO
    dcc.Tab(
        label="2. Contexto",
        children=[
            html.Div(
                [
                    html.H2("Contexto", className="text-center mb-4"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Descripción del conjunto de datos", className="fw-bold"),
                                        dbc.CardBody(
                                            html.P(
                                                r"""El dataset contiene información financiera, demográfica y de 
                                                comportamiento crediticio. Estas variables permiten caracterizar 
                                                perfiles de riesgo y sirven como insumo para el modelo predictivo.
                                                
                                                El dataset escogido para el desarrollo del proyecto se encuentra alojado
                                                en Kaggle bajo el nombre de "Credit Score Classification" y el mismo
                                                se encuentra disponible en el siguiente enlace: https://www.kaggle.com/datasets/parisrohan/credit-score-classification
                                                """
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #0dcaf0"},
                                ),
                                md=7,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Contexto institucional", className="fw-bold"),
                                        dbc.CardBody(
                                            [
                                                html.Img(
                                                    src="/assets/Banco.jpg",
                                                    style={
                                                        "width": "100%",
                                                        "maxHeight": "140px",
                                                        "objectFit": "contain",
                                                        "marginBottom": "10px",
                                                    },
                                                ),
                                                html.P(
                                                    "El análisis se desarrolla en un contexto académico para el "
                                                    "curso de Visualización de Datos, integrando PostgreSQL con Dash "
                                                    "para la consulta dinámica y visualización del riesgo crediticio."
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #20c997"},
                                ),
                                md=5,
                            ),
                        ],
                        className="mb-4"
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader("Variables de interés", className="fw-bold"),
                            dbc.CardBody(
                                html.Ul(
                                    [
                                        html.Li("credit_score (objetivo) — Good, Standard, Poor."),
                                        html.Li("annual_income, monthly_inhand_salary."),
                                        html.Li("num_bank_accounts, num_credit_card."),
                                        html.Li("interest_rate, outstanding_debt."),
                                        html.Li("num_of_delayed_payment, credit_history_age."),
                                    ]
                                )
                            ),
                        ],
                        className="shadow-sm",
                        style={"borderLeft": "6px solid #ffc107"},
                    ),
                ],
                className="p-4",
            )
        ],
    ),

    # 3. PLANTEAMIENTO DEL PROBLEMA
    dcc.Tab(
        label="3. Planteamiento del Problema",
        children=[
            html.Div(
                [
                    html.H2("Planteamiento del Problema", className="text-center mb-4"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("📉 Problema Actual", className="fw-bold fs-5"),
                                        dbc.CardBody(
                                            html.P(
                                                "Los métodos tradicionales de evaluación crediticia pueden ser "
                                                "lentos y poco precisos, afectando la capacidad de los bancos "
                                                "para identificar perfiles de riesgo."
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #dc3545"},
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("⚠️ Impacto", className="fw-bold fs-5"),
                                        dbc.CardBody(
                                            html.P(
                                                "Una clasificación inexacta del riesgo puede resultar en pérdidas "
                                                "financieras o decisiones injustas hacia clientes solventes."
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #ffc107"},
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("🔍 Contexto del Riesgo", className="fw-bold fs-5"),
                                        dbc.CardBody(
                                            html.P(
                                                "El score crediticio depende de múltiples factores financieros "
                                                "y de comportamiento, lo que requiere modelos robustos para su "
                                                "interpretación adecuada."
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #0d6efd"},
                                ),
                                md=4,
                            ),
                        ],
                        className="mb-4"
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader("❓ Pregunta Problema", className="fw-bold fs-4"),
                            dbc.CardBody(
                                html.P(
                                    "¿Qué variables influyen de manera más significativa en la clasificación "
                                    "del credit score de los usuarios y cómo visualizar estos patrones mediante "
                                    "un dashboard conectado a PostgreSQL?"
                                )
                            ),
                        ],
                        className="shadow-sm",
                        style={"borderLeft": "6px solid #20c997"},
                    ),
                ],
                className="p-4",
            )
        ],
    ),

    # 4. OBJETIVOS Y JUSTIFICACIÓN
    dcc.Tab(
        label="4. Objetivos y Justificación",
        children=[
            html.Div(
                [
                    html.Div(
                        html.Img(
                            src="/assets/Objeticos.jpg",
                            style={"height": "90px", "display": "block", "margin": "0 auto"},
                        ),
                        className="mb-4",
                    ),
                    html.H2("Objetivos y Justificación", className="text-center mb-4"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("🎯 Objetivo General", className="fw-bold fs-5"),
                                        dbc.CardBody(
                                            html.P(
                                                "Desarrollar un modelo predictivo multiclase para clasificar el "
                                                "credit score en Good, Standard y Poor."
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #0d6efd"},
                                ),
                                md=6,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("📌 Objetivos Específicos", className="fw-bold fs-5"),
                                        dbc.CardBody(
                                            html.Ul(
                                                [
                                                    html.Li("Realizar un análisis exploratorio de los datos."),
                                                    html.Li("Entrenar y evaluar modelos predictivos."),
                                                    html.Li("Comparar métricas de desempeño."),
                                                    html.Li("Integrar el modelo en un dashboard interactivo."),
                                                ]
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #198754"},
                                ),
                                md=6,
                            ),
                        ],
                        className="mb-4"
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader("🧩 Justificación", className="fw-bold fs-5"),
                            dbc.CardBody(
                                html.P(
                                    "El proyecto combina visualización, bases de datos y modelado predictivo, "
                                    "fortaleciendo la interpretación del riesgo crediticio mediante técnicas de "
                                    "Machine Learning."
                                )
                            ),
                        ],
                        className="shadow-sm",
                        style={"borderLeft": "6px solid #fd7e14"},
                    ),
                ],
                className="p-4",
            )
        ],
    ),

    # 5. MARCO TEÓRICO
    dcc.Tab(
        label="5. Marco Teórico",
        children=[
            html.Div(
                [
                    html.H2("Marco Teórico", className="text-center mb-4"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("📘 Riesgo Crediticio y Scoring", className="fw-bold"),
                                        dbc.CardBody(
                                            html.P(
                                                "El riesgo crediticio se refiere a la probabilidad de que un cliente "
                                                "incumpla sus obligaciones financieras. El scoring crediticio utiliza "
                                                "variables socioeconómicas, financieras y de comportamiento para "
                                                "estimar dicho riesgo."
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #0d6efd"},
                                ),
                                md=6,
                                className="mb-4",
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("🔢 Problemas de Clasificación Multiclase", className="fw-bold"),
                                        dbc.CardBody(
                                            html.P(
                                                "Un modelo multiclase asigna cada observación a una categoría "
                                                "específica entre tres o más etiquetas. En este proyecto, la variable "
                                                "objetivo tiene tres clases: Good, Standard y Poor."
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #6610f2"},
                                ),
                                md=6,
                                className="mb-4",
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("🌲 XGBoost y Modelos Ensemble", className="fw-bold"),
                                        dbc.CardBody(
                                            html.P(
                                                "XGBoost es un algoritmo basado en boosting que construye múltiples "
                                                "árboles secuencialmente, corrigiendo los errores de etapas previas. "
                                                "Su eficiencia, regularización y manejo de interacciones complejas "
                                                "lo hacen adecuado para tareas de crédito."
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #198754"},
                                ),
                                md=6,
                                className="mb-4",
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("📊 Métricas de Evaluación", className="fw-bold"),
                                        dbc.CardBody(
                                            html.P(
                                                "El desempeño del modelo se evalúa mediante métricas como Accuracy, "
                                                "Precision, Recall y F1-score, tanto por clase como en sus promedios "
                                                "macro y ponderados. La matriz de confusión permite identificar "
                                                "errores específicos entre clases."
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #dc3545"},
                                ),
                                md=6,
                                className="mb-4",
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("📈 Dashboards y Visualización de Datos", className="fw-bold"),
                                        dbc.CardBody(
                                            html.P(
                                                "Los dashboards son herramientas clave para comunicar resultados de "
                                                "manera clara y dinámica. Dash permite integrar consultas SQL, "
                                                "modelos predictivos y visualizaciones interactivas en una sola "
                                                "interfaz para análisis de riesgo."
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                    style={"borderLeft": "6px solid #fd7e14"},
                                ),
                                md=12,
                            )
                        ]
                    ),
                ],
                className="p-4",
            )
        ],
    ),

    # 6. METODOLOGÍA
    dcc.Tab(
        label="6. Metodología",
        children=[
            html.Div(
                [
                    html.Div(
                        html.Img(
                            src="/assets/P1.png",
                            style={"height": "90px", "display": "block", "margin": "0 auto"},
                        ),
                        className="mb-3",
                    ),
                    html.H2("Metodología", className="text-center mb-4"),
                    subtabs_metodologia,
                ],
                className="p-4",
            )
        ],
    ),

    # 7. RESULTADOS
    dcc.Tab(
        label="7. Resultados y Análisis Final",
        children=[
            html.Div(
                [
                    html.Div(
                        html.Img(
                            src="/assets/Score.png",
                            style={"height": "90px", "display": "block", "margin": "0 auto"},
                        ),
                        className="mb-3",
                    ),
                    html.H2("Resultados y Análisis Final", className="text-center mb-4"),
                    subtabs_resultados,
                ],
                className="p-4",
            )
        ],
    ),

    # 8. CONCLUSIONES
    dcc.Tab(
        label="8. Conclusiones",
        children=[
            html.Div(
                [
                    html.H2("Conclusiones", className="text-center mb-4"),
                    dbc.Card(
                        [
                            dbc.CardHeader("Conclusiones Generales", className="fw-bold fs-5"),
                            dbc.CardBody(
                                [
                                    html.P(
                                        "El modelo XGBoost implementado alcanzó un desempeño sólido, con un "
                                        "accuracy del 78.4% y un F1-score macro cercano a 0.78. Estos resultados "
                                        "indican una buena capacidad de clasificación entre las categorías "
                                        "Good, Standard y Poor."
                                    ),
                                    html.P(
                                        "Las variables de mayor importancia fueron credit_mix, outstanding_debt, "
                                        "interest_rate, num_of_delayed_payment y payment_of_min_amount, lo que "
                                        "confirma su relevancia en la literatura del riesgo crediticio."
                                    ),
                                    
                                    html.P(
                                        """ Algunas limitaciones que encontramos a la hora de desarrollar
                                        el proyecto fueron tanto la presencia de datos faltantes y atípicos,
                                        como el registro erróneo de algunas variables, todo esto fue tratado antes
                                        de la implementación de modelo. Además, el dataset contaba con un tamaño
                                        considerable, lo cual implicó cierto costos computacionales a la hora de la 
                                        búsqueda de mejores hiperparámetros para el modelo XGBoost.
                                        """
                                    ),  
                                    html.P(
                                        "El dashboard desarrollado integra PostgreSQL, análisis exploratorio, "
                                        "modelos predictivos y visualizaciones interactivas, logrando una "
                                        "herramienta completa para explorar el comportamiento crediticio."
                                    ),
                                    html.P(
                                        """ Algunas consideraciones para futuras versiones del análisis incluyen 
                                        la implementación de modelos de tipo stacking y/o redes neuronales, que
                                        permitan captar de mejor manera los patrones de estos datos.
                                        """
                                    ),      
                                    html.P(
                                        "Este proyecto constituye una base sólida para trabajos futuros que "
                                        "involucren optimización adicional de hiperparámetros, incorporación de "
                                        "nuevas variables o ampliación hacia aplicaciones reales en instituciones "
                                        "financieras."
                                    ),
                                ]
                            ),
                        ],
                        className="shadow-sm",
                        style={"borderLeft": "6px solid #20c997"},
                    ),
                ],
                className="p-4",
            )
        ],
    ),
]

# ============================================================
# 10. LAYOUT
# ============================================================

app.layout = dbc.Container(
    [
        html.H1("Dashboard del Proyecto Final", className="text-center my-4"),
        dcc.Tabs(tabs_principales),
    ],
    fluid=True,
)

# ============================================================
# 11. CALLBACKS DE EDA
# ============================================================

@app.callback(
    Output("overview_cards", "children"),
    Input("f_clase", "value"),
    Input("f_age", "value"),
    Input("f_income", "value"),
)
def update_overview(clase_sel, age_range, income_range):
    try:
        dfv = apply_base_filters(df, clase_sel, age_range, income_range)
        return cards_overview(dfv if not dfv.empty else df.head(0))
    except Exception as e:
        print("OVERVIEW ERROR:", e)
        return html.Div("No se pudo generar Overview.")

@app.callback(
    Output("num_hist", "figure"),
    Output("num_box", "figure"),
    Input("f_clase", "value"),
    Input("f_age", "value"),
    Input("f_income", "value"),
    Input("num_var", "value"),
)
def update_numeric(clase_sel, age_range, income_range, var):
    try:
        if not var or var not in df.columns:
            return _empty_fig(), _empty_fig()
        dfv = apply_base_filters(df, clase_sel, age_range, income_range)
        if dfv.empty:
            return _empty_fig(), _empty_fig()
        return numeric_histogram(dfv, var), numeric_box(dfv, var)
    except Exception as e:
        print("NUMERIC ERROR:", e)
        return _empty_fig(), _empty_fig()

@app.callback(
    Output("cat_bar", "figure"),
    Input("f_clase", "value"),
    Input("f_age", "value"),
    Input("f_income", "value"),
    Input("cat_var", "value"),
)
def update_categorical(clase_sel, age_range, income_range, var):
    try:
        if not var or var not in df.columns:
            return _empty_fig("Selecciona una variable")
        dfv = apply_base_filters(df, clase_sel, age_range, income_range)
        if dfv.empty:
            return _empty_fig()
        return categorical_bars(dfv, var)
    except Exception as e:
        print("CATEGORICAL ERROR:", e)
        return _empty_fig()

@app.callback(
    Output("corr_heat", "figure"),
    Input("f_clase", "value"),
    Input("f_age", "value"),
    Input("f_income", "value"),
    Input("corr_vars", "value"),
)
def update_corr(clase_sel, age_range, income_range, vars_sel):
    try:
        dfv = apply_base_filters(df, clase_sel, age_range, income_range)
        if dfv.empty:
            return _empty_fig()
        return corr_heatmap(dfv, vars_sel)
    except Exception as e:
        print("CORR ERROR:", e)
        return _empty_fig()

# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)


