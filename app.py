import base64
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from pathlib import Path
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from flask import Flask
import zipfile
import io
import pandas as pd
import os
import plotly.express as px
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc


# —————————————————————————————
# 1) Configuración inicial
# —————————————————————————————
TMP_DIR = Path(tempfile.gettempdir()) / "ohs_tmp"
TMP_DIR.mkdir(exist_ok=True)
ALLOWED = {".jpg", ".jpeg", ".png"}

POSE = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
SEG  = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
P    = mp.solutions.pose.PoseLandmark

CLR_LINE, CLR_PT = (0, 230, 127), (250, 250, 250)
IDEAL_RGBA      = (0, 255, 0, 110)

METRIC_EXPLANATIONS = {
    "Hip flex":         "Ángulo hombro–cadera–rodilla: flexión de cadera.",
    "Knee flex":        "Ángulo cadera–rodilla–tobillo: flexión de rodilla.",
    "Shoulder flex":    "Ángulo cadera–hombro–muñeca: flexión de hombro.",
    "|Trunk-Tibia|":    "Diferencia ángulo tronco–tibia y tibia–tronco.",
    "Ankle DF":         "Dorsiflexión de tobillo: inclinación del pie.",
    "Apertura rodillas":"Distancia horizontal entre rodillas.",
    "Apertura pies":    "Distancia horizontal entre tobillos."
}

# —————————————————————————————
# 2) Funciones auxiliares
# —————————————————————————————

def b64_to_cv2(content):
    _, b64 = content.split(",", 1)
    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(b64), np.uint8),
        cv2.IMREAD_COLOR
    )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def cv2_to_b64(img):
    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )
    return base64.b64encode(buf).decode()

def angle_between(u, v):
    cos = np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def crop_person(img, lm):
    H, W = img.shape[:2]
    pts = np.array([(p.x*W, p.y*H) for p in lm])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    pad = 0.2
    x0 = max(0, int(x0 - pad*(x1-x0)))
    y0 = max(0, int(y0 - pad*(y1-y0)))
    x1 = min(W, int(x1 + pad*(x1-x0)))
    y1 = min(H, int(y1 + (pad+0.1)*(y1-y0)))
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (480, int(480*crop.shape[0]/crop.shape[1])))

def card(var, val):
    cid = f"card-{var}".replace(" ", "-")
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Small(var, className="text-muted"),
                html.H4(f"{val:.1f}°" if isinstance(val, (int, float)) else f"{val}", className="mb-0")
            ])
        ], color="dark", outline=True, className="m-1 p-2", style={"minWidth":"120px"}, id=cid),
        dbc.Tooltip(METRIC_EXPLANATIONS.get(var, ""), target=cid)
    ])

# —————————————————————————————
# 3) Análisis Sagital
# —————————————————————————————

def analyze_sagital(img):
    # 1) detectamos pose en toda la imagen
    res1 = POSE.process(img)
    if not res1.pose_landmarks:
        return None, None, {}
    crop = crop_person(img, res1.pose_landmarks.landmark)
    h, w = crop.shape[:2]

    # 2) re-detectamos sobre el recorte
    res2 = POSE.process(crop)
    if not res2.pose_landmarks:
        return crop, crop, {}
    lm2 = res2.pose_landmarks.landmark

    # 3) extraemos los puntos relevantes
    side = "R" if lm2[P.RIGHT_HIP].visibility >= lm2[P.LEFT_HIP].visibility else "L"
    choose = lambda L, R: R if side=="R" else L
    ids = [choose(getattr(P, f"LEFT_{n}"), getattr(P, f"RIGHT_{n}"))
           for n in ("SHOULDER","HIP","KNEE","ANKLE","HEEL","FOOT_INDEX","WRIST")]
    SHp, HIp, KNp, ANp, HEp, FTp, WRp = [
        (int(lm2[i].x*w), int(lm2[i].y*h)) for i in ids
    ]

    # 4) calculamos ángulos
    hip_flex      = angle_between(np.array(SHp)-HIp, np.array(KNp)-HIp)
    knee_flex     = angle_between(np.array(HIp)-KNp, np.array(ANp)-KNp)
    shoulder_flex = angle_between(np.array(HIp)-SHp, np.array(WRp)-SHp)
    trunk_tibia   = abs(hip_flex - knee_flex)
    raw_heel      = angle_between(np.array(KNp)-ANp, np.array(HEp)-ANp) - 90
    raw_toe       = angle_between(np.array(KNp)-ANp, np.array(FTp)-ANp) - 90
    ankle_df      = (abs(raw_heel) + abs(raw_toe)) / 2

    data = {
        "Hip flex":      hip_flex,
        "Knee flex":     knee_flex,
        "Shoulder flex": shoulder_flex,
        "|Trunk-Tibia|": trunk_tibia,
        "Ankle DF":      ankle_df
    }

    # 5) segmentamos y difuminamos fondo
    seg  = SEG.process(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    mask = seg.segmentation_mask > 0.6
    blur = cv2.GaussianBlur(crop, (17,17), 0)
    vis  = np.where(mask[...,None], crop, blur).astype(np.uint8)

    # 6) dibujamos ángulos con flechas azules + puntos + texto
    for name, (A, B, C) in [
        ("Hip flex",      (SHp, HIp, KNp)),
        ("Knee flex",     (HIp, KNp, ANp)),
        ("Shoulder flex", (HIp, SHp, WRp))
    ]:
        # flechas azules: vectores que definen el ángulo
        cv2.arrowedLine(vis, B, A, (255, 0, 0), 2, tipLength=0.1)
        cv2.arrowedLine(vis, B, C, (255, 0, 0), 2, tipLength=0.1)
        # puntos en blanco
        for pt in (A, B, C):
            cv2.circle(vis, pt, 7, CLR_PT, -1)
        # valor numérico
        txt = f"{data[name]:.1f}"
        pos = (B[0] + 12, B[1] - 12)
        cv2.putText(vis, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (  0,  0,  0), 2, cv2.LINE_AA)

    # 7) dibujamos tobillo (puntos + texto)
    cv2.line(vis, KNp, ANp, CLR_LINE, 6)
    cv2.line(vis, HEp, FTp, CLR_LINE, 6)
    for pt in (KNp, ANp, HEp, FTp):
        cv2.circle(vis, pt, 7, CLR_PT, -1)
    txt = f"{ankle_df:.1f}"
    pos = (ANp[0] + 12, ANp[1] - 12)
    cv2.putText(vis, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(vis, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (  0,  0,  0), 2, cv2.LINE_AA)

    return crop, vis, data


# —————————————————————————————
# 4) Análisis Frontal (con rodillas y pies)
# —————————————————————————————

def analyze_frontal(img):
    res = POSE.process(img)
    if not res.pose_landmarks:
        return None, None, None

    # 1) recorte
    crop = crop_person(img, res.pose_landmarks.landmark)
    h, w = crop.shape[:2]
    lm2 = POSE.process(crop).pose_landmarks.landmark

    # 2) puntos de interés
    SHL = (int(lm2[P.LEFT_SHOULDER].x*w),  int(lm2[P.LEFT_SHOULDER].y*h))
    SHR = (int(lm2[P.RIGHT_SHOULDER].x*w), int(lm2[P.RIGHT_SHOULDER].y*h))
    LHL = (int(lm2[P.LEFT_HIP].x*w),       int(lm2[P.LEFT_HIP].y*h))
    RHL = (int(lm2[P.RIGHT_HIP].x*w),      int(lm2[P.RIGHT_HIP].y*h))
    LWL = (int(lm2[P.LEFT_WRIST].x*w),     int(lm2[P.LEFT_WRIST].y*h))
    RWL = (int(lm2[P.RIGHT_WRIST].x*w),    int(lm2[P.RIGHT_WRIST].y*h))
    LKL = (int(lm2[P.LEFT_KNEE].x*w),      int(lm2[P.LEFT_KNEE].y*h))
    RKL = (int(lm2[P.RIGHT_KNEE].x*w),     int(lm2[P.RIGHT_KNEE].y*h))
    LAL = (int(lm2[P.LEFT_ANKLE].x*w),     int(lm2[P.LEFT_ANKLE].y*h))
    RAL = (int(lm2[P.RIGHT_ANKLE].x*w),    int(lm2[P.RIGHT_ANKLE].y*h))

    vis = crop.copy()

    # 3) líneas de hombro→cadera→muñeca
    for a,b in [(SHL,LHL),(SHR,RHL),(SHL,LWL),(SHR,RWL)]:
        cv2.line(vis, a, b, IDEAL_RGBA[:3], 4)

    # 4) líneas apertura en naranja
    cv2.line(vis, LKL, RKL, (0,165,255), 3, cv2.LINE_AA)
    cv2.line(vis, LAL, RAL, (0,165,255), 3, cv2.LINE_AA)

    # 5) líneas cadera→rodilla→tobillo
    for hip, knee, ankle in [(LHL,LKL,LAL),(RHL,RKL,RAL)]:
        cv2.line(vis, hip, knee, CLR_LINE, 6)
        cv2.line(vis, knee, ankle, CLR_LINE, 6)

    # 6) puntos
    for p in (SHL,SHR,LHL,RHL,LWL,RWL,LKL,RKL,LAL,RAL):
        cv2.circle(vis, p, 8, CLR_PT, -1)

    # 7) horizontales de referencia
    for y in (SHL[1], SHR[1], LHL[1], RHL[1], LWL[1], RWL[1]):
        cv2.line(vis, (0,y), (w,y), CLR_LINE, 3)

    # 8) métricas
    D_rod = abs(RKL[0] - LKL[0])
    D_pie = abs(RAL[0] - LAL[0])
    data = {
        "Left Shoulder (px)":   SHL[1],
        "Right Shoulder (px)":  SHR[1],
        "Δ Shoulder (px)":      abs(SHR[1]-SHL[1]),
        "Left Hip (px)":        LHL[1],
        "Right Hip (px)":       RHL[1],
        "Δ Hip (px)":           abs(RHL[1]-LHL[1]),
        "Left Wrist (px)":      LWL[1],
        "Right Wrist (px)":     RWL[1],
        "Δ Wrist (px)":         abs(RWL[1]-LWL[1]),
        "Apertura rodillas":    D_rod,
        "Apertura pies":        D_pie
    }

    return crop, vis, data


# —————————————————————————————
# 5) Configuración Dash y Layout
# —————————————————————————————

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Overhead-Squat Analyzer"

app.layout = dbc.Container([
    # — Navbar igual —
    dbc.Navbar(
        dbc.Container([
            html.Img(src="/assets/angle.png", height="40px"),
            dbc.NavbarBrand("OHS Analyzer", className="ms-2"),
        ]),
        color="light", dark=False, className="mb-4"
    ),
    # — Sección educativa ampliada con ejemplos —
    dbc.Card([
        dbc.CardHeader(" ¿Cómo calculamos las métricas?"),
        dbc.CardBody([
            html.Ul([
                html.Li("MediaPipe detecta puntos clave en la silueta: hombros, caderas, rodillas, tobillos, talón y punta de pie."),

                html.Li("Convertimos esas posiciones normalizadas (0–1) a píxeles, multiplicando por el ancho y alto del área recortada."),

                html.Li([
                    html.B("Vista Sagital (de lado):"),
                    html.Ul([
                        html.Li("**Hip flex**: imagina dos palos unidos en la cadera. Uno va hasta el hombro, otro hasta la rodilla. Medimos el ángulo que forman en la cadera, como abrir o cerrar una puerta."),

                        html.Li("**Knee flex**: lo mismo en la rodilla: un palo de cadera a rodilla y otro de rodilla a tobillo, mide cuánto dobla la pierna."),

                        html.Li("**Shoulder flex**: palo de cadera a hombro y palo de hombro a muñeca, para ver cuánto levantas el brazo."),

                        html.Li("**Trunk–Tibia**: restamos el número de Hip flex menos el de Knee flex y tomamos el valor absoluto, así vemos si tu tronco y tibia están bien alineados o si se desvían."),

                        html.Li("**Ankle DF**: hacemos dos ángulos en el tobillo, uno con el talón y otro con la punta del pie. A cada uno le restamos noventa grados para referirlo a la vertical, y luego promediamos ambos valores para tener la dorsiflexión final.")
                    ])
                ]),

                html.Li([
                    html.B("Vista Frontal (de frente):"),
                    html.Ul([
                        html.Li("**Simetría de altura**: comparamos la altura de hombros, caderas y muñecas. Restamos la coordenada vertical (Y) de cada lado: si da cero, están al mismo nivel."),

                        html.Li("**Apertura de rodillas**: restamos la posición horizontal (X) de la rodilla derecha menos la izquierda. Así medimos cuán separadas las tienes."),

                        html.Li("**Apertura de pies**: igual, pero con los tobillos. Una base más ancha da más estabilidad."),

                        html.Li("**Interpretación**: cuanto más cerca de cero salgan estas diferencias, mejor tu alineación y equilibrio lateral.")
                    ])
                ]),

                html.Li([
                    html.B("Ejemplo práctico:"),
                    html.Ul([
                        html.Li("Si tu Hip flex es 50 y tu Knee flex 45, entonces Trunk–Tibia = |50 – 45| = 5 (perfecta alineación si es bajo)."),
                        html.Li("Si tus tobillos están a 30 píxeles de separación, sabrás cuán ancha es tu base de apoyo.")
                    ])
                ])
            ])
        ])
    ], color="info", inverse=True, className="mb-4"),


    # — Título y descripción —  
    html.Div([
        html.H5("¿Qué métricas medimos?", className="text-secondary text-center"),
        html.P("Ángulos siempre positivos y aperturas en vista frontal.", className="text-center")
    ], className="mb-4"),

    # — Row principal con dos columnas idénticas —  
    dbc.Row([
        # Columna Sagital
        dbc.Col([
            html.H5("Sagittal View", className="text-secondary text-center mb-2"),
            dcc.Upload(
                id="up-sag",
                children=dbc.Button("Upload Sagittal Image", color="primary", className="w-100"),
                multiple=False
            ),
            dbc.Spinner(html.Div(id="out-sag")),
        ], md=6, style={'minHeight': '600px'}),

        # Columna Frontal
        dbc.Col([
            html.H5("Frontal View", className="text-secondary text-center mb-2"),
            dcc.Upload(
                id="up-front",
                children=dbc.Button("Upload Frontal Image", color="primary", className="w-100"),
                multiple=False
            ),
            dbc.Spinner(html.Div(id="out-front")),
        ], md=6, style={'minHeight': '600px'}),
    ], justify="center", className="g-4 mb-4"),

    html.Hr(),

    # — Footer —  
    dbc.Row(
        dbc.Col(
            html.Div("Powered by STA METHODOLOGIES • Luciano Sacaba",
                     className="text-center text-muted small"),
            width=12
        )
    )
], fluid=True)


# —————————————————————————————
# Callbacks para análisis
# —————————————————————————————
@app.callback(
    Output("out-sag", "children"),
    Input("up-sag", "contents"),
    State("up-sag", "filename")
)
def analyze_sag(contents, filename):
    if not contents:
        return ""
    if Path(filename).suffix.lower() not in ALLOWED:
        return dbc.Alert("Formato no soportado", color="danger")
    img = b64_to_cv2(contents)
    crop, vis, data = analyze_sagital(img)
    if crop is None:
        return dbc.Alert("No se detectó pose", color="warning")

    crop_b64 = cv2_to_b64(crop)
    vis_b64  = cv2_to_b64(vis)
    fig = px.bar(x=list(data.keys()), y=list(data.values()),
                 labels={'x':'Métrica','y':'Valor (°)'},
                 title="Métricas sagital")

    return html.Div([
        dbc.Row([
            dbc.Col(html.Img(src=f"data:image/jpg;base64,{crop_b64}",
                             style={"width":"100%","maxWidth":"400px","borderRadius":"6px"}),
                    width=6),
            dbc.Col([
                html.H5("Métricas", className="text-white mb-2"),
                html.Div([card(k, v) for k, v in data.items()],
                         style={"display":"flex","flexWrap":"wrap"}),
                dcc.Graph(figure=fig)
            ], width=6)
        ], align="start"),
        html.Hr(className="border-secondary"),
        dbc.Row(dbc.Col(html.Img(src=f"data:image/jpg;base64,{vis_b64}",
                                 style={"width":"100%","maxWidth":"800px","borderRadius":"6px"}),
                        width={"size":8,"offset":2})),
        html.Div(create_zip(vis_b64, data, "sagittal_analysis.zip"), className="mt-3")
    ])


@app.callback(
    Output("out-front", "children"),
    Input("up-front", "contents"),
    State("up-front", "filename")
)
def analyze_front(contents, filename):
    if not contents:
        return ""
    if Path(filename).suffix.lower() not in ALLOWED:
        return dbc.Alert("Formato no soportado", color="danger")
    img = b64_to_cv2(contents)
    crop, vis, data = analyze_frontal(img)
    if crop is None:
        return dbc.Alert("No se detectó pose", color="warning")

    crop_b64 = cv2_to_b64(crop)
    vis_b64  = cv2_to_b64(vis)
    fig = px.bar(x=list(data.keys()), y=list(data.values()),
                 labels={'x':'Métrica','y':'Pixeles'},
                 title="Métricas frontal")

    return html.Div([
        dbc.Row([
            dbc.Col(html.Img(src=f"data:image/jpg;base64,{crop_b64}",
                             style={"width":"100%","maxWidth":"400px","borderRadius":"6px"}),
                    width=6),
            dbc.Col([
                html.H5("Métricas", className="text-white mb-2"),
                html.Div([card(k, v) for k, v in data.items()],
                         style={"display":"flex","flexWrap":"wrap"}),
                dcc.Graph(figure=fig)
            ], width=6)
        ], align="start"),
        html.Hr(className="border-secondary"),
        dbc.Row(dbc.Col(html.Img(src=f"data:image/jpg;base64,{vis_b64}",
                                 style={"width":"100%","maxWidth":"800px","borderRadius":"6px"}),
                        width={"size":8,"offset":2})),
        html.Div(create_zip(vis_b64, data, "frontal_analysis.zip"), className="mt-3")
    ])


def analyze_frontal(img):
    res = POSE.process(img)
    if not res.pose_landmarks:
        return None, None, None

    # 1) Recorte
    crop = crop_person(img, res.pose_landmarks.landmark)
    h, w = crop.shape[:2]
    lm = POSE.process(crop).pose_landmarks.landmark

    # 2) Coordenadas de puntos
    SHL = (int(lm[P.LEFT_SHOULDER].x*w),  int(lm[P.LEFT_SHOULDER].y*h))
    SHR = (int(lm[P.RIGHT_SHOULDER].x*w), int(lm[P.RIGHT_SHOULDER].y*h))
    LHL = (int(lm[P.LEFT_HIP].x*w),       int(lm[P.LEFT_HIP].y*h))
    RHL = (int(lm[P.RIGHT_HIP].x*w),      int(lm[P.RIGHT_HIP].y*h))
    LWL = (int(lm[P.LEFT_WRIST].x*w),     int(lm[P.LEFT_WRIST].y*h))
    RWL = (int(lm[P.RIGHT_WRIST].x*w),    int(lm[P.RIGHT_WRIST].y*h))
    LKL = (int(lm[P.LEFT_KNEE].x*w),      int(lm[P.LEFT_KNEE].y*h))
    RKL = (int(lm[P.RIGHT_KNEE].x*w),     int(lm[P.RIGHT_KNEE].y*h))
    LAL = (int(lm[P.LEFT_ANKLE].x*w),     int(lm[P.LEFT_ANKLE].y*h))
    RAL = (int(lm[P.RIGHT_ANKLE].x*w),    int(lm[P.RIGHT_ANKLE].y*h))

    vis = crop.copy()

    # 3) Dibujar líneas hombro→cadera→muñeca
    for a, b in [(SHL,LHL),(SHR,RHL),(SHL,LWL),(SHR,RWL)]:
        cv2.line(vis, a, b, IDEAL_RGBA[:3], 4)

    # 4) Dibujar líneas cadera→rodilla→tobillo
    for hip, knee, ankle in [(LHL,LKL,LAL),(RHL,RKL,RAL)]:
        cv2.line(vis, hip, knee, CLR_LINE, 6)
        cv2.line(vis, knee, ankle, CLR_LINE, 6)

    # 5) Dibujar puntos
    for p in (SHL,SHR,LHL,RHL,LWL,RWL,LKL,RKL,LAL,RAL):
        cv2.circle(vis, p, 8, CLR_PT, -1)

    # 6) Líneas horizontales de referencia
    for y in (SHL[1], SHR[1], LHL[1], RHL[1], LWL[1], RWL[1]):
        cv2.line(vis, (0,y), (w,y), CLR_LINE, 3)

    # 7) Cálculo de métricas
    D_rod = abs(RKL[0] - LKL[0])
    D_pie = abs(RAL[0] - LAL[0])

    data = {
        "Left Shoulder (px)":   SHL[1],
        "Right Shoulder (px)":  SHR[1],
        "Δ Shoulder (px)":      abs(SHR[1] - SHL[1]),
        "Left Hip (px)":        LHL[1],
        "Right Hip (px)":       RHL[1],
        "Δ Hip (px)":           abs(RHL[1] - LHL[1]),
        "Left Wrist (px)":      LWL[1],
        "Right Wrist (px)":     RWL[1],
        "Δ Wrist (px)":         abs(RWL[1] - LWL[1]),
        "Apertura rodillas":    D_rod,
        "Apertura pies":        D_pie
    }

    return crop, vis, data
# —————————————————————————————
# Función de descarga ZIP
# —————————————————————————————

def create_zip(img_b64: str, metrics_dict: dict, zip_filename: str) -> html.A:
    """
    Empaqueta una imagen (base64) y un Excel con las métricas
    en un ZIP que se puede descargar desde el navegador.
    """
    # Decodificar imagen
    img_bytes = base64.b64decode(img_b64)

    # Crear DataFrame y Excel en memoria
    df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Metrics")

    # Empaquetar en ZIP
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("analyzed_image.jpg", img_bytes)
        zf.writestr("metrics.xlsx", excel_buffer.getvalue())
    mem_zip.seek(0)
    zip_b64 = base64.b64encode(mem_zip.read()).decode()

    # Enlace de descarga
    return html.A(
        "⬇️ Descargar imagen + métricas (.zip)",
        href=f"data:application/zip;base64,{zip_b64}",
        download=zip_filename,
        className="btn btn-outline-info mt-2"
    )

# —————————————————————————————
# Ejecución
# —————————————————————————————
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
