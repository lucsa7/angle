import base64, cv2, mediapipe as mp, numpy as np, tempfile
from pathlib import Path
import dash, dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from flask import Flask
import zipfile, io
import pandas as pd
import os


TMP_DIR = Path(tempfile.gettempdir()) / "ohs_tmp"; TMP_DIR.mkdir(exist_ok=True)
ALLOWED = {".jpg", ".jpeg", ".png"}
POSE = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
SEG = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
P = mp.solutions.pose.PoseLandmark

CLR_LINE, CLR_PT = (0, 230, 127), (250, 250, 250)
IDEAL_RGBA = (0, 255, 0, 110)

def b64_to_cv2(content):
    _, b64 = content.split(",", 1)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def cv2_to_b64(img):
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buf).decode()

def ang(a, b, c):
    v1, v2 = np.array(a) - b, np.array(c) - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def line(im, a, b): cv2.line(im, a, b, CLR_LINE, 6, cv2.LINE_AA)
def circ(im, p):    cv2.circle(im, p, 7, CLR_PT, -1, cv2.LINE_AA)

def crop_person(img, lm):
    H, W = img.shape[:2]
    pts = np.array([(p.x * W, p.y * H) for p in lm])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    pad = 0.20
    x0 = max(0, int(x0 - pad * (x1 - x0)))
    y0 = max(0, int(y0 - pad * (y1 - y0)))
    x1 = min(W, int(x1 + pad * (x1 - x0)))
    y1 = min(H, int(y1 + (pad + 0.1) * (y1 - y0)))
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (480, int(480 * crop.shape[0] / crop.shape[1])))

def card(var, val):
    unit = "°" if isinstance(val, (int, float)) else ""
    return dbc.Card(
        dbc.CardBody([
            html.Small(var, className="text-muted"),
            html.H4(f"{val}{unit}", className="mb-0")
        ]),
        color="dark", outline=True,
        className="m-1 p-2", style={"minWidth": "120px", "textAlign": "center"}
    )
def ankle_df(kn, an, heel, toe):
    vals = []
    for pt in (heel, toe):
        a = ang(kn, an, pt) - 90
        if 0 <= a <= 90: vals.append(a)
    return round(sum(vals) / len(vals), 1) if vals else 0

def analyze_sagital(img):
    res = POSE.process(img)
    lm = res.pose_landmarks.landmark
    crop = crop_person(img, lm)
    h, w = crop.shape[:2]
    lm = POSE.process(crop).pose_landmarks.landmark
    side = "R" if lm[P.RIGHT_HIP].visibility >= lm[P.LEFT_HIP].visibility else "L"
    gid = lambda L, R: R if side == "R" else L
    ids = [gid(getattr(P, "LEFT_" + n), getattr(P, "RIGHT_" + n))
           for n in ("SHOULDER", "HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX", "WRIST")]
    SH, HI, KN, AN, HE, FT, WR = ids
    pt = lambda i: (int(lm[i].x * w), int(lm[i].y * h))
    SHp, HIp, KNp, ANp, HEp, FTp, WRp = map(pt, ids)

    data = {
        "Knee flex": round(ang(HIp, KNp, ANp), 1),
        "Hip flex": round(ang(SHp, HIp, KNp), 1),
        "Ankle DF": ankle_df(KNp, ANp, HEp, FTp),
        "Shoulder flex": round(ang(HIp, SHp, WRp), 1),
        "|Trunk-Tibia|": round(abs(ang(SHp, HIp, KNp) - ang(KNp, ANp, HIp)), 1)
    }

    mask = (SEG.process(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)).segmentation_mask > .6)[:, :, None]
    blur = cv2.GaussianBlur(crop, (17, 17), 0)
    vis = (crop * mask + blur * (1 - mask)).astype(np.uint8)

    for a, b in [(SHp, HIp), (HIp, KNp), (KNp, ANp), (SHp, WRp)]: line(vis, a, b)
    for p in (SHp, HIp, KNp, ANp, FTp, WRp): circ(vis, p)

    return crop, vis, data

def analyze_frontal(img):
    res = POSE.process(img)
    if not res.pose_landmarks:
        return None, None, None
    lm = res.pose_landmarks.landmark
    crop = crop_person(img, lm)
    h, w = crop.shape[:2]
    lm = POSE.process(crop).pose_landmarks.landmark
    toP = lambda m: (int(m.x * w), int(m.y * h))
    SHL, SHR = toP(lm[P.LEFT_SHOULDER]), toP(lm[P.RIGHT_SHOULDER])
    LHL, RHL = toP(lm[P.LEFT_HIP]), toP(lm[P.RIGHT_HIP])
    LWL, RWL = toP(lm[P.LEFT_WRIST]), toP(lm[P.RIGHT_WRIST])
    vis = crop.copy()

    for a, b in [(SHL, LHL), (SHR, RHL), (SHL, LWL), (SHR, RWL)]:
        cv2.line(vis, a, b, IDEAL_RGBA[:3], 4, cv2.LINE_AA)
        cv2.circle(vis, a, 8, CLR_PT, -1, cv2.LINE_AA)
        cv2.circle(vis, b, 8, CLR_PT, -1, cv2.LINE_AA)
    for y in (SHL[1], SHR[1], LHL[1], RHL[1], LWL[1], RWL[1]):
        cv2.line(vis, (0, y), (w, y), CLR_LINE, 3, cv2.LINE_AA)

    data = {
        "Hombro izq": SHL[1], "Hombro der": SHR[1], "Δ Hombro (px)": SHR[1] - SHL[1],
        "Cadera izq": LHL[1], "Cadera der": RHL[1], "Δ Cadera (px)": RHL[1] - LHL[1],
        "Muñeca izq": LWL[1], "Muñeca der": RWL[1], "Δ Muñeca (px)": RWL[1] - LWL[1]
    }
    return crop, vis, data
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.CYBORG])
app.title = "OHS Analyzer - Lift Style"

app.layout = dbc.Container([
    # — Logo centrado —
    dbc.Row(
        dbc.Col(
            html.Img(src="/assets/angle.png",
                     style={"height": "60px", "display":"block", "margin":"0 auto 20px"}),
            width=12
        )
    ),

    html.H2("Overhead-Squat Analyzer", className="text-center text-info mb-4"),

    dbc.Row([
        # Columna subida sagital
        dbc.Col([
            html.H4("🦵 Sagital View", className="text-white"),
            dcc.Upload(...),
            dbc.Spinner(html.Div(id="out-sag"))
        ], md=6),
        # Columna subida frontal
        dbc.Col([
            html.H4("🧍 Frontal View", className="text-white"),
            dcc.Upload(...),
            dbc.Spinner(html.Div(id="out-front"))
        ], md=6),
    ], className="mb-4"),

    html.Footer("Powered by STB • Luciano Sacaba", className="text-center mt-4 text-muted")
], fluid=True, className="p-4 bg-dark")


@app.callback(
    Output("out-sag", "children"),
    Input("up-sag", "contents"),
    State("up-sag", "filename")
)
def analyze_sag(contents, name):
    if not contents:
        return ""
    if Path(name).suffix.lower() not in ALLOWED:
        return dbc.Alert("Formato no soportado", color="danger")

    img = b64_to_cv2(contents)
    crop, vis, data = analyze_sagital(img)
    vis_b64 = cv2_to_b64(vis)
    cards = [card(k, v) for k, v in data.items()]

    return html.Div([
        # Imagen + métricas en la misma fila
        dbc.Row([
            # Imagen original recortada
            dbc.Col(
                html.Img(src="data:image/jpg;base64," + cv2_to_b64(crop),
                         style={"width": "100%", "maxWidth": "400px", "borderRadius": "6px"}),
                width=6
            ),
            # Métricas
            dbc.Col([
                html.H5("Métricas", className="text-white"),
                dbc.Row(cards, className="flex-column")
            ], width=6, style={"maxHeight": "200px", "overflowY": "auto"})
        ], align="start"),

        html.Hr(className="border-secondary"),

        # Imagen con contornos
        dbc.Row(
            dbc.Col(
                html.Img(src="data:image/jpg;base64," + vis_b64,
                         style={"width": "100%", "maxWidth": "800px", "borderRadius": "6px"}),
                width={"size": 8, "offset": 2}
            )
        ),

        html.Div(create_zip(vis_b64, data, "analisis_sagital.zip"), className="mt-3")
    ])


@app.callback(
    Output("out-front", "children"),
    Input("up-front", "contents"),
    State("up-front", "filename")
)
def analyze_front(contents, name):
    # ... validaciones idénticas ...
    crop, vis, data = analyze_frontal(img)
    vis_b64 = cv2_to_b64(vis)
    cards = [card(k, v) for k, v in data.items()]

    return html.Div([
        dbc.Row([
            dbc.Col(
                html.Img(src="data:image/jpg;base64," + cv2_to_b64(crop),
                         style={"width": "100%", "maxWidth": "400px", "borderRadius": "6px"}),
                width=6
            ),
            dbc.Col([
                html.H5("Métricas", className="text-white"),
                dbc.Row(cards, className="flex-column")
            ], width=6, style={"maxHeight": "400px", "overflowY": "auto"})
        ], align="start"),

        html.Hr(className="border-secondary"),

        dbc.Row(
            dbc.Col(
                html.Img(src="data:image/jpg;base64," + vis_b64,
                         style={"width": "100%", "maxWidth": "800px", "borderRadius": "6px"}),
                width={"size": 8, "offset": 2}
            )
        ),

        html.Div(create_zip(vis_b64, data, "analisis_frontal.zip"), className="mt-3")
    ])



def get_download_link(img_b64, filename):
    return html.A("⬇️ Descargar imagen", href="data:image/jpg;base64," + img_b64,
                  download=filename, target="_blank", className="btn btn-outline-light mt-2")


def create_zip(img_b64, metrics_dict, zip_filename):
    img_bytes = base64.b64decode(img_b64)

    # Crear Excel en memoria
    excel_buffer = io.BytesIO()
    df = pd.DataFrame(list(metrics_dict.items()), columns=["Métrica", "Valor"])
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Métricas')
    excel_bytes = excel_buffer.getvalue()

    # Crear ZIP en memoria
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imagen_analizada.jpg", img_bytes)
        zf.writestr("metricas.xlsx", excel_bytes)
    mem_zip.seek(0)
    zip_b64 = base64.b64encode(mem_zip.read()).decode()

    return html.A("⬇️ Descargar imagen + métricas (.zip)",
                  href="data:application/zip;base64," + zip_b64,
                  download=zip_filename, target="_blank",
                  className="btn btn-outline-info mt-2")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)





