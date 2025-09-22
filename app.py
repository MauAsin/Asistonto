# Cabrito Hoops — StatSheets (Streamlit) v0.1
# Autor: ChatGPT (para Mauricio)
# Descripción: MVP para analizar hojas de estadística de baloncesto (box score por juego)
# Stack: Streamlit + Pandas + Plotly
# Objetivo: Subir CSV/Excel, mapear columnas, calcular métricas avanzadas y visualizar por jugador/equipo/lineup.

import io
import math
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Cabrito Hoops — StatSheets",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Utilidades
# ---------------------------

def _safe_div(a, b):
    try:
        return a / b if b else 0.0
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False)
def read_any(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    # Try as CSV fallback
    data = file.read()
    return pd.read_csv(io.BytesIO(data))

# ---------------------------
# Esquema estándar y mapeo de columnas
# ---------------------------

STANDARD = {
    "player": "Nombre del jugador",
    "team": "Equipo",
    "opponent": "Rival (opcional)",
    "game_id": "ID de juego (opcional)",
    "date": "Fecha (opcional)",
    "min": "Minutos (MIN)",
    "fgm": "Tiros de campo anotados (FGM)",
    "fga": "Tiros de campo intentados (FGA)",
    "tpm": "Triples anotados (3PM)",
    "tpa": "Triples intentados (3PA)",
    "ftm": "Libres anotados (FTM)",
    "fta": "Libres intentados (FTA)",
    "orb": "Rebotes ofensivos (OREB)",
    "drb": "Rebotes defensivos (DREB)",
    "ast": "Asistencias (AST)",
    "tov": "Pérdidas (TOV)",
    "stl": "Robos (STL)",
    "blk": "Tapones (BLK)",
    "pf": "Faltas (PF)",
    "pts": "Puntos (PTS)",
    "+/-": "+/- (opcional)",
    "period": "Periodo/Cuarto (opcional)",
    "lineup": "Lineup/quinteto (opcional)",
    "x": "Coordenada tiro X (opcional)",
    "y": "Coordenada tiro Y (opcional)",
}

NUMERIC_FIELDS = [
    "min","fgm","fga","tpm","tpa","ftm","fta","orb","drb","ast","tov","stl","blk","pf","pts","+/-","x","y"
]

# ---------------------------
# Sidebar — Carga y mapeo
# ---------------------------

st.sidebar.title("🏀 Cabrito Hoops — StatSheets")
st.sidebar.caption("MVP para entrenadores. Sube tu box score por juego y mapea columnas.")

sample_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
st.sidebar.markdown(
    "- **Formato recomendado:** 1 fila por jugador y juego.\n"
    "- **Soportado:** CSV / Excel.\n"
    "- **Opcionales:** rival, fecha, periodo, lineup, x/y para shot chart."
)

file = st.sidebar.file_uploader("Sube tu archivo de estadística", type=["csv","xlsx","xls"]) 
raw = read_any(file)

if raw.empty:
    st.info("Sube un archivo para comenzar. Puedes usar tu box score exportado de FIBA LiveStats, Genius, HUDL Assist, etc.")
    st.stop()

st.sidebar.subheader("Mapeo de columnas → Esquema estándar")

mappings: Dict[str, Optional[str]] = {}
cols = [None] + list(raw.columns)

with st.sidebar.expander("Mapeo rápido", expanded=True):
    for std_key, label in STANDARD.items():
        default_guess = None
        # heurísticas simples
        guess_candidates = {
            "player": ["player","jugador","nombre","name"],
            "team": ["team","equipo","club"],
            "opponent": ["opponent","rival","contrario","opp"],
            "game_id": ["game","game_id","id","match_id"],
            "date": ["date","fecha"],
            "min": ["min","mins","minutes"],
            "fgm": ["fgm","fg made","tc anotados","tc a"],
            "fga": ["fga","fg att","tc intentados","tc i"],
            "tpm": ["3pm","tpm","tp anotados","3p made"],
            "tpa": ["3pa","tpa","tp intentados","3p att"],
            "ftm": ["ftm","libres anotados","ft made"],
            "fta": ["fta","libres intentados","ft att"],
            "orb": ["orb","oreb","off reb"],
            "drb": ["drb","dreb","def reb"],
            "ast": ["ast","asist"],
            "tov": ["tov","turnovers","perdidas"],
            "stl": ["stl","robos"],
            "blk": ["blk","tapones","bloq"],
            "pf": ["pf","faltas"],
            "pts": ["pts","puntos","points"],
            "+/-": ["+/-","plusminus","plus_minus"],
            "period": ["period","cuarto","qtr","quarter"],
            "lineup": ["lineup","unit","quinteto"],
            "x": ["x","loc_x","shot_x"],
            "y": ["y","loc_y","shot_y"],
        }.get(std_key, [])
        for c in raw.columns:
            if c.lower() in guess_candidates:
                default_guess = c
                break
        mappings[std_key] = st.selectbox(label, options=cols, index=(cols.index(default_guess) if default_guess in cols else 0), key=f"map_{std_key}")

# aplicar mapeo
ren = {m: k for k, m in mappings.items() if m}
df = raw.rename(columns=ren).copy()

# Tipar columnas numéricas
for c in NUMERIC_FIELDS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Limpiar strings claves
for c in ["player","team","opponent","game_id","date","period","lineup"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# ---------------------------
# Filtros
# ---------------------------

st.sidebar.subheader("Filtros")

f_team = st.sidebar.multiselect("Equipo(s)", sorted(df["team"].dropna().unique()) if "team" in df else [], default=None)
f_opp = st.sidebar.multiselect("Rival(es)", sorted(df["opponent"].dropna().unique()) if "opponent" in df else [], default=None)
f_game = st.sidebar.multiselect("Juego(s)/ID", sorted(df["game_id"].dropna().unique()) if "game_id" in df else [], default=None)
f_period = st.sidebar.multiselect("Periodo(s)", sorted(df["period"].dropna().unique()) if "period" in df else [], default=None)

mask = pd.Series([True]*len(df))
if f_team and "team" in df:
    mask &= df["team"].isin(f_team)
if f_opp and "opponent" in df:
    mask &= df["opponent"].isin(f_opp)
if f_game and "game_id" in df:
    mask &= df["game_id"].isin(f_game)
if f_period and "period" in df:
    mask &= df["period"].isin(f_period)

fdf = df[mask].copy()

# ---------------------------
# Métricas avanzadas
# ---------------------------

def calc_team_totals(frame: pd.DataFrame) -> pd.Series:
    s = pd.Series(dtype=float)
    for c in ["fgm","fga","tpm","tpa","ftm","fta","orb","drb","ast","tov","stl","blk","pf","pts"]:
        if c in frame:
            s[c] = frame[c].sum()
        else:
            s[c] = 0.0
    # posesiones estimadas (Dean Oliver)
    poss = s.get("fga",0) - s.get("orb",0) + s.get("tov",0) + 0.44*s.get("fta",0)
    s["poss"] = poss
    s["efg"] = _safe_div(s.get("fgm",0) + 0.5*s.get("tpm",0), s.get("fga",0))
    tsa = s.get("fga",0) + 0.44*s.get("fta",0)
    s["ts"] = _safe_div(s.get("pts",0), 2*tsa) if tsa else 0
    s["tov_rate"] = _safe_div(s.get("tov",0), s.get("poss",0))
    s["orb_rate"] = _safe_div(s.get("orb",0), s.get("orb",0) + s.get("drb",0))  # sin rivales
    s["ast_rate"] = _safe_div(s.get("ast",0), s.get("fgm",0))
    return s

@st.cache_data(show_spinner=False)
def compute_player_adv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    out["eFG%"] = np.where(out.get("fga")>0, (out.get("fgm",0)+0.5*out.get("tpm",0))/out.get("fga",1), 0)
    tsa = out.get("fga",0) + 0.44*out.get("fta",0)
    out["TS%"] = np.where(tsa>0, out.get("pts",0)/(2*tsa), 0)
    out["AST/TO"] = np.where(out.get("tov",0)>0, out.get("ast",0)/out.get("tov",1), np.where(out.get("ast",0)>0, np.inf, 0))
    out["REB"] = out.get("orb",0) + out.get("drb",0)
    out["STOCKS"] = out.get("stl",0) + out.get("blk",0)
    # posesiones individuales aproximadas
    out["Poss_est"] = out.get("fga",0) - out.get("orb",0) + out.get("tov",0) + 0.44*out.get("fta",0)
    # por 100 posesiones
    for col in ["pts","ast","reb","stl","blk","tov"]:
        base = {
            "reb": out.get("REB",0),
            "stl": out.get("stl",0),
            "blk": out.get("blk",0),
            "tov": out.get("tov",0),
        }.get(col, out.get(col,0))
        out[f"{col.upper()} per100"] = np.where(out["Poss_est"]>0, base/out["Poss_est"]*100, 0)
    return out

# Totales del marco filtrado
team_tot = calc_team_totals(fdf)

# ---------------------------
# Layout principal
# ---------------------------

st.markdown("""
# 🏀 Cabrito Hoops — StatSheets
**MVP (v0.1)** para analizar hojas de estadísticas. Carga tu archivo, mapea columnas y explora KPIs, rankings y visualizaciones.
""")

kpi_cols = st.columns(6)

kpi_cols[0].metric("PTS", f"{team_tot.get('pts',0):.0f}")
kpi_cols[1].metric("eFG%", f"{team_tot.get('efg',0)*100:.1f}%")
kpi_cols[2].metric("TS%", f"{team_tot.get('ts',0)*100:.1f}%")
kpi_cols[3].metric("AST", f"{team_tot.get('ast',0):.0f}")
kpi_cols[4].metric("TOV", f"{team_tot.get('tov',0):.0f}")
kpi_cols[5].metric("Poss (est)", f"{team_tot.get('poss',0):.0f}")

st.divider()

# Tabs
TAB1, TAB2, TAB3, TAB4 = st.tabs(["📊 Jugadores","📈 Tiro","🧩 Lineups","📥 Datos"]) 

with TAB1:
    st.subheader("Ranking de jugadores")
    if "player" not in fdf:
        st.warning("Necesitas mapear la columna de **jugador** para ver este ranking.")
    else:
        players = compute_player_adv(fdf)
        # Selección de métrica
        metric = st.selectbox("Métrica a ordenar", [
            "PTS","REB","AST","STL","BLK","TS%","eFG%","AST/TO","+/-" if "+/-" in players else None,
        ])
        if metric is None:
            metric = "PTS"
        agg_cols = {c:"sum" for c in ["pts","orb","drb","ast","stl","blk","tov","REB","STOCKS","Poss_est"] if c in players}
        agg_cols.update({"eFG%":"mean","TS%":"mean","AST/TO":"mean"})
        if "+/-" in players:
            agg_cols["+/-"] = "sum"
        rank = players.groupby(["player","team"], dropna=False).agg(agg_cols).reset_index()
        # renombrar para mostrar bonito
        show = rank.rename(columns={
            "pts":"PTS","orb":"OREB","drb":"DREB","ast":"AST","stl":"STL","blk":"BLK","tov":"TOV"
        })
        show = show.sort_values(metric, ascending=False)
        st.dataframe(show.head(30), use_container_width=True)
        # gráfico
        topN = st.slider("Top N", 5, 30, 10)
        fig = px.bar(show.head(topN), x="player", y=metric, color="team", hover_data=show.columns)
        fig.update_layout(xaxis_title="Jugador", yaxis_title=metric, bargap=0.2, height=450)
        st.plotly_chart(fig, use_container_width=True)

with TAB2:
    st.subheader("Shot chart")
    needed = ["x","y"]
    if not all(c in fdf for c in needed):
        st.info("Para ver el shot chart, mapea columnas X/Y de cada intento de tiro. Si no cuentas con trazas de tiro, omite esta sección.")
    else:
        # Filtro adicional por jugador
        who = st.multiselect("Jugador(es)", sorted(fdf["player"].dropna().unique()) if "player" in fdf else [])
        sdf = fdf.copy()
        if who and "player" in sdf:
            sdf = sdf[sdf["player"].isin(who)]
        # Asumimos coordenadas en un sistema de 0..50 (ancho) x 0..94 (largo) o similar; normalizamos a half-court
        # Normalización simple a [0,50]x[0,47]
        xs = sdf["x"].astype(float)
        ys = sdf["y"].astype(float)
        xs_n = (xs - xs.min()) / max(1e-6, (xs.max() - xs.min())) * 50
        ys_n = (ys - ys.min()) / max(1e-6, (ys.max() - ys.min())) * 47
        base = pd.DataFrame({"x": xs_n, "y": ys_n})
        fig = px.density_heatmap(base, x="x", y="y", nbinsx=30, nbinsy=28, histfunc="count")
        fig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=520, title="Densidad de tiros (normalizado)"
        )
        st.plotly_chart(fig, use_container_width=True)

with TAB3:
    st.subheader("Lineups y Net Rating (aprox.)")
    if "lineup" not in fdf:
        st.info("Para ver lineups, mapea la columna **lineup/quinteto**. Puedes usar formatos como 'Player1|Player2|...'.")
    else:
        group_cols = ["lineup"]
        if "team" in fdf:
            group_cols.insert(0, "team")
        g = fdf.groupby(group_cols).agg({"pts":"sum","Poss_est":"sum"} if "Poss_est" in fdf else {"pts":"sum"})
        g = g.reset_index()
        if "Poss_est" in g:
            g["ORtg"] = np.where(g["Poss_est"]>0, g["pts"] / g["Poss_est"] * 100, 0)
        else:
            g["ORtg"] = g["pts"]  # fallback pobre
        # Sin defensa por falta de rivales/on-off, mostramos solo ofensiva
        g = g.sort_values("ORtg", ascending=False)
        st.dataframe(g.head(20), use_container_width=True)
        fig = px.bar(g.head(15), x="lineup", y="ORtg", color="team" if "team" in g else None)
        fig.update_layout(height=450, xaxis_title="Lineup", yaxis_title="ORtg (por 100 poss)")
        st.plotly_chart(fig, use_container_width=True)

with TAB4:
    st.subheader("Vista de datos y descarga")
    st.dataframe(fdf, use_container_width=True)
    def to_csv_bytes(df_: pd.DataFrame) -> bytes:
        return df_.to_csv(index=False).encode("utf-8")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Descargar jugadores (agregados)", data=to_csv_bytes(
        compute_player_adv(fdf).groupby(["player","team"], dropna=False).sum(numeric_only=True).reset_index()
    ), file_name="players_agg.csv", mime="text/csv")
    c2.download_button("⬇️ Descargar dataset filtrado", data=to_csv_bytes(fdf), file_name="dataset_filtrado.csv", mime="text/csv")

st.caption("v0.1 — Este MVP calcula eFG%, TS%, posesiones estimadas, rankings de jugadores, densidad de tiros (si hay X/Y) y una vista simple de lineups. Próximas versiones: on/off, defensa, play-by-play, sincronía con video.")
