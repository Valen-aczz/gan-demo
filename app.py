"""App de Streamlit para jugar con la GAN ya entrenada.   streamlit run app.py"""
import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from gan_models import Generator

st.set_page_config(page_title="GAN de imágenes sintéticas", page_icon="🌿", layout="wide")

def bonito(c):
    """Tomato___Early_blight -> Early blight (si no tiene ___ , se deja igual)."""
    return c.split("___")[-1].replace("_", " ")



NOMBRES_MODELO = {
    "faces_50ep": "Caras · 50 epochs (20.000 fotos)",
    "faces": "Caras · 40 epochs (60.000 fotos)",
    "plantvillage": "Hojas de tomate",
}


@st.cache_resource
def cargar(path):
    ck = torch.load(path, map_location="cpu")
    G = Generator(len(ck["classes"]), nz=ck["nz"])
    G.load_state_dict(ck["state"])
    G.eval()
    return G, ck["classes"], ck["epoch"]


def a_pil(x, escala=3):
    arr = ((x + 1) * 127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(a).resize((64 * escala, 64 * escala), Image.LANCZOS) for a in arr]


def generar(G, clase, n, seed, psi=1.0):
    g = torch.Generator().manual_seed(int(seed))
    z = torch.randn(n, G.nz, generator=g) * psi  # truncamiento: z más cerca del centro
    with torch.no_grad():
        return G(z, torch.full((n,), clase, dtype=torch.long)), z


st.title("🌿 Generador de imágenes con GAN")
st.caption("cDCGAN entrenada con bases de datos libres · cada imagen sale de un vector de ruido z")

runs = sorted(Path("runs").glob("*/generator.pt"))
if not runs:
    st.warning("Todavía no hay un modelo entrenado. Corre primero `python train_gan.py --dataset plantvillage --crop Tomato --epochs 100`.")
    st.stop()

ruta = st.sidebar.selectbox("Modelo entrenado", runs, format_func=lambda p: NOMBRES_MODELO.get(p.parent.name, p.parent.name))
G, clases, epoch = cargar(ruta)
st.sidebar.write(f"Entrenado por **{epoch}** epochs · **{len(clases)}** clases")
nombre = bonito
clase_txt = st.sidebar.selectbox("Clase a generar", clases, format_func=nombre)
clase = clases.index(clase_txt)
psi = st.sidebar.slider("Truncamiento ψ (menos = caras más limpias, más parecidas)", 0.3, 1.0, 0.8, 0.05)

tab1, tab2, tab3 = st.tabs(["Generar", "Interpolación en el espacio latente", "Cómo funciona"])

with tab1:
    c1, c2 = st.columns(2)
    n = c1.slider("Cantidad de imágenes", 1, 64, 16)
    seed = c2.number_input("Semilla (misma semilla = mismas imágenes)", 0, 10**6, 42)
    if st.button("Generar ✨", type="primary"):
        x, _ = generar(G, clase, n, seed, psi)
        imgs = a_pil(x)
        cols = st.columns(8)
        for i, im in enumerate(imgs):
            cols[i % 8].image(im, use_container_width=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i, im in enumerate(imgs):
                b = io.BytesIO(); im.save(b, "PNG"); zf.writestr(f"{clase_txt}_{i:03d}.png", b.getvalue())
        st.download_button("Descargar ZIP", buf.getvalue(), f"{clase_txt}_sinteticas.zip", "application/zip")

with tab2:
    st.write("Dos vectores z distintos y los puntos intermedios entre ellos: si el generador aprendió "
             "bien, la imagen cambia de forma suave (no a saltos).")
    c1, c2, c3 = st.columns(3)
    s1 = c1.number_input("Semilla A", 0, 10**6, 1)
    s2 = c2.number_input("Semilla B", 0, 10**6, 2)
    pasos = c3.slider("Pasos", 3, 12, 8)
    za = torch.randn(1, G.nz, generator=torch.Generator().manual_seed(int(s1))) * psi
    zb = torch.randn(1, G.nz, generator=torch.Generator().manual_seed(int(s2))) * psi
    t = torch.linspace(0, 1, pasos).unsqueeze(1)
    with torch.no_grad():
        x = G((1 - t) * za + t * zb, torch.full((pasos,), clase, dtype=torch.long))
    cols = st.columns(pasos)
    for col, im in zip(cols, a_pil(x, 2)):
        col.image(im, use_container_width=True)

with tab3:
    st.markdown("**El juego:** el generador G convierte ruido z en una imagen falsa; el discriminador D "
                "intenta distinguir imágenes reales de falsas. Ambos mejoran peleando.")
    st.latex(r"\min_G \max_D V(D,G)=\mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]")
    st.markdown("G y D además reciben una **etiqueta de clase** (GAN condicional): con varias clases puedes pedir una "
                "concreta; con una sola (como las caras) funciona como una GAN normal. "
                "Para G se usa el costo no saturante y para D, one-sided label smoothing (0,9).")
    st.info("⚠️ Todo lo que ves es sintético: lo inventó una red neuronal a partir de ruido, "
            "no corresponde a ninguna foto ni persona real.")
