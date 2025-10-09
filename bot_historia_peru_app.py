# bot_historia_peru_app.py
# -*- coding: utf-8 -*-
"""
🇵🇪 Bot de Historia del Perú
Versión final – Streamlit Cloud Edition
Desarrollado por: Yeltsin Solano Díaz
"""

import json
import unicodedata
from typing import List, Dict, Tuple

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 📘 DATASET BASE — 30 Preguntas con referencias
# =====================================================
DATASET_30: List[Dict[str, object]] = [
    {"pregunta": "primer presidente del peru",
     "respuesta": "El primer presidente del Perú fue José de la Riva-Agüero en 1823.",
     "fuentes": [{"titulo": "José de la Riva-Agüero", "url": "https://es.wikipedia.org/wiki/Jos%C3%A9_de_la_Riva-Ag%C3%BCero_y_S%C3%A1nchez_Boquete"}]},
    {"pregunta": "independencia del peru",
     "respuesta": "La independencia del Perú fue proclamada el 28 de julio de 1821 por José de San Martín.",
     "fuentes": [{"titulo": "Independencia del Perú", "url": "https://es.wikipedia.org/wiki/Independencia_del_Per%C3%BA"},
                 {"titulo": "BNP: Independencia del Perú", "url": "https://www.bnp.gob.pe/independencia-del-peru/"}]},
    {"pregunta": "batalla de ayacucho",
     "respuesta": "La Batalla de Ayacucho se libró el 9 de diciembre de 1824 y selló la independencia sudamericana.",
     "fuentes": [{"titulo": "Batalla de Ayacucho", "url": "https://es.wikipedia.org/wiki/Batalla_de_Ayacucho"}]},
    {"pregunta": "culturas preincas",
     "respuesta": "Algunas culturas preincas fueron Chavín, Paracas, Nazca, Moche, Recuay y Wari.",
     "fuentes": [{"titulo": "Cultura Chavín", "url": "https://es.wikipedia.org/wiki/Cultura_Chav%C3%ADn"},
                 {"titulo": "Cultura Moche", "url": "https://es.wikipedia.org/wiki/Cultura_Moche"},
                 {"titulo": "Cultura Nazca", "url": "https://es.wikipedia.org/wiki/Cultura_Nazca"},
                 {"titulo": "Cultura Paracas", "url": "https://es.wikipedia.org/wiki/Cultura_Paracas"},
                 {"titulo": "Cultura Wari", "url": "https://es.wikipedia.org/wiki/Cultura_Wari"}]},
    {"pregunta": "civilizacion caral",
     "respuesta": "Caral es una de las civilizaciones más antiguas de América, desarrollada entre 3000 y 1800 a.C. en el valle de Supe.",
     "fuentes": [{"titulo": "Caral-Supe", "url": "https://es.wikipedia.org/wiki/Caral-Supe"},
                 {"titulo": "Zona Arqueológica Caral (oficial)", "url": "https://www.zonacaral.gob.pe/"}]},
    {"pregunta": "quien proclamo la independencia del peru",
     "respuesta": "José de San Martín proclamó la independencia del Perú en Lima el 28 de julio de 1821.",
     "fuentes": [{"titulo": "José de San Martín", "url": "https://es.wikipedia.org/wiki/Jos%C3%A9_de_San_Mart%C3%ADn"}]},
    {"pregunta": "quien fue simon bolivar",
     "respuesta": "Simón Bolívar fue un libertador que culminó la independencia del Perú tras las batallas de Junín y Ayacucho.",
     "fuentes": [{"titulo": "Simón Bolívar", "url": "https://es.wikipedia.org/wiki/Sim%C3%B3n_Bol%C3%ADvar"}]},
    {"pregunta": "batalla de junin",
     "respuesta": "La Batalla de Junín se libró el 6 de agosto de 1824 y fue una victoria patriota antes de Ayacucho.",
     "fuentes": [{"titulo": "Batalla de Junín", "url": "https://es.wikipedia.org/wiki/Batalla_de_Jun%C3%ADn"}]},
    {"pregunta": "guerra del pacifico",
     "respuesta": "La Guerra del Pacífico se desarrolló entre 1879 y 1884 entre Perú, Bolivia y Chile.",
     "fuentes": [{"titulo": "Guerra del Pacífico", "url": "https://es.wikipedia.org/wiki/Guerra_del_Pac%C3%ADfico"}]},
    {"pregunta": "fundacion de lima",
     "respuesta": "Lima fue fundada el 18 de enero de 1535 por Francisco Pizarro con el nombre de Ciudad de los Reyes.",
     "fuentes": [{"titulo": "Lima (fundación)", "url": "https://es.wikipedia.org/wiki/Lima#Historia"}]},
    {"pregunta": "quien fue pachacutec",
     "respuesta": "Pachacútec fue el noveno gobernante inca que expandió el Tahuantinsuyo y reformó el Cusco.",
     "fuentes": [{"titulo": "Pachacútec", "url": "https://es.wikipedia.org/wiki/Pachac%C3%BAtec"}]},
    {"pregunta": "que fue el tahuantinsuyo",
     "respuesta": "El Tahuantinsuyo fue el imperio inca dividido en cuatro suyos: Chinchaysuyo, Collasuyo, Antisuyo y Cuntisuyo.",
     "fuentes": [{"titulo": "Imperio inca", "url": "https://es.wikipedia.org/wiki/Imperio_inca"}]},
    {"pregunta": "rebelion de tupac amaru ii",
     "respuesta": "La rebelión de Túpac Amaru II se inició en 1780 como levantamiento indígena contra los abusos coloniales.",
     "fuentes": [{"titulo": "Túpac Amaru II", "url": "https://es.wikipedia.org/wiki/T%C3%BApac_Amaru_II"}]},
    {"pregunta": "quien fue jose de la mar",
     "respuesta": "José de La Mar fue presidente del Perú entre 1827 y 1829 y participó en la independencia.",
     "fuentes": [{"titulo": "José de La Mar", "url": "https://es.wikipedia.org/wiki/Jos%C3%A9_de_La_Mar"}]},
    {"pregunta": "reformas del virrey toledo",
     "respuesta": "Francisco de Toledo reorganizó la administración virreinal, reducciones y la mita minera.",
     "fuentes": [{"titulo": "Francisco de Toledo (virrey)", "url": "https://es.wikipedia.org/wiki/Francisco_de_Toledo_(virrey)"}]},
    {"pregunta": "batalla de arica",
     "respuesta": "La Batalla de Arica se libró el 7 de junio de 1880; destaca el sacrificio del coronel Francisco Bolognesi.",
     "fuentes": [{"titulo": "Batalla de Arica", "url": "https://es.wikipedia.org/wiki/Batalla_de_Arica"}]},
    {"pregunta": "quien fue miguel grau",
     "respuesta": "Miguel Grau fue el 'Caballero de los Mares', héroe naval peruano caído en Angamos (1879).",
     "fuentes": [{"titulo": "Miguel Grau", "url": "https://es.wikipedia.org/wiki/Miguel_Grau"}]},
    {"pregunta": "quien fue ramon castilla",
     "respuesta": "Ramón Castilla fue presidente que abolió la esclavitud y modernizó el Estado en el siglo XIX.",
     "fuentes": [{"titulo": "Ramón Castilla", "url": "https://es.wikipedia.org/wiki/Ram%C3%B3n_Castilla"}]},
    {"pregunta": "constitucion de 1823",
     "respuesta": "La Constitución de 1823 fue la primera del Perú y estableció la forma republicana de gobierno.",
     "fuentes": [{"titulo": "Constitución de 1823 (Perú)", "url": "https://es.wikipedia.org/wiki/Constituci%C3%B3n_Peruana_de_1823"}]},
    {"pregunta": "constitucion de 1993",
     "respuesta": "La Constitución de 1993 fue promulgada durante el gobierno de Alberto Fujimori y está vigente.",
     "fuentes": [{"titulo": "Constitución Política del Perú de 1993", "url": "https://es.wikipedia.org/wiki/Constituci%C3%B3n_Pol%C3%ADtica_del_Per%C3%BA_de_1993"}]},
    {"pregunta": "quien fue jose olaya",
     "respuesta": "José Olaya fue un mártir de la independencia, mensajero clandestino, ejecutado en 1823.",
     "fuentes": [{"titulo": "José Olaya", "url": "https://es.wikipedia.org/wiki/Jos%C3%A9_Olaya"}]},
    {"pregunta": "quien fue maria parrado de bellido",
     "respuesta": "María Parado de Bellido fue heroína de la independencia, fusilada por los realistas en 1822.",
     "fuentes": [{"titulo": "María Parado de Bellido", "url": "https://es.wikipedia.org/wiki/Mar%C3%ADa_Parado_de_Bellido"}]},
    {"pregunta": "quien fue francisco bolognesi",
     "respuesta": "Francisco Bolognesi defendió el Morro de Arica en 1880 y es símbolo del deber y honor militar.",
     "fuentes": [{"titulo": "Francisco Bolognesi", "url": "https://es.wikipedia.org/wiki/Francisco_Bolognesi"}]},
    {"pregunta": "quien fue ricardo palma",
     "respuesta": "Ricardo Palma fue escritor peruano, autor de las 'Tradiciones Peruanas'.",
     "fuentes": [{"titulo": "Ricardo Palma", "url": "https://es.wikipedia.org/wiki/Ricardo_Palma"}]},
    {"pregunta": "imperio wari",
     "respuesta": "El Imperio Wari se desarrolló entre los siglos VII y XIII en la región de Ayacucho.",
     "fuentes": [{"titulo": "Cultura Wari", "url": "https://es.wikipedia.org/wiki/Cultura_Wari"}]},
    {"pregunta": "quien fue jose carlos mariategui",
     "respuesta": "José Carlos Mariátegui fue un pensador marxista peruano, autor de 'Siete ensayos...'.",
     "fuentes": [{"titulo": "José Carlos Mariátegui", "url": "https://es.wikipedia.org/wiki/Jos%C3%A9_Carlos_Mari%C3%A1tegui"}]},
    {"pregunta": "batalla de tarapaca",
     "respuesta": "La Batalla de Tarapacá se libró el 27 de noviembre de 1879; fue una victoria peruana.",
     "fuentes": [{"titulo": "Batalla de Tarapacá", "url": "https://es.wikipedia.org/wiki/Batalla_de_Tarapac%C3%A1_(1879)"}]},
    {"pregunta": "quien fue juan santos atahualpa",
     "respuesta": "Juan Santos Atahualpa lideró una rebelión en la selva central en el siglo XVIII.",
     "fuentes": [{"titulo": "Juan Santos Atahualpa", "url": "https://es.wikipedia.org/wiki/Juan_Santos_Atahualpa"}]},
    {"pregunta": "quien fue manuel pardo",
     "respuesta": "Manuel Pardo fue el primer presidente civil del Perú (1872–1876) y fundador del Partido Civil.",
     "fuentes": [{"titulo": "Manuel Pardo", "url": "https://es.wikipedia.org/wiki/Manuel_Pardo_y_Lavalle"}]},
    {"pregunta": "batalla de san juan y miraflores",
     "respuesta": "Las batallas de San Juan y Miraflores se libraron en enero de 1881 durante la defensa de Lima.",
     "fuentes": [{"titulo": "Campaña de Lima", "url": "https://es.wikipedia.org/wiki/Campa%C3%B1a_de_Lima"}]}
]


# =====================================================
# 🧠 UTILIDADES Y MODELO
# =====================================================
def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    return ' '.join(text.split())

@st.cache_resource(show_spinner=False)
def build_bot(data: List[Dict[str, object]], ngram=(1, 2)):
    questions = [normalize_text(d["pregunta"]) for d in data]
    answers = [d["respuesta"] for d in data]
    sources = [d.get("fuentes", []) for d in data]
    vec = TfidfVectorizer(ngram_range=ngram)
    X = vec.fit_transform(questions)
    return vec, X, questions, answers, sources

def predict(vec, X, questions, answers, sources, user_text: str, threshold: float):
    user_norm = normalize_text(user_text)
    sims = cosine_similarity(vec.transform([user_norm]), X)[0]
    idx = sims.argmax()
    score = float(sims[idx])
    if score < threshold:
        return "Lo siento, solo puedo responder preguntas de Historia del Perú.", score, []
    return answers[idx], score, sources[idx]


# =====================================================
# 🎨 INTERFAZ STREAMLIT
# =====================================================
def main():
    st.set_page_config(page_title="Bot de Historia del Perú — Yeltsin Solano Díaz", page_icon="🇵🇪", layout="centered")

    st.markdown(
        """
        <style>
        .stTextInput > div > div > input { border-radius: 12px; }
        .block-container { padding-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🇵🇪 Bot de Historia del Perú")
    st.caption("NLP clásico: TF-IDF + similitud coseno • Fuentes confiables incluidas")

    with st.sidebar:
        st.subheader("⚙️ Configuración")
        threshold = st.slider("Umbral de confianza (coseno)", 0.0, 1.0, 0.25, 0.01)
        uploaded = st.file_uploader("Subir dataset (.json)", type=["json"])
        if uploaded:
            try:
                data_ext = json.load(uploaded)
                assert isinstance(data_ext, list)
                st.success(f"Dataset cargado: {len(data_ext)} entradas.")
                data = data_ext
            except Exception as e:
                st.error(f"JSON inválido: {e}")
                data = DATASET_30
        else:
            data = DATASET_30

        st.download_button(
            "⬇️ Descargar dataset de ejemplo (JSON)",
            data=json.dumps(DATASET_30, ensure_ascii=False, indent=2),
            file_name="dataset_historia_30_refs.json",
            mime="application/json",
            use_container_width=True
        )

        st.markdown("---")
        st.caption("Desarrollado por **Yeltsin Solano Díaz**")

    vec, X, questions, answers, sources = build_bot(data)

    if "q" not in st.session_state:
        st.session_state["q"] = ""

    col1, col2 = st.columns([3, 1])
    with col1:
        user_q = st.text_input("✍️ Escribe tu pregunta", value=st.session_state["q"],
                               placeholder="Ej. ¿Quién fue Miguel Grau?", key="q")
    with col2:
        if st.button("Limpiar", use_container_width=True):
            st.session_state["q"] = ""
            if hasattr(st, "rerun"):
                st.rerun()

    if st.session_state["q"]:
        ans, sc, refs = predict(vec, X, questions, answers, sources, st.session_state["q"], threshold)
        st.markdown("### 🧠 Respuesta")
        st.write(ans)
        st.caption(f"Confianza: **{sc:.3f}**")
        if refs:
            st.markdown("#### 🔗 Fuentes sugeridas")
            for r in refs:
                st.markdown(f"- [{r.get('titulo')}]({r.get('url')})")

    st.markdown("---")
    st.caption("Desarrollado por **Yeltsin Solano Díaz** · Hecho con ❤️ en Python + Streamlit")


if __name__ == "__main__":
    main()
