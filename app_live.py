import os
import sys
import re
import requests
import streamlit as st
import psycopg2
import fitz  # PyMuPDF
import numpy as np
import jax
import jax.numpy as jnp
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

# =====================================================================
# SYSTEM CONFIGURATION & SECURITY
# =====================================================================
load_dotenv()

DB_CONFIG = {
    "dbname": "vojker_detaas",          
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432"
}

@st.cache_resource
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def load_industrial_data():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SET search_path TO vojker_detaas, public;")
        cur.execute("""
            SELECT t.teksti, t.embedding, s.sivu_numero, j.lehti_nimi, j.vuosi, j.kuukausi
            FROM tekstirivit t
            JOIN sivut s ON t.sivu_id = s.id
            JOIN julkaisut j ON s.julkaisu_id = j.id
            ORDER BY t.id ASC;
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not rows: return [], jnp.array([])
            
        text_metadata = []
        embedding_list = []
        for row in rows:
            text_metadata.append({
                "teksti": row[0], "sivu": row[2], "lehti": row[3], "vuosi": row[4], "kuukausi": row[5]
            })
            embedding_list.append(row[1])
        return text_metadata, jnp.array(embedding_list)
    except Exception as e:
        st.error(f"🚨 TIETOKANTAVIRHE: {e}")
        return [], jnp.array([])

@jax.jit
def execute_strict_cosine_search(matrix, query):
    matrix_norms = jnp.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_normed = matrix / (matrix_norms + 1e-9)
    query_norm = jnp.linalg.norm(query)
    query_normed = query / (query_norm + 1e-9)
    return jnp.dot(matrix_normed, query_normed)

# ---------------------------------------------------------------------
# ENTITY EXTRACTION ENGINE (Sliding Context Window)
# ---------------------------------------------------------------------
def extract_sales_intelligence(target_paragraph, safe_context_window):
    company_match = re.search(r'\b([A-ZÄÖ][A-Za-zÄÖäö0-9\-\s&]+(?:OY|Oy|AB|Ab|OYJ|Oyj|KY|Ky|Yhtiöt|Valmiste|Konepaja))\b', target_paragraph)
    if not company_match:
        company_match = re.search(r'\b([A-ZÄÖ][A-Za-zÄÖäö0-9\-\s&]+(?:OY|Oy|AB|Ab|OYJ|Oyj|KY|Ky|Yhtiöt|Valmiste|Konepaja))\b', safe_context_window)
    company = company_match.group(1).strip() if company_match else "Target Company (Tarkista raakateksti)"
    
    person_match = re.search(r'(?:kertoo|sanoo|toteaa)[,\s]*([A-ZÄÖ][a-zäö]+ [A-ZÄÖ][a-zäö]+)', target_paragraph)
    if not person_match:
        person_match = re.search(r'([A-ZÄÖ][a-zäö]+ [A-ZÄÖ][a-zäö]+)[,\s]*(?:kertoo|sanoo|toteaa)', target_paragraph)
    if not person_match:
        person_match = re.search(r'(?:kertoo|sanoo|toteaa)[,\s]*([A-ZÄÖ][a-zäö]+ [A-ZÄÖ][a-zäö]+)', safe_context_window)
    person = person_match.group(1).strip() if person_match else "Päättäjä (Nimi ei suoraan osumassa)"
    
    phone_match = re.search(r'(?:Puh\.?|Puhelin|Tel\.?)?[\s]*((?:0|\+358)[\d\s\-]{6,11})', safe_context_window)
    phone = phone_match.group(1).strip() if phone_match else "Ei ilmoitettu (Hae Googlesta)"
    
    return company, person, phone

def highlight_entities(text, company, person):
    style = "<span style='color: #FFD700; font-weight: bold; background-color: rgba(255, 215, 0, 0.1); padding: 0 4px;'>"
    if company != "Target Company (Tarkista raakateksti)":
        text = text.replace(company, f"{style}{company}</span>")
    if person != "Päättäjä (Nimi ei suoraan osumassa)":
        for n in person.split():
            if len(n) > 3: text = text.replace(n, f"{style}{n}</span>")
    return text

def generate_pitch(target_paragraph):
    reasons = []
    low = target_paragraph.lower()
    if "investoin" in low or "automaat" in low or "kone" in low:
        reasons.append("🎯 **Tuore laiteinvestointi mainittu.** Rauta on kuumaa; akuutti tarve laiterahoituksen järjestelyyn.")
    elif "sahanter" in low or "hioma" in low:
        reasons.append("🎯 **Teollinen tarvikekauppa.** Aktiivista tilauskantaa ja tuotanto pyörii.")
    else:
        reasons.append("🎯 **Julkinen esiintyminen mediassa.** Vahva signaali kasvuhakuisuudesta.")
        
    if "kapasiteetti" in low or "pullonkaula" in low or "alihank" in low:
        reasons.append("🔥 **Tuotannon pullonkaula havaittu.** Rahoituspäätös voisi auttaa skaalaamaan toimintaa välittömästi.")
    else:
        reasons.append("🔥 **Operatiivinen tiedustelu.** Löydetty tieto antaa vahvan verukkeen avata keskustelu rahoitustarpeista.")
        
    reasons.append("💡 **Päättäjän ego-kulma.** Kehu lähdettä (lehti/video/artikkeli) avauksena. Se takaa lämpimän vastaanoton.")
    return reasons

# =====================================================================
# MULTI-MODAL INGESTION ENGINES
# =====================================================================
def insert_blocks_to_db(lehti_nimi, vuosi, kuukausi, paragraphs, ai_model):
    """Yhteinen tietokanta-inserteri kaikille datalähteille."""
    if not paragraphs: return 0
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SET search_path TO vojker_detaas, public;")
    cur.execute("INSERT INTO julkaisut (lehti_nimi, vuosi, kuukausi) VALUES (%s, %s, %s) RETURNING id;", (lehti_nimi, vuosi, kuukausi))
    j_id = cur.fetchone()[0]
    
    # Non-PDF lähteissä koko data menee "sivulle 1"
    cur.execute("INSERT INTO sivut (julkaisu_id, sivu_numero) VALUES (%s, %s) RETURNING id;", (j_id, 1))
    s_id = cur.fetchone()[0]
    
    vecs = ai_model.encode(paragraphs, show_progress_bar=False)
    for n, (t, v) in enumerate(zip(paragraphs, vecs), 1):
        cur.execute("INSERT INTO tekstirivit (sivu_id, rivi_numero, teksti, embedding) VALUES (%s,%s,%s,%s);", (s_id, n, t, v.tolist()))
    
    conn.commit()
    cur.close(); conn.close()
    return len(paragraphs)

def ingest_pdf_via_stream(uploaded_file, lehti_nimi, vuosi, kuukausi, ai_model):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SET search_path TO vojker_detaas, public;")
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    cur.execute("INSERT INTO julkaisut (lehti_nimi, vuosi, kuukausi) VALUES (%s, %s, %s) RETURNING id;", (lehti_nimi, vuosi, kuukausi))
    j_id = cur.fetchone()[0]
    total = 0
    p_bar = st.progress(0.0)
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        txts = [b[4].replace("-\n", "").replace("\n", " ").strip() for b in blocks if len(b[4].strip()) > 15 and b[6] == 0]
        if txts:
            cur.execute("INSERT INTO sivut (julkaisu_id, sivu_numero) VALUES (%s, %s) RETURNING id;", (j_id, i+1))
            s_id = cur.fetchone()[0]
            vecs = ai_model.encode(txts, show_progress_bar=False)
            for n, (t, v) in enumerate(zip(txts, vecs), 1):
                cur.execute("INSERT INTO tekstirivit (sivu_id, rivi_numero, teksti, embedding) VALUES (%s,%s,%s,%s);", (s_id, n, t, v.tolist()))
                total += 1
        p_bar.progress((i + 1) / len(doc))
    conn.commit()
    cur.close(); conn.close(); doc.close()
    return total

def ingest_youtube(url, lehti_nimi, vuosi, kuukausi, ai_model):
    """Purkaa YouTube-videon transkription teksteiksi."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not match: return 0
    video_id = match.group(1)
    
    try:
        # Hakee suomen tai englannin tekstitykset
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['fi', 'en'])
        
        # Nidotaan lyhyet 2 sekunnin lauseet n. 500 merkin teollisiksi lohkoiksi
        paragraphs = []
        current_chunk = ""
        for item in transcript:
            current_chunk += " " + item['text'].replace('\n', ' ')
            if len(current_chunk) > 500:
                paragraphs.append(current_chunk.strip())
                current_chunk = ""
        if current_chunk:
            paragraphs.append(current_chunk.strip())
            
        return insert_blocks_to_db(lehti_nimi, vuosi, kuukausi, paragraphs, ai_model)
    except Exception as e:
        st.error(f"YouTube-virhe (onko videossa tekstitykset?): {e}")
        return 0

def ingest_website(url, lehti_nimi, vuosi, kuukausi, ai_model):
    """Kaapii verkkosivun leipätekstin (p-tagit)."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if len(text) > 40: # Ohitetaan lyhyet napit ja linkit
                paragraphs.append(text)
                
        return insert_blocks_to_db(lehti_nimi, vuosi, kuukausi, paragraphs, ai_model)
    except Exception as e:
        st.error(f"Verkkosivun lukuvirhe: {e}")
        return 0

def ingest_raw_text(text, lehti_nimi, vuosi, kuukausi, ai_model):
    """Pilkkoo raakatekstin kappaleisiin rivinvaihtojen perusteella."""
    raw_blocks = text.split('\n\n')
    paragraphs = [p.replace('\n', ' ').strip() for p in raw_blocks if len(p.strip()) > 30]
    return insert_blocks_to_db(lehti_nimi, vuosi, kuukausi, paragraphs, ai_model)

# =====================================================================
# UX / UI ENTERPRISE LAYOUT
# =====================================================================
st.set_page_config(page_title="VOJKER DETaaS - Second Brain", layout="wide")
st.title("🤖 VOJKER DETaaS — Multi-Modal Sales Second Brain")
st.write("---")

model = get_embedding_model()
tab_search, tab_ingest = st.tabs(["🔍 Intelligence Search Engine", "📥 Multi-Modal Ingestion Workspace"])

# --- WORKSPACE 1: SEARCH ENGINE ---
with tab_search:
    db_texts, archive_matrix = load_industrial_data()
    if len(db_texts) == 0:
        st.warning("📊 Tietokanta on vielä tyhjä. Siirry 'Ingestion Workspace' -välilehdelle lataamaan materiaalia.")
    else:
        st.success(f"⚡ Muistissa {len(db_texts)} teollista/kognitiivista kappaletta. JAX-vektorisointi aktiivinen.")
        task_prompt = st.text_input("⌨️ Syötä haku tai tunnista hiljainen signaali (esim. kapasiteetin nosto):", key="s_input")
        if task_prompt:
            q_vec = model.encode(task_prompt)
            scores = execute_strict_cosine_search(archive_matrix, jnp.array(q_vec))
            indices = np.argsort(scores)[::-1]
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.header("🧠 Agent 1: Signal Extractor")
                shown = 0
                for idx in indices:
                    if shown >= 5: break  # Näytetään top 5 osumaa
                    meta = db_texts[int(idx)]
                    target_paragraph = meta['teksti'].strip()
                    if len(target_paragraph) < 80: continue
                    shown += 1
                    
                    full_page_blocks = [t['teksti'] for t in db_texts if t['sivu'] == meta['sivu'] and t['lehti'] == meta['lehti']]
                    
                    try:
                        block_idx = full_page_blocks.index(target_paragraph)
                        start_idx = max(0, block_idx - 2)
                        end_idx = min(len(full_page_blocks), block_idx + 2)
                        safe_context_window = " ".join(full_page_blocks[start_idx:end_idx])
                    except ValueError:
                        safe_context_window = target_paragraph
                        
                    comp, pers, phon = extract_sales_intelligence(target_paragraph, safe_context_window)
                    
                    with st.expander(f"🏅 SIGNAL {shown} | Lähde: {meta['lehti']} | Sivu/Osa {meta['sivu']} | Varmuus: {float(scores[idx]):.4f}", expanded=True):
                        st1, st2 = st.tabs(["💼 Qred Sales Agent", "📄 Context Data"])
                        with st1:
                            st.markdown(f"**Kohdeyritys:** {comp}\n\n**Päättäjä:** {pers}\n\n**Puhelin:** `{phon}`")
                            for r in generate_pitch(target_paragraph): st.markdown(r)
                        with st2:
                            st.markdown(f"**Tarkka Osuma (Hot-Zone):**\n\n{highlight_entities(target_paragraph, comp, pers)}", unsafe_allow_html=True)
                            st.write("---")
                            st.markdown("🔍 **Lähikappaleet (Sliding Context):**")
                            st.markdown(f"*{highlight_entities(safe_context_window, comp, pers)}*", unsafe_allow_html=True)

            with c2:
                st.header("🕵️‍♂️ Signal Analytics")
                st.info("Kone oppii tunnistamaan korrelaatioita yritysten välillä. Multi-modaalinen JAX-haku yhdistää luonnollisesti artikkelit, videot ja sivustot toisiinsa semanttisen etäisyyden perusteella.")

# --- WORKSPACE 2: MULTI-MODAL INGESTION ---
with tab_ingest:
    st.header("📥 Multi-Modal Ingestion Agent")
    st.write("Valitse datalähde. Järjestelmä siivoaa, vektoroi ja arkistoi datan Second Brain -tietokantaasi.")
    
    data_source = st.radio("Mitä dataa syötetään moottoriin?", 
                           ["📄 PDF-dokumentti (Lehdet, raportit)", 
                            "▶️ YouTube-video (Webinaarit, haastattelut)", 
                            "🌐 Verkkosivu (URL)", 
                            "📝 Raakateksti (Sähköpostit, muistiinpanot)"], 
                           horizontal=True)
    st.write("---")
    
    col_a, col_b = st.columns([1, 1])
    
    # 1. DUMPPAUSKENTÄT DATATYYPIN MUKAAN
    with col_a:
        st.subheader("1. Syötä data")
        if data_source == "📄 PDF-dokumentti (Lehdet, raportit)":
            input_data = st.file_uploader("Valitse PDF-dokumentti", type=["pdf"])
        elif data_source == "▶️ YouTube-video (Webinaarit, haastattelut)":
            input_data = st.text_input("YouTube-videon osoite (URL):", placeholder="https://www.youtube.com/watch?v=...")
        elif data_source == "🌐 Verkkosivu (URL)":
            input_data = st.text_input("Verkkosivun osoite (URL):", placeholder="https://www.esimerkki.fi/uutinen")
        elif data_source == "📝 Raakateksti (Sähköpostit, muistiinpanot)":
            input_data = st.text_area("Liitä teksti tähän:", height=200, placeholder="Kopioi ja liitä pitkä teksti tähän...")

    # 2. METATIEDOT
    with col_b:
        st.subheader("2. Määrittele konteksti")
        in_name = st.text_input("Lähteen Nimi / Brändi:", placeholder="Esim. Teknologiateollisuus Ry, Asiakaspalaveri")
        in_year = st.number_input("Vuosi:", 2000, 2050, 2026)
        in_desc = st.text_input("Lisätieto / Kuukausi:", placeholder="Esim. Webinaari 10.6. tai Q3 Katsaus")

    # 3. KÄSKYTYS
    if st.button("🚀 Käskytä Agentti: Prosessoi ja Arkistoi", type="primary", use_container_width=True):
        if not input_data or not in_name:
            st.error("⚠️ Syötä data ja määritä lähteen nimi ennen prosessointia!")
        else:
            with st.spinner(f"Agentti purkaa lähdettä '{in_name}' Second Brainiin..."):
                added_blocks = 0
                if data_source == "📄 PDF-dokumentti (Lehdet, raportit)":
                    added_blocks = ingest_pdf_via_stream(input_data, in_name, in_year, in_desc, model)
                elif data_source == "▶️ YouTube-video (Webinaarit, haastattelut)":
                    added_blocks = ingest_youtube(input_data, in_name, in_year, in_desc, model)
                elif data_source == "🌐 Verkkosivu (URL)":
                    added_blocks = ingest_website(input_data, in_name, in_year, in_desc, model)
                elif data_source == "📝 Raakateksti (Sähköpostit, muistiinpanot)":
                    added_blocks = ingest_raw_text(input_data, in_name, in_year, in_desc, model)
                
                st.cache_resource.clear()
                
                if added_blocks > 0:
                    st.success(f"✅ Loistavaa! {added_blocks} uutta kognitiivista lohkoa prosessoitu ja lisätty tekoälyaivoihin.")
                    st.balloons()
                else:
                    st.warning("⚠️ Dataa ei pystytty purkamaan. Varmista että URL on julkinen tai videossa on tekstitykset.")