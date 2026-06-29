import os
import re
import shutil
import psycopg2
import fitz  # PyMuPDF
from jax_engine.vectorizer import model

# TIETOKANNAN ASETUKSET
DB_CONFIG = {
    "dbname": "vojker_detaas",          
    "user": "postgres",
    "password": "password", # <- LAITA OMA SALASANASI
    "host": "localhost",
    "port": "5432"
}

# KANSIORAKENTEEN VAKIOINTI (Luo nämä kansiot projektisi juureen!)
INPUT_DIR = "pdf_syote"    # Tänne vain dumppaat uudet lehdet
ARCHIVE_DIR = "pdf_arkisto" # Kone siirtää valmiit lehdet tänne automaattisesti

def setup_folders():
    """Varmistaa, että LEAN-kansionhallinta on pystyssä."""
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)

def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SET search_path TO vojker_detaas, public;")
    return conn, cur

def process_single_pdf(pdf_path, filename, cur):
    """Pureskelee yhden PDF:n älykkäiksi lohkoiksi kantaan."""
    # VAKIOITU NIMEN LUKU: Puretaan esim. "EuroMetalli_2026_Kesakuu.pdf" alaviivoista
    clean_name = filename.replace(".pdf", "")
    parts = clean_name.split("_")
    
    if len(parts) < 3:
        print(f"⚠️ Ohitetaan tiedosto {filename}: Nimen on oltava muotoa Lehti_Vuosi_Kuukausi.pdf")
        return False
        
    lehti_nimi = parts[0]
    vuosi = int(parts[1])
    kuukausi = parts[2]
    
    print(f"\n📖 Automaatio havaitsi uuden lehden: {lehti_nimi} ({kuukausi} {vuosi})")
    doc = fitz.open(pdf_path)
    
    cur.execute(
        "INSERT INTO julkaisut (lehti_nimi, vuosi, kuukausi) VALUES (%s, %s, %s) RETURNING id;",
        (lehti_nimi, vuosi, kuukausi)
    )
    julkaisu_id = cur.fetchone()[0]
    
    block_counter = 0
    for sivu_idx, page in enumerate(doc):
        sivu_nro = sivu_idx + 1
        blocks = page.get_text("blocks")
        cleaned_paragraphs = []
        
        for b in blocks:
            block_text = b[4].strip()
            if len(block_text) > 15 and b[6] == 0:  
                clean_text = block_text.replace("-\n", "").replace("\n", " ")
                cleaned_paragraphs.append(clean_text)
        
        if not cleaned_paragraphs:
            continue
            
        cur.execute(
            "INSERT INTO sivut (julkaisu_id, sivu_numero) VALUES (%s, %s) RETURNING id;",
            (julkaisu_id, sivu_nro)
        )
        sivu_id = cur.fetchone()[0]
        
        # JAX-tensorointi livenä
        vectors = model.encode(cleaned_paragraphs, show_progress_bar=False)
        
        for block_nro, (text, vec) in enumerate(zip(cleaned_paragraphs, vectors), start=1):
            cur.execute(
                "INSERT INTO tekstirivit (sivu_id, rivi_numero, teksti, embedding) VALUES (%s, %s, %s, %s);",
                (sivu_id, block_nro, text, vec.tolist())
            )
            block_counter += 1
            
    doc.close()
    print(f"✅ Suoritettu: {filename} ({block_counter} laadukasta lohkoa valettu tietokantaan).")
    return True

def run_auto_pipeline():
    setup_folders()
    conn, cur = get_connection()
    
    # Huom: Poistettu TRUNCATE, jotta uudet lehdet LISÄTÄÄN vanhojen alle, ei pyyhitä yli!
    
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"ℹ️ Kansio '{INPUT_DIR}' on tyhjä. Ei uutta materiaalia käsiteltävänä.")
        cur.close()
        conn.close()
        return

    print(f"🚀 Löydetty {len(pdf_files)} uutta PDF-tiedostoa. Käynnistetään LEAN-linjasto...")
    
    try:
        for filename in pdf_files:
            source_path = os.path.join(INPUT_DIR, filename)
            dest_path = os.path.join(ARCHIVE_DIR, filename)
            
            # Prosessoidaan tiedosto
            success = process_single_pdf(source_path, filename, cur)
            
            if success:
                # SIIRRETÄÄN VALMIS TIEODSTO ARKISTOON (Tärkeä! Estää tuplalataukset)
                shutil.move(source_path, dest_path)
                print(f"📦 Tiedosto siirretty arkistoon: {ARCHIVE_DIR}/{filename}")
                
        conn.commit()
        print("\n⚡ Kaikki uudet materiaalit käsitelty onnistuneesti.")
    except Exception as e:
        conn.rollback()
        print(f"❌ VIRHE LINJASTOLLA: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    run_auto_pipeline()