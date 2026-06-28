import os
import sys
import psycopg2
import fitz  # Tämä on se PyMuPDF, jonka juuri asensit!
from jax_engine.vectorizer import model  # sentence-transformers neuroverkko

# TIETOKANNAN KYTKENTÄTIEDOT (Päivitä salasanasi tähän)
DB_CONFIG = {
    "dbname": "vojker_detaas",       # pgAdminin tietokannan nimi
    "user": "postgres",         # Käyttäjätunnus
    "password": "password", # <-- LAITA TÄHÄN OMA PGADMIN-SALASANASI
    "host": "localhost",
    "port": "5432"
}

def get_connection():
    """Ottaa yhteyden ja kääntää istunnon suoraan vojker_detaas -skeemaan."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SET search_path TO vojker_detaas, public;")
    cur.close()
    return conn

def ingest_pdf_to_postgres(pdf_polku, lehti_nimi, vuosi, kuukausi):
    """
    Avaa aidon PDF-tiedoston, lukee sen sivu sivulta ja rivi riviltä,
    laskee tensolit neuroverkolla ja tallentaa kaiken PostgreSQL:ään.
    """
    if not os.path.exists(pdf_polku):
        print(f"\n❌ Virhe: PDF-tiedostoa ei löydy polusta: {pdf_polku}")
        print("👉 Vinkki: Laita tiedosto esim. projektin juureen ja anna sen nimi tähän.")
        return

    print(f"\n📖 Avataan aito PDF-dokumentti: {pdf_polku}")
    doc = fitz.open(pdf_polku)
    total_pages = len(doc)
    print(f"     -> Dokumentissa on {total_pages} sivua. Käynnistetään louhinta...")

    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # 1. Lisätään julkaisun perustiedot
        cur.execute(
            "INSERT INTO julkaisut (lehti_nimi, vuosi, kuukausi) VALUES (%s, %s, %s) RETURNING id;",
            (lehti_nimi, vuosi, kuukausi)
        )
        julkaisu_id = cur.fetchone()[0]
        
        total_lines_inserted = 0
        
        # 2. Rullataan aito dokumentti sivu sivulta (fitz indeksoi sivut nollasta)
        for sivu_idx, page in enumerate(doc):
            sivu_nro = sivu_idx + 1
            
            # Poimitaan sivun puhdas teksti
            sivu_teksti = page.get_text("text")
            
            # Pilkotaan teksti yksittäisiksi riveiksi
            rivit = [r.strip() for r in sivu_teksti.split("\n") if r.strip()]
            if not rivit:
                continue # Ohitetaan tyhjät sivut (kuten pelkät kuvat)
                
            cur.execute(
                "INSERT INTO sivut (julkaisu_id, sivu_numero) VALUES (%s, %s) RETURNING id;",
                (julkaisu_id, sivu_nro)
            )
            sivu_id = cur.fetchone()[0]
            
            # Lasketaan tekoälyvektorit kerralla koko sivun riveille (Batch-ajo)
            vektorit = model.encode(rivit, show_progress_bar=False)
            
            # 3. Tallennetaan rivi riviltä tietokantaan
            for rivi_nro, (teksti, vektori) in enumerate(zip(rivit, vektorit), start=1):
                # Muutetaan numpy-vektori tavalliseksi listaksi, jonka REAL[] ymmärtää natively
                emb_list = vektori.tolist()
                
                cur.execute(
                    "INSERT INTO tekstirivit (sivu_id, rivi_numero, teksti, embedding) VALUES (%s, %s, %s, %s);",
                    (sivu_id, rivi_nro, teksti, emb_list)
                )
                total_lines_inserted += 1
                
            print(f"     -> Sivu {sivu_nro}/{total_pages} ajettu kantaan... ({len(rivit)} riviä)")
                
        conn.commit()
        print(f"\n✅ Teollinen PDF-louhinta valmis!")
        print(f"📁 Julkaisu: {lehti_nimi} ({kuukausi} / {vuosi})")
        print(f"🚀 Tietokantaan räjäytetty yhteensä: {total_lines_inserted} aitoa tekstiriviä tensoreineen.")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Kriittinen virhe tietokanta-ajossa: {e}")
    finally:
        cur.close()
        conn.close()
        doc.close()

if __name__ == "__main__":
    # TÄSSÄ MÄÄRITETÄÄN REITTI JA TIEDOSTO
    # Voit ladata minkä tahansa koneellasi olevan PDF-lehden laittamalla sen polun tähän!
    AITO_PDF_REITTI = "konekuriiri_kesakuu.pdf" 
    
    print("==========================================================")
    print("VOJKER DETaaS - AITO PDF -> POSTGRESQL INGESTION PIPELINE")
    print("==========================================================")
    
    # KÄYNNISTETÄÄN LATAUS
    # Muuta tiedoston nimi yläpuolelta sellaiseksi, joka sinulla on kansiossa.
    ingest_pdf_to_postgres(AITO_PDF_REITTI, "Konekuriiri", 2026, "Kesäkuu")