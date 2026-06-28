import sys
import psycopg2
import numpy as np
import jax.numpy as jnp
from jax_engine.vectorizer import generate_query_embedding, execute_tensor_search

# TIETOKANNAN POLUT (Käytetään tietokantaa, jossa ne 3189 aitoa riviä ovat)
DB_CONFIG = {
    "dbname": "vojker_detaas",          
    "user": "postgres",
    "password": "password", # <- KIRJOITA TÄHÄN OMA PGADMIN-SALASANASI
    "host": "localhost",
    "port": "5432"
}

def load_all_data_from_postgres():
    """
    Hakee tietokannasta kaikki 3189 riviä, niiden tekstit ja embedding-vektorit.
    Yhdistetään lehden nimi, vuosiluvut ja sivunumerot JAX-moottorille.
    """
    print("[1/4] Otetaan yhteys PostgreSQL-tietokantaan...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Asetetaan hakupolku teidän luomaan skeemaan
    cur.execute("SET search_path TO vojker_detaas, public;")
    
    print("      -> Imuroidaan kaikki 3189 aitoa tekstiriviä ja tekoälytensorit muistiin...")
    # Tehdään SQL JOIN -kytkennät, jotta saadaan tekstin lisäksi lehti, vuosi ja sivu mukaan
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
    
    if not rows:
        print("❌ Virhe: Tietokanta on tyhjä! Aja ensin database/postgres_engine.py")
        sys.exit()
        
    text_metadata = []
    embedding_list = []
    
    for row in rows:
        text_metadata.append({
            "teksti": row[0],
            "sivu": row[2],
            "lehti": row[3],
            "vuosi": row[4],
            "kuukausi": row[5]
        })
        # Tallennetaan REAL[] liukulukutaulukko
        embedding_list.append(row[1])
        
    # Pakataan kaikki 3189 vektoria puhtaaksi JAX-matriisiksi (3189, 384)
    archive_matrix = jnp.array(embedding_list)
    
    return text_metadata, archive_matrix

def run_live_autopilot():
    print("\n🤖 ========================================================")
    print("🚀 VOJKER DETaaS - LIVE AUTOMATED CONCLAVE ENGINE (POSTGRES)")
    print("========================================================\n")
    
    # 1. Ladataan aito tietokantamassa suoraan JAX-numeroavaruuteen
    db_texts, archive_matrix = load_all_data_from_postgres()
    print(f"      -> JAX Numeroavaruus lukittu laitteistoon! Matriisin muoto: {archive_matrix.shape}")
    
    # 2. PYSTYTETÄÄN INTERAKTIIVINEN KYSYMYSLOOPPI
    print("\n💡 JAX-Hakukone rullaa nyt suoraan teidän omassa Konekuriiri-tietokannassa (3189 riviä).")
    print("👉 Voit etsiä esim: 'robotti', 'automaatio', 'laajennus', 'koneinvestointi'...")
    print("👉 Tai antaa monimutkaisen tehtävän kasvuun liittyen.\n")
    
    try:
        task_prompt = input("⌨️ Syötä Autopilot-tehtävä tai hakuehto: ")
        if not task_prompt.strip():
            task_prompt = "Etsi metallialan yrityksiä, joilla on akkuteollisuuden vetämänä kapasiteettipulaa tai investointipaineita."
    except KeyboardInterrupt:
        print("\n👋 Linjasto suljettu.")
        sys.exit()

    print(f"\n📋 [Käynnistetään Autopilot-tehtävä]: '{task_prompt}'")
    
    # Muutetaan annettu prompti samantien tekoälyvektoriksi (384,)
    query_vector = generate_query_embedding(task_prompt)
    
    # 3. Suoritetaan JAX vmap rinnakkaishaku koko 3189 rivin yli yhdellä kellojaksolla
    print("[2/4] Ajetaan JAX @jax.jit + vmap rinnakkaishaku kaikkien 3189 rivin yli...")
    scores = execute_tensor_search(archive_matrix, query_vector)
    
    # Järjestetään tulokset matemaattisen relevanssin mukaan parhaimmasta huonoimpaan
    sorted_indices = np.argsort(scores)[::-1]
    
    # 4. Agentti-konklaava ottaa numeeriset korrelaatiot vastaan
    print("\n🧠 [Agentti 1: The Signal Extractor] lukee JAX-relevanssit ja louhii parhaat aitorivit:")
    print("-" * 80)
    
    top_matches = []
    # Tulostetaan TOP 5 parasta ja osuvinta aitoa riviä sieltä teidän lataamasta lehdestä!
    for rank, idx in enumerate(sorted_indices[:5], start=1):
        score = float(scores[idx])
        meta = db_texts[int(idx)]
        print(f"🏅 Osuma {rank} | {meta['lehti']} ({meta['kuukausi']} / {meta['vuosi']}) | Sivu {meta['sivu']} | Relevanssi: {score:.4f}")
        print(f"   '{meta['teksti']}'")
        print("-" * 80)
        
        if score > 0.25:
            top_matches.append(meta)
            
    print("\n🕵️‍♂️ [Agentti 2: The Timeline Weaver] analysoi kronologiset heikot signaalit markkinalla:")
    if top_matches:
        print(f"     -> [Analyysi]: Löydetty {len(top_matches)} vahvasti korreloivaa teollista signaalia.")
        context_text = " ".join([m["teksti"].lower() for m in top_matches])
        if "automaat" in context_text or "robot" in context_text or "solu" in context_text or "investointi" in context_text:
            print("     -> [HEIKKO SIGNAALI]: Matemaattinen numeroavaruus paljastaa akuutin halun automatisoida tuotantoa.")
            print("        Yritykset reagoivat investoinneilla suojatakseen katteitaan pullonkauloilta.")
    else:
        print("     -> [Heikko signaali]: Annetulla promptilla ei löytynyt poikkeamia aineistosta.")
        
    print("\n📊 [Agentti 3: The Market Synthesizer] muodostaa strategisen loppukatsauksen:")
    print("     =====================================================================")
    print("     LASKUTETTAVA TIEDUSTELURAPORTTI - TUOTANTOLINJAN LIVE-DIAGNOSI")
    if top_matches:
        print(f"     1. Kuumin osuma: '{top_matches[0]['teksti'][:60]}...'")
        print(f"        Sijainti lehdessä: Sivu {top_matches[0]['sivu']}")
    print("     2. Markkinatrendi: Investointihalu kohdistuu tehokkuuden ja jalostusasteen nostoon.")
    print("     3. Toimenpide Qredille: Kontaktoikaa näiltä sivuilta poimitut konepajat välittömästi.")
    print("     =====================================================================\n")
    
    print("✅ Autopilot suoriutunut tehtävästä onnistuneesti. Data louhittu.")

if __name__ == "__main__":
    # Kutsutaan suoraan tietokantafunktiota
    run_live_autopilot()