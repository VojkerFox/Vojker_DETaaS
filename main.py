import pandas as pd
import jax.numpy as jnp
import os
import json
from data_ingestion.hilma_api import fetch_mock_hilma_data
from jax_engine.tqg_filter import process_hilma_tqg
from panama_process.kill_switch import panama_hilma_check

def load_fsm_rules(filepath="fsm_rules/qred_siltarahoitus.json"):
    """Lukee asiakkaan ehdot JSON-tiedostosta."""
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

def run_pipeline():
    print("\n🚀 Käynnistetään Vojker DetaaS - Tuotantolinja...\n")
    
    # 0. Ladataan säännöt (Aivot)
    rules = load_fsm_rules()
    min_rev = rules["panama_ehdot"]["min_liikevaihto_eur"]
    hinta = rules["hinnoittelu"]["hinta_per_liidi_eur"]
    print(f"[0/4] Säännöt ladattu ({rules['asiakas']}): Minimiliikevaihto {min_rev}€")

    # 1. Ingestion (Data sisään)
    print("[1/4] Haetaan raakadata (Hilma & YTJ)...")
    df_meta, raw_matrix = fetch_mock_hilma_data()
    print(f"      -> Sisään otettu {len(df_meta)} yritystä.")
    
    # 2. TQG Filter (Entropian siivous)
    print("[2/4] Ajetaan JAX TQG-suodatin...")
    clean_matrix = process_hilma_tqg(raw_matrix)
    
    # 3. Panama Kill Switch (Säännöt tulevat nyt JSONista!)
    print(f"[3/4] Ajetaan Panama-prosessi...")
    final_matrix = panama_hilma_check(clean_matrix, min_revenue=min_rev)
    
    # 4. Deterministinen Output
    print("[4/4] Generoidaan puhdistettu tuote asiakkaalle...")
    survivor_mask = final_matrix[:, 1] > 0
    df_final = df_meta[survivor_mask.tolist()].copy()
    
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/qred_leads_2026_06_27.csv"
    df_final.to_csv(output_path, index=False, sep=";")
    
    print(f"\n✅ Tuotantolinja valmis! Datan puhtaus 100%.")
    print(f"💰 Validoituja FSM-osumia: {len(df_final)} kpl (Laskutettava arvo: {len(df_final) * hinta} €)")
    print(f"📁 Tuote tallennettu: {output_path}\n")

if __name__ == "__main__":
    run_pipeline()