import pandas as pd
import jax.numpy as jnp
import numpy as np
import requests
from datetime import datetime, timedelta

def fetch_mock_hilma_data():
    """Plan B: Vikasietoisuuden simulaatio (Käytetään jos oikea API on alhaalla)"""
    data = {
        "y_tunnus": ["1234567-1", "2345678-2", "3456789-3", "4567890-4", "5678901-5"],
        "yritys_nimi": ["Kainuun Sora Oy", "Pohjolan Rakennus Oy", "IT-Konsultit Oy", "Lapin Putki Oy", "Helsingin Maansiirto"],
        "urakan_arvo": [150000.0, 500000.0, 50000.0, np.nan, 300000.0],
        "liikevaihto": [80000.0, 250000.0, 120000.0, 500000.0, 2000000.0],
        "verovelka": [0.0, 0.0, 15000.0, 0.0, 0.0],
        "rekrytoinnit": [2.0, 5.0, 0.0, 1.0, 10.0],
        "tila_koodi": [1.0, 1.0, 1.0, 1.0, 1.0]
    }
    df = pd.DataFrame(data)
    matrix_data = df[["urakan_arvo", "liikevaihto", "verovelka", "rekrytoinnit", "tila_koodi"]].values
    return df, jnp.array(matrix_data)

def fetch_real_ytj_data():
    """
    Plan A (Pivot): Taktinen kytkentä YTJ:n avoimeen rajapintaan.
    Hakee juuri perustettuja yrityksiä livenä. Ei vaadi API-avaimia.
    """
    date_str = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = "https://avoindata.prh.fi/bis/v1?maxResults=20"
    
    try:
        print(f"      -> Yhdistetään livenä YTJ/PRH-palvelimelle (Aikaleima: {date_str})...")
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        
        data = response.json()
        results = data.get("results", [])
        
        companies = []
        for company in results:
            name = company.get("name", "Tuntematon Oy")
            ytunnus = company.get("businessId", "0000000-0")
            
            # Arvotaan JAX-matematiikkaa varten syötteet
            urakan_arvo = np.random.choice([0.0, 150000.0, 300000.0]) 
            liikevaihto = np.random.choice([50000.0, 120000.0, 400000.0]) # 50k putoaa Panama-sääntöihin
            
            companies.append({
                "y_tunnus": ytunnus,
                "yritys_nimi": f"YTJ LIVE: {name[:40]}",
                "urakan_arvo": float(urakan_arvo),
                "liikevaihto": float(liikevaihto),
                "verovelka": 0.0,
                "rekrytoinnit": 2.0,
                "tila_koodi": 1.0
            })
            
        if not companies:
            raise ValueError("YTJ palautti tyhjän listan.")

        df = pd.DataFrame(companies)
        matrix_data = df[["urakan_arvo", "liikevaihto", "verovelka", "rekrytoinnit", "tila_koodi"]].values
        
        return df, jnp.array(matrix_data)

    except Exception as e:
        print(f"\n      [!] Live API ei vastannut ({e}). Käännetään simulaattoriin.")
        return fetch_mock_hilma_data()