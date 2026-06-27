import pandas as pd
import jax.numpy as jnp
import numpy as np

def fetch_mock_hilma_data():
    """
    Simuloi API-kutsuja Hilmaan ja YTJ:hin.
    Palauttaa Pandas DataFramen (ihmisten lukuun) ja JAX-matriisin (koneen laskentaan).
    """
    # 1. Generoidaan realistinen testidatasetti tiistain palaveriin
    data = {
        "y_tunnus": ["1234567-1", "2345678-2", "3456789-3", "4567890-4", "5678901-5"],
        "yritys_nimi": ["Kainuun Sora Oy", "Pohjolan Rakennus Oy", "IT-Konsultit Oy", "Lapin Putki Oy", "Helsingin Maansiirto"],
        "urakan_arvo": [150000.0, 500000.0, 50000.0, np.nan, 300000.0],  # np.nan testaa moottorin entropian sietoa
        "liikevaihto": [80000.0, 250000.0, 120000.0, 500000.0, 2000000.0], # 80k putoaa Panama-filtterissä
        "verovelka": [0.0, 0.0, 15000.0, 0.0, 0.0],
        "rekrytoinnit": [2.0, 5.0, 0.0, 1.0, 10.0],
        "tila_koodi": [1.0, 1.0, 1.0, 1.0, 1.0]
    }
    df = pd.DataFrame(data)
    
    # 2. JAX-matriisi (N, 5) eristetään vain koneen laskentaa varten
    matrix_data = df[["urakan_arvo", "liikevaihto", "verovelka", "rekrytoinnit", "tila_koodi"]].values
    jax_matrix = jnp.array(matrix_data)
    
    return df, jax_matrix