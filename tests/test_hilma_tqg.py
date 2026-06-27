import pytest
import jax
import jax.numpy as jnp
import hashlib
import time
import numpy as np


from jax_engine.tqg_filter import process_hilma_tqg
from panama_process.kill_switch import panama_hilma_check
# -------------------------------------------------------------------------
# TESTIDATAN GENEROINTI (Simuloidaan Hilmasta ja YTJ:stä tullut sekava data)
# Matriisin muoto: [Urakan_arvo, Liikevaihto, Verovelka, Rekrytoinnit, Tila_Koodi]
# -------------------------------------------------------------------------
@pytest.fixture
def mock_hilma_matrix():
    key = jax.random.PRNGKey(42)
    # Generoidaan 10 000 yrityksen "raakadata"
    return jax.random.uniform(key, shape=(10000, 5), minval=0, maxval=1000000)

# -------------------------------------------------------------------------
# TESTI 1: Shape Contract (Muotosopimus)
# -------------------------------------------------------------------------
def test_hilma_shape_contract(mock_hilma_matrix):
    # Data sisään
    output = process_hilma_tqg(mock_hilma_matrix)
    
    # Tarkistetaan että JAX ei kadottanut dimensioita (Pysyy 2D-matriisina)
    assert len(output.shape) == 2, "Datan on pysyttävä kaksiulotteisena matriisina."
    # Sarakkeiden määrän (5) on pysyttävä samana, vain rivien (yritysten) määrä saa tippua
    assert output.shape[1] == 5, f"Odotettiin 5 saraketta, saatiin {output.shape[1]}"

# -------------------------------------------------------------------------
# TESTI 2: Pure Function & Determinismi (SHA256)
# -------------------------------------------------------------------------
def test_hilma_determinism(mock_hilma_matrix):
    # Ajetaan täsmälleen sama data kahdesti
    run_1 = process_hilma_tqg(mock_hilma_matrix)
    run_2 = process_hilma_tqg(mock_hilma_matrix)
    
    # Muutetaan tulokset tavuiksi ja lasketaan tiivisteet (Hash)
    hash_1 = hashlib.sha256(np.array(run_1).tobytes()).hexdigest()
    hash_2 = hashlib.sha256(np.array(run_2).tobytes()).hexdigest()
    
    # Jos hashit eivät täsmää, funktiossa on piilotettua entropiaa (vibes)
    assert hash_1 == hash_2, "Moottori ei ole deterministinen! Sama input antoi eri outputin."

# -------------------------------------------------------------------------
# TESTI 3: Entropy / NaN Handling (Kohinan sieto)
# -------------------------------------------------------------------------
def test_hilma_entropy_handling():
    # Syötetään tahallaan korruptoitunutta dataa (NaN ja Inf)
    bad_data = jnp.array([
        [100000.0, jnp.nan, 0.0, 5.0, 1.0],
        [jnp.inf, 500000.0, 0.0, 0.0, 1.0],
        [50000.0, 200000.0, 0.0, 2.0, 1.0]
    ])
    
    try:
        output = process_hilma_tqg(bad_data)
        # Järjestelmä ei saa kaatua, ja sen pitää osata joko pudottaa tai nollata vialliset rivit
        assert not jnp.any(jnp.isnan(output)), "JAX-moottori päästi NaN-arvoja läpi!"
    except Exception as e:
        pytest.fail(f"Moottori kaatui kohinaiseen dataan: {e}")

# -------------------------------------------------------------------------
# TESTI 4: Panama Kill Switch (FSM-lukitus)
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# TESTI 4: Panama Kill Switch (FSM-lukitus)
# -------------------------------------------------------------------------
def test_panama_kill_switch():
    # [Urakan_arvo, Liikevaihto, Verovelka, Rekrytoinnit, Tila_Koodi]
    test_matrix = jnp.array([
        [200000.0, 50000.0, 0.0, 2.0, 1.0],  # Hylättävä (liikevaihto 50k)
        [200000.0, 200000.0, 0.0, 2.0, 1.0]  # Hyväksyttävä (liikevaihto 200k)
    ])
    
    filtered_output = panama_hilma_check(test_matrix, min_revenue=100000.0)
    
    # UUSI JAX-TESTI: Matriisin koko ei saa muuttua!
    assert filtered_output.shape[0] == 2, "Panama-prosessi muutti matriisin kokoa (kielletty JIT:issä)!"
    
    # Hylätyn yrityksen (Rivi 0) KAIKKI arvot pitää olla nollia
    assert jnp.all(filtered_output[0] == 0.0), "Kill Switch ei nollannut hylättyä yritystä!"
    
    # Hyväksytyn yrityksen (Rivi 1) urakan arvon (indeksi 0) pitää olla ennallaan (200000)
    assert filtered_output[1][0] == 200000.0, "Kill Switch tuhosi hyväksytyn yrityksen datan!"
# -------------------------------------------------------------------------
# TESTI 5: SPC Performance Test (Tahtiaika & Cpk 3.0)
# -------------------------------------------------------------------------
def test_spc_performance_cpk(mock_hilma_matrix):
    # USL (Upper Specification Limit) = 50 millisekuntia per 10k riviä
    usl_time_seconds = 0.05 
    
    # Warm-up (Kompiloidaan JIT, jotta se ei vääristä mittausta)
    _ = process_hilma_tqg(mock_hilma_matrix)
    
    execution_times = []
    iterations = 100
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = process_hilma_tqg(mock_hilma_matrix)
        # JAXin asynkronisuuden vuoksi pitää pakottaa laskenta loppuun (block_until_ready)
        _.block_until_ready() if hasattr(_, "block_until_ready") else None
        end = time.perf_counter()
        execution_times.append(end - start)
        
    mean_time = np.mean(execution_times)
    std_dev = np.std(execution_times)
    
    # Estetään nollajako, jos prosessi on "liian" täydellinen
    if std_dev == 0.0:
        std_dev = 1e-9
        
    # Cpk laskenta (Kuinka hyvin prosessi mahtuu USL-rajan sisään)
    # Koska LSL:ää ajalle ei periaatteessa ole (mitä nopeampi sen parempi), katsomme vain USL:ää.
    cpk = (usl_time_seconds - mean_time) / (3 * std_dev)
    
    print(f"\nSPC Tulokset | Keskiaika: {mean_time:.5f}s | Hajonta: {std_dev:.5f}s | Cpk: {cpk:.2f}")
    
    assert cpk >= 3.0, f"SPC Fail: Prosessin Cpk ({cpk:.2f}) ei saavuta tavoitetta 3.0! Tahtiaika vaihtelee liikaa tai on liian hidas."