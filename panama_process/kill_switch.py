import jax
import jax.numpy as jnp

@jax.jit
def panama_hilma_check(matrix, min_revenue):
    """
    Panama Kill Switch JAX-arkkitehtuurilla:
    Ei poista rivejä fyysisesti (Shape pysyy vakiona), 
    vaan nollaa hylättyjen yritysten koko datarivin.
    """
    # 1. Otetaan liikevaihdot (Sarake 1)
    revenues = matrix[:, 1]
    
    # 2. Luodaan boolean maski (True jos hyväksytty, False jos hylätty)
    valid_mask = revenues >= min_revenue
    
    # 3. Laajennetaan maski kattamaan kaikki 5 saraketta
    # JAX vaatii, että maskin muoto (N, 1) kerrotaan matriisin muodolla (N, 5)
    mask_expanded = jnp.expand_dims(valid_mask, axis=1)
    
    # 4. Kerrotaan matriisi maskilla.
    # Hyväksytyt rivit kerrotaan True (1), jolloin ne pysyvät ennallaan.
    # Hylätyt rivit kerrotaan False (0), jolloin koko rivistä tulee puhdas [0, 0, 0, 0, 0]
    filtered_matrix = matrix * mask_expanded
    
    return filtered_matrix