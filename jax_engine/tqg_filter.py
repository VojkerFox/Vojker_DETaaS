import jax
import jax.numpy as jnp

@jax.jit
def process_hilma_tqg(matrix):
    """
    TQG-suodatin: Imee sisään yritysdatan.
    Askel 1: Siivoaa entropian (korvaa NaN ja Inf nollilla).
    """
    # Etsitään NaN (Not a Number) ja Inf (Ääretön) arvot ja korvataan ne 0.0:lla
    # Tämä on vektorisoitu, joten se tapahtuu koko 10 000 rivin matriisille yhdellä kellojaksolla
    clean_matrix = jnp.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    return clean_matrix