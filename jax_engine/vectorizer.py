import jax
import jax.numpy as jnp
from sentence_transformers import SentenceTransformer

# Alustetaan monikielinen embedding-malli, joka tukee erinomaisesti suomea.
# Malli muuttaa jokaisen artikkelin/lauseen 384-ulotteiseksi vektoriksi.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

def generate_embeddings(texts):
    """Muuttaa listan tekstejä oikeiksi numeerisiksi JAX-vektoreiksi (N, 384)."""
    embeddings = model.encode(texts, show_progress_bar=False)
    return jnp.array(embeddings)

def generate_query_embedding(query):
    """Muuttaa käyttäjän tai Autopilot-agentin antaman tehtävän vektoriksi (384,)."""
    embedding = model.encode(query, show_progress_bar=False)
    return jnp.array(embedding)

def compute_similarity(archive_vector, query_vector):
    """Laskee kosinisimilaarisuuden kahden vektorin välillä tensoriavaruudessa."""
    norm_a = jnp.linalg.norm(archive_vector)
    norm_q = jnp.linalg.norm(query_vector)
    # Pistetulo jaettuna normien tulolla = kosinisimilaarisuus (estetään nollajako 1e-9:llä)
    return jnp.dot(archive_vector, query_vector) / (norm_a * norm_q + 1e-9)

# Kytketään JAX vmap: Rullaa arkistomatriisia rivi riviltä (0), pitää hakuvektorin vakiona (None)
vmap_semantic_search = jax.vmap(compute_similarity, in_axes=(0, None))

@jax.jit
def execute_tensor_search(archive_matrix, query_vector):
    """Suorittaa salamannopean JIT-käännetyn haun 35 vuoden arkistotensorille ilman for-looppeja."""
    return vmap_semantic_search(archive_matrix, query_vector)