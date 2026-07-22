import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE

# Configuration de la page Streamlit
st.set_page_config(
    page_title="TP-tSNE MNIST", layout="centered", initial_sidebar_state="expanded"
)

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%); color: white; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 25px;">
        <div style="font-size: 11pt; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; opacity: 0.85;">EL MEHDI - Master IAENG</div>
        <h1 style="margin: 10px 0 0 0; font-size: 24pt;">TP-t-SNE</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Réduction Non Linéaire & Visualisation des Clusters MNIST</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Chargement optimisé des données MNIST
@st.cache_data
def load_mnist_data():
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    X = mnist.data.astype("float32") / 255.0
    y = mnist.target.astype("int")
    return X, y


with st.spinner("Chargement du dataset MNIST en cours (veuillez patienter)..."):
    X, y = load_mnist_data()

# Panneau de configuration (Sidebar)
st.sidebar.header("Paramètres t-SNE")
sample_size = st.sidebar.slider(
    "Taille de l'échantillon (vitesse)",
    min_value=1000,
    max_value=5000,
    value=2500,
    step=500,
)
perplexity = st.sidebar.slider(
    "Perplexité (perplexity)", min_value=5, max_value=50, value=30
)
learning_rate = st.sidebar.slider(
    "Taux d'apprentissage (learning_rate)", min_value=10, max_value=500, value=200
)

# Réduction d'échantillon pour fluidifier le calcul t-SNE
indices = np.random.choice(len(X), sample_size, replace=False)
X_subset = X.iloc[indices].values if hasattr(X, "iloc") else X[indices]
y_subset = y.iloc[indices].values if hasattr(y, "iloc") else y[indices]

# Application de l'algorithme t-SNE (intégré dans scikit-learn)
st.subheader("1. Projection t-SNE en 2D")
with st.spinner("Exécution de la projection t-SNE (calcul itératif)..."):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=42,
    )
    embedding = tsne.fit_transform(X_subset)

# Affichage du graphique interactif Matplotlib avec anti-coupure (bbox_inches='tight')
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=y_subset,
    cmap="tab10",
    s=10,
    alpha=0.7,
)
legend = ax.legend(
    *scatter.legend_elements(),
    title="Chiffres",
    loc="upper right",
    bbox_to_anchor=(1.25, 1),
)
ax.add_artist(legend)
ax.set_title(
    "Projection t-SNE des chiffres MNIST (Espace 2D)",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Dimension t-SNE 1")
ax.set_ylabel("Dimension t-SNE 2")
ax.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig, bbox_inches="tight")

st.markdown("---")
st.markdown("### 💡 Ce que montre ce graphique :")
st.markdown(
    "- Le **t-SNE** convertit les proximités entre pixels de grande dimension en probabilités, créant des îlots (clusters) magnifiquement isolés pour chaque chiffre de 0 à 9."
)
