import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_openml, load_digits
from sklearn.manifold import TSNE

# 1. Configuration de la page Streamlit
st.set_page_config(
    page_title="TP-tSNE MNIST - EL MEHDI",
    layout="centered",
    initial_sidebar_state="expanded",
)

# En-tête personnalisé avec design soigné
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


# 2. Chargement sécurisé avec repli anti-plantage réseau
@st.cache_data(show_spinner=False)
def load_mnist_data():
    try:
        mnist = fetch_openml("mnist_784", version=1, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        y = mnist.target.astype(np.int64)
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        return X, y
    except Exception:
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(np.int64)
        return X, y


try:
    with st.spinner("Chargement des données..."):
        X, y = load_mnist_data()
except Exception as e:
    st.error(f"Erreur critique : {e}")
    st.stop()


# 3. Panneau de configuration (Sidebar épurée sans la graine)
st.sidebar.header("Paramètres t-SNE")

max_samples = min(len(X), 3000)
sample_size = st.sidebar.slider(
    "Taille de l'échantillon (vitesse)",
    min_value=500,  # Abaissé à 500 pour plus de rapidité
    max_value=max_samples,
    value=1500,  # Valeur par défaut plus légère
    step=500,
)

perplexity = st.sidebar.slider(
    "Perplexité (perplexity)", min_value=5, max_value=50, value=30
)
learning_rate = st.sidebar.slider(
    "Taux d'apprentissage (learning_rate)",
    min_value=10,
    max_value=500,
    value=200,
)

# Graine fixée en arrière-plan (plus de champ visuel encombrant)
RANDOM_SEED = 42


# 4. Sous-échantillonnage ultra-rapide et mis en cache
@st.cache_data(show_spinner=False)
def get_subset(X, y, sample_size, seed):
    np.random.seed(seed)
    actual_size = min(sample_size, len(X))
    indices = np.random.choice(len(X), actual_size, replace=False)
    return X[indices], y[indices]


X_subset, y_subset = get_subset(X, y, sample_size, RANDOM_SEED)


# 5. Mise en cache stricte du calcul t-SNE
@st.cache_data(show_spinner=False)
def compute_tsne(X_sub, perplexity, learning_rate, seed):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=seed,
        init="pca",
        max_iter=1000,
    )
    return tsne.fit_transform(X_sub)


# 6. Exécution et affichage de la projection
st.subheader("1. Projection t-SNE en 2D")

with st.spinner("Calcul t-SNE en cours..."):
    try:
        embedding = compute_tsne(
            X_subset, perplexity, learning_rate, RANDOM_SEED
        )
    except Exception as e:
        st.error(f"Erreur durant l'exécution de t-SNE : {e}")
        st.stop()

# 7. Création du graphique Matplotlib
fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=y_subset,
    cmap="tab10",
    s=12,
    alpha=0.75,
    edgecolors="none",
)

legend = ax.legend(
    *scatter.legend_elements(),
    title="Chiffres",
    loc="upper right",
    bbox_to_anchor=(1.22, 1),
    frameon=True,
    facecolor="#f8fafc",
    edgecolor="#cbd5e1",
)
ax.add_artist(legend)

ax.set_title(
    "Projection t-SNE des chiffres MNIST (Espace 2D)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Dimension t-SNE 1", fontsize=10)
ax.set_ylabel("Dimension t-SNE 2", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

st.pyplot(fig, bbox_inches="tight")

# 8. Section explicative
st.markdown("---")
st.markdown("### 💡 Analyse & Performance :")
st.markdown(
    """
* **Échantillon flexible :** Vous pouvez descendre jusqu'à **500 échantillons** dans la barre latérale pour obtenir un affichage quasi instantané.
"""
)
