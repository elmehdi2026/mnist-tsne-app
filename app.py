import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_openml
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
        <h1 style="margin: 10px 0 0 0; font-size: 24pt;">TP-t-SNE Avancé</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Réduction Non Linéaire & Visualisation des Clusters MNIST</p>
    </div>
""",
    unsafe_allow_html=True,
)


# 2. Chargement optimisé et sécurisé pour les accès simultanés
@st.cache_data(show_spinner=False, persist="disk")
def load_mnist_data():
    """Charge et normalise le dataset MNIST une seule fois globalement."""
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    y = mnist.target.astype(np.int64)
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    return X, y


# Chargement sécurisé avec indicateur
try:
    with st.spinner(
        "Chargement initial du dataset MNIST (partagé & optimisé)..."
    ):
        X, y = load_mnist_data()
except Exception as e:
    st.error(f"Erreur critique lors du chargement des données OpenML : {e}")
    st.stop()


# 3. Panneau de configuration (Sidebar)
st.sidebar.header("Paramètres t-SNE")
sample_size = st.sidebar.slider(
    "Taille de l'échantillon (vitesse)",
    min_value=1000,
    max_value=3000,
    value=2000,
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
random_state_val = st.sidebar.number_input(
    "Graine aléatoire (Random State)", value=42, step=1
)


# 4. Sous-échantillonnage reproductible et sécurisé
@st.cache_data(show_spinner=False)
def get_subset(X, y, sample_size, seed):
    """Garantit un sous-échantillonnage stable pour un jeu de paramètres donné."""
    np.random.seed(seed)
    indices = np.random.choice(len(X), sample_size, replace=False)
    return X[indices], y[indices]


X_subset, y_subset = get_subset(X, y, sample_size, int(random_state_val))


# 5. Mise en cache de l'algorithme t-SNE
@st.cache_data(show_spinner=False)
def compute_tsne(X_sub, perplexity, learning_rate, random_state):
    """Exécute l'algorithme t-SNE de scikit-learn avec les hyperparamètres choisis."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=random_state,
        init="pca",
        n_iter=1000,
    )
    return tsne.fit_transform(X_sub)


# 6. Exécution et affichage de la projection
st.subheader("1. Projection t-SNE en 2D")

with st.spinner("Calcul itératif t-SNE en cours (patientez quelques secondes)..."):
    try:
        embedding = compute_tsne(
            X_subset, perplexity, learning_rate, int(random_state_val)
        )
    except Exception as e:
        st.error(f"Erreur durant l'exécution de t-SNE : {e}")
        st.stop()

# 7. Création du graphique Matplotlib haut de gamme
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

# Gestion propre de la légende externe
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
    "Projection t-SNE des chiffres MNIST (Espace 2D non linéaire)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Dimension t-SNE 1", fontsize=10)
ax.set_ylabel("Dimension t-SNE 2", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Rendu Streamlit avec gestion anti-coupure
st.pyplot(fig, bbox_inches="tight")

# 8. Section explicative et pédagogique
st.markdown("---")
st.markdown("### 💡 Analyse & Interprétation :")
st.markdown(
    """
* **Séparation non linéaire :** Le **t-SNE** préserve les voisinages locaux, ce qui permet de détacher nettement chaque cluster de chiffre (de 0 à 9).
* **Impact de la perplexité :** Ajuster ce paramètre modifie le compromis entre l'attention portée aux voisinages locaux vs. globaux de vos données.
* **Optimisation Concurrente :** Grâce au mécanisme de cache intelligent (`st.cache_data`), l'application gère efficacement les accès multiples sans saturer la mémoire ni re-télécharger le dataset distant.
"""
)
