import marimo

__generated_with = "0.12.6"
app = marimo.App()


@app.cell
def _():
    import random 

    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.datasets import make_blobs

    from mocoords import ParallelCoordinates

    X, y = make_blobs(n_samples=50_000, n_features=10, centers=5)
    d = {f"v{i}": X[:, i] for i in range(X.shape[1])}
    d["color"] = y
    return ParallelCoordinates, X, d, make_blobs, mo, np, pl, random, y


@app.cell
def _(ParallelCoordinates, d, mo, pl):
    widget = mo.ui.anywidget(ParallelCoordinates(pl.DataFrame(d)))
    widget
    return (widget,)


@app.cell
def _(pl, widget):
    out = None
    if widget.selection:
        out = pl.DataFrame(widget.selection["data"])
    out
    return (out,)


@app.cell
def _(pl):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    df_spam = pl.read_csv("spam.csv")
    return SentenceTransformer, df_spam, model


@app.cell
def _(df_spam, model):
    texts = df_spam["text"].to_list()
    embeddings = model.encode(texts)
    return embeddings, texts


@app.cell
def _():
    from sklearn.decomposition import PCA
    from umap import UMAP
    from sklearn.metrics.pairwise import cosine_similarity
    return PCA, UMAP, cosine_similarity


@app.cell
def _(PCA, UMAP, embeddings, method, ndim_ui):
    if method.value == "UMAP":
        X_tfm = UMAP(n_components=ndim_ui.value).fit_transform(embeddings)
    else:
        X_tfm = PCA(n_components=ndim_ui.value).fit_transform(embeddings)
    return (X_tfm,)


@app.cell
def _(mo):
    methods = ["PCA", "UMAP"]
    text_ui = mo.ui.text(label="query")
    ndim_ui = mo.ui.slider(2, 10, 1, label="dimensions")
    method = mo.ui.dropdown(options=methods, value=methods[1])
    mo.hstack([text_ui, ndim_ui, method])
    return method, methods, ndim_ui, text_ui


@app.cell
def _(X_tfm, cosine_similarity, embeddings, model, text_ui):
    def norm(x):
        return (x - x.min())/(x.max() - x.min())

    d_pca = {f"v{i}": X_tfm[:, i] for i in range(X_tfm.shape[1])}
    emb_query = model.encode([text_ui.value])
    d_pca["color"] = cosine_similarity(emb_query, embeddings)[0]
    return d_pca, emb_query, norm


@app.cell
def _(ParallelCoordinates, d_pca, mo, pl):
    widget_pca = mo.ui.anywidget(ParallelCoordinates(pl.DataFrame(d_pca)))
    widget_pca
    return (widget_pca,)


@app.cell
def _(div, p, texts, widget_pca):
    div(*[p(texts[i]) for i in widget_pca.get_indices()][:50])
    return


@app.cell
def _():
    from mohtml import p, div
    return div, p


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
