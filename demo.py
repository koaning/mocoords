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
    d["color"] = X[:, 0]
    return ParallelCoordinates, X, d, make_blobs, mo, np, pl, random, y


@app.cell
def _(ParallelCoordinates, d, mo, pl):
    widget = mo.ui.anywidget(ParallelCoordinates(pl.DataFrame(d)))
    widget
    return (widget,)


@app.cell
def _(pl, widget):
    pl.DataFrame(widget.selection)
    return


@app.cell
def _(pl):
    from model2vec import StaticModel

    model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    df_spam = pl.read_csv("spam.csv")
    return StaticModel, df_spam, model


@app.cell
def _(df_spam, model):
    texts = df_spam["text"].to_list()
    embeddings = model.encode(texts)
    return embeddings, texts


@app.cell
def _():
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    return PCA, cosine_similarity


@app.cell
def _(PCA, embeddings):
    X_tfm = PCA(n_components=10).fit_transform(embeddings)
    return (X_tfm,)


@app.cell
def _(mo):
    text_ui = mo.ui.text(label="query")
    text_ui
    return (text_ui,)


@app.cell
def _(X_tfm, cosine_similarity, embeddings, model, pl, text_ui):
    d_pca = {f"v{i}": X_tfm[:, i] for i in range(X_tfm.shape[1])}
    emb_query = model.encode([text_ui.value])
    d_pca["color"] = cosine_similarity(emb_query, embeddings)[0]
    pl.DataFrame(d_pca)
    return d_pca, emb_query


@app.cell
def _(ParallelCoordinates, d_pca, mo, pl):
    widget_pca = mo.ui.anywidget(ParallelCoordinates(pl.DataFrame(d_pca)))
    widget_pca
    return (widget_pca,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
