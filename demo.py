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

    X, y = make_blobs(n_samples=10_000, n_features=10, centers=5)
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
    pl.DataFrame(widget.selection)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
