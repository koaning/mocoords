from pathlib import Path
import anywidget
import traitlets
import polars as pl


class ParallelCoordinates(anywidget.AnyWidget):
    """
    A parallel coordinates widget that allows to advances data selection of embeddings.
    """
    _esm = Path(__file__).parent / 'static' / 'parcoords.js'
    _css = Path(__file__).parent / 'static' / 'parcoords.css'
    data = traitlets.List([]).tag(sync=True)
    selection = traitlets.List([]).tag(sync=True)

    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        if isinstance(data, pl.DataFrame):
            self.data = data.to_dicts()
        else:
            self.data = data

    @property
    def data_as_pandas(self):
        import pandas as pd 
        return pd.DataFrame(self.data)

    @property
    def data_as_polars(self):
        import polars as pl
        return pl.DataFrame(self.data)
