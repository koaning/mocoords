from pathlib import Path
import anywidget
import traitlets
from IPython.display import IFrame


class ParallelCoordinates(anywidget.AnyWidget):
    """
    A parallel coordinates widget that allows to advances data selection of embeddings.
    """
    _esm = Path(__file__).parent / 'static' / 'parcoords.js'
    _css = Path(__file__).parent / 'static' / 'parcoords.css'
    data = traitlets.List([]).tag(sync=True)

    @property
    def data_as_pandas(self):
        import pandas as pd 
        return pd.DataFrame(self.data)

    @property
    def data_as_polars(self):
        import polars as pl
        return pl.DataFrame(self.data)
