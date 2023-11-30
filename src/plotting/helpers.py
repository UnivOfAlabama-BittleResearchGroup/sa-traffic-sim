from plotly import graph_objects as go
from IPython.display import Image


def plotly_2_image(fig: go.Figure, width: int = 800, height: int = 1200) -> Image:
    Image(fig.to_image(format="png", width=800, height=1200))
