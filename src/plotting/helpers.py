from plotly import graph_objects as go
from IPython.display import Image


def plotly_2_image(fig: go.Figure, width: int = 800, height: int = 1200, scale: int = 1) -> Image:
    return Image(fig.to_image(format="png", width=width, height=height, scale=scale))
