import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import polars as pl
from typing import List, Tuple, Dict
import numpy as np
import plotly.express as px


from src.plotting.latex import NAME_2_LATEX

RYG_COLORSCALE = [
    "rgba(0, 147, 146, 0)",
    "rgba(114, 170, 161, 0.2)",
    "rgba(177, 199, 179, 0.4)",
    "rgba(241, 234, 200, 0.6)",
    "rgba(229, 185, 173, 0.8)",
    "rgba(217, 137, 148, 1)",
    "rgba(208, 88, 126, 1)",
]


def myround(x, base=5):
    return base * round(x / base)


def plot_bars(
    si_df: pl.DataFrame,
    plot_columns: List[str],
    pretty_columns: List[str],
    variable_column: str,
    sort_columns: List[str],
    use_latex: bool = True,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        start_cell="top-left",
        shared_yaxes=True,
        # shared_xaxes=True,
        subplot_titles=pretty_columns,
        vertical_spacing=0.075,
    )

    # max_y = myround(si_df.loc[idx[:], idx[:, 'ST']].max(axis=1).max() * 100 + 5, 5) / 100
    row_col = [(1, 1), (1, 2), (2, 1), (2, 2)]

    if use_latex:
        si_df = si_df.with_columns(
            pl.col(variable_column).map_dict(NAME_2_LATEX).alias("latex_name")
        )
    else:
        si_df = si_df.with_columns(pl.col(variable_column).alias("latex_name"))

    si_df = si_df.sort(
        sort_columns,
        descending=True,
    )

    for i, col in enumerate(plot_columns):
        print(col)

        max_y = (
            myround(
                (max(si_df[f"{col}_ST"]) + max(si_df[f"{col}_ST_conf"])) * 100 + 5, 5
            )
            / 100
        )

        fig.add_trace(
            go.Bar(
                x=si_df[f"{col}_S1"],
                y=si_df["latex_name"],
                name=r"$\large{S_1}$",
                orientation="h",
                marker=dict(
                    color="rgba(0, 0, 0, 0.4)",
                    line=dict(color="rgba(0, 0, 0, 1.0)", width=2),
                    pattern_shape="x",
                ),
                showlegend=i < 1,
            ),
            row=row_col[i][0],
            col=row_col[i][1],
        )

        fig.add_trace(
            go.Bar(
                x=si_df[f"{col}_ST"] - si_df[f"{col}_S1"],
                y=si_df["latex_name"],
                name=r"$\large{S_T}$",
                error_x=dict(
                    type="data",
                    array=si_df[f"{col}_ST_conf"].to_numpy(),
                    # make the color grey, but alpha 1
                    color="rgba(105,105,105, 1.0)",
                    # make it thicker
                    thickness=3,
                ),
                orientation="h",
                marker=dict(
                    color="rgba(0, 0, 0, 0.8)",
                    line=dict(color="rgba(0, 0, 0, 1.0)", width=2),
                ),
                showlegend=i < 1,
            ),
            row=row_col[i][0],
            col=row_col[i][1],
        )

        fig.update_layout(
            **{
                ("yaxis" if i < 1 else f"yaxis{i+1}"): dict(
                    # dtick=0.1, tickangle=45,
                    showgrid=True,
                    tickvals=si_df["latex_name"],
                    # range=[-0.05, max_y],
                    minor_showgrid=True,
                    # tickfont_size=22
                    # title=col
                ),
            },
            **{
                ("xaxis" if i < 1 else f"xaxis{i+1}"): dict(
                    # dtick=0.1, tickangle=45,
                    showgrid=True,  # title=col
                    minor_showgrid=True,
                    range=[0, max_y],
                ),
            },
        )

        # break

    fig.update_layout(
        template="simple_white",
        # font_family="helvetica",
        font_family="Open Sans",
        title_font_family="Open Sans",
        title_font_size=24,
        font_size=24,
        height=1200,
        width=800,
        barmode="stack",
        # bargap=0.15,
        # legend=dict(yanchor="top", y=0.95, xanchor="right", x=1, orientation="h"),
        margin=dict(l=50, r=50, b=20, t=50, pad=4),
        bargap=0.15,
    )

    # # fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text}</b>'))
    fig.update_annotations(font=dict(family="Open Sans", size=24))

    return fig
    # # fig.write_html("tmp.html", include_plotlyjs="cdn", include_mathjax="cdn")
    # fig.write_image("ST_all_New.png", scale=4)
    # fig.show()


def plot_si_v_n(
    results: pl.DataFrame,
    plot_columns: List[str],
    plot_variables: List[str],
    pretty_columns: List[str],
) -> go.Figure:
    fig = go.Figure()

    colorscale = [
        "rgba(229, 185, 173, 1)",
        "rgba(177, 199, 179, 1)",
        "rgba(241, 234, 200, 1)",
        "rgba(0, 147, 146, 1)",
        "rgba(114, 170, 161, 1)",
        "rgba(229, 185, 173, 1)",
        "rgba(217, 137, 148, 1)",
        "rgba(208, 88, 126, 1)",
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        start_cell="top-left",
        shared_yaxes=True,
        shared_xaxes=True,
        vertical_spacing=0.075,
        horizontal_spacing=0.05,
        subplot_titles=pretty_columns,
    )
    row_col = [(1, 1), (1, 2), (2, 1), (2, 2)]
    target_ci = False

    results = results.sort(
        "N",
    )

    for plot_col, display_col, idx in zip(plot_columns, pretty_columns, row_col):
        for var in plot_variables:
            plot_df = results.filter(pl.col("names") == var)

            fig.add_trace(
                go.Scatter(
                    x=plot_df["N"],
                    y=plot_df[plot_col] - plot_df[f"{plot_col}_conf"],
                    name=var,
                    showlegend=False,
                    line=dict(color=colorscale[plot_variables.index(var)]),
                ),
                row=idx[0],
                col=idx[1],
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_df["N"],
                    y=plot_df[plot_col] + plot_df[f"{plot_col}_conf"],
                    name=NAME_2_LATEX[var] if var in NAME_2_LATEX else var,
                    showlegend=False,
                    line=dict(color=colorscale[plot_variables.index(var)]),
                    fill="tonexty",
                ),
                row=idx[0],
                col=idx[1],
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_df["N"],
                    y=plot_df[plot_col],
                    name=NAME_2_LATEX[var] if var in NAME_2_LATEX else var,
                    showlegend=sum(idx) == 2,
                    line=dict(color=colorscale[plot_variables.index(var)]),
                ),
                row=idx[0],
                col=idx[1],
            )

            if (
                plot_df[plot_col].max()
                >= results.filter(pl.col("N") == plot_df["N"].max())[plot_col].max()
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[plot_df["N"].max() - 2_000, plot_df["N"].max() + 2_000],
                        y=[
                            plot_df[plot_col][-1] + plot_df[plot_col][-1] * 0.1,
                        ]
                        * 2,
                        name=r"$\large{\text{Target CI}}$",
                        showlegend=not target_ci,
                        line=dict(
                            color="crimson",
                            width=4,
                        ),  # dash="dash"),
                        mode="lines",
                        legendrank=100000,
                    ),
                    row=idx[0],
                    col=idx[1],
                    # fill="tonexty"
                )

                target_ci = True

                fig.add_trace(
                    go.Scatter(
                        x=[plot_df["N"].max() - 2_000, plot_df["N"].max() + 2_000],
                        y=[
                            plot_df[plot_col][-1] - plot_df[plot_col][-1] * 0.1,
                        ]
                        * 2,
                        showlegend=False,
                        line=dict(
                            color="crimson",
                            width=4,
                        ),  # dash="dash"),
                        mode="lines",
                    ),
                    row=idx[0],
                    col=idx[1],
                    # fill="tonexty"
                )

            fig.update_layout(
                **{
                    "xaxis{}".format(sum(idx)): dict(
                        title="Number of Simulations" if idx[0] == 2 else "",
                        dtick=10_000,
                        # tickformat="d",
                        tickangle=45,
                        # showexponent="all",
                        # exponentformat="e",
                    ),
                    "yaxis{}".format(sum(idx) - 1): dict(
                        title=r"Sensitivity Index" if idx[1] == 0 else "",
                        range=[0, 1],
                    ),
                    # "yaxis": dict(title=r"Sensitivity Index", range=[0, 1]),
                }
                # tickvals=tickvals,
            )

    fig.update_layout(
        template="simple_white",
        font_family="Open Sans",
        font_size=22,
        height=800,
        width=800,
        margin=dict(l=50, r=50, b=20, t=20, pad=4),
        legend=dict(yanchor="top", y=-0.15, xanchor="left", x=0, orientation="h"),
    )

    fig.update_annotations(font=dict(family="Open Sans", size=24))

    fig.update_layout(
        yaxis=dict(
            title=r"Sensitivity Index",
            range=[0, 1],
        ),
        yaxis3=dict(
            title=r"Sensitivity Index",
            range=[0, 1],
        ),
    )

    return fig


def plot_ecdfs(
    plot_columns: List[str],
    pretty_columns: List[str],
    results_df: pl.DataFrame = None,
    paper_mapping: Dict[str, Tuple[pl.DataFrame, str]] = None,
    ecdf_mapping: Dict[str, Tuple[pl.DataFrame, str]] = None,
):
    fig = make_subplots(
        rows=2, cols=2, start_cell="top-left", shared_yaxes=True, vertical_spacing=0.2
    )

    if ecdf_mapping is None and results_df is not None:
        ecdf_mapping = {
            "SA Sample": (results_df, "black"),
        }

    for i, (p_name, name) in enumerate(
        zip(
            pretty_columns,
            plot_columns,
        )
    ):
        if paper_mapping is not None:
            # plot the paper ranges
            for p_paper_name, (df, color) in paper_mapping.items():
                fig.add_trace(
                    go.Scatter(
                        x=[df[name].quantile(0.05), df[name].quantile(0.05)],
                        y=(0, 1),
                        line_color=color,
                        showlegend=False,
                    ),
                    row=i // 2 + 1,
                    col=i % 2 + 1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=[df[name].quantile(0.95), df[name].quantile(0.95)],
                        y=(0, 1),
                        line_color=color,
                        name=p_paper_name,
                        showlegend=i < 1,
                        fill="tonextx",
                    ),
                    row=i // 2 + 1,
                    col=i % 2 + 1,
                )

        for sa_name, (df, color) in ecdf_mapping.items():
            # plot the SA samples as an ecdf
            x, counts = np.unique(df[name], return_counts=True)
            y = np.cumsum(counts)
            _tmp_df = pd.DataFrame(
                data=[np.insert(x, 0, x[0]), np.insert(y / y[-1], 0, 0.0)]
            ).T

            skipper = 1 if len(_tmp_df) < 10000 else 10

            fig.add_trace(
                go.Scatter(
                    x=_tmp_df[0].values[::skipper],
                    y=_tmp_df[1].values[::skipper],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=5,
                        symbol="circle",
                    ),
                    showlegend=i < 1,
                    name=sa_name,
                ),
                row=i // 2 + 1,
                col=i % 2 + 1,
            )
            fig.update_layout(
                **{
                    "xaxis" if i < 1 else f"xaxis{i+1}": dict(
                        title=p_name,
                        nticks=6,
                        showgrid=True,
                        minor_showgrid=True,
                        tickfont=dict(
                            size=16,
                        ),
                    ),
                    "yaxis" if i < 1 else f"yaxis{i+1}": dict(
                        showgrid=True,
                        minor_showgrid=True,
                        ticks="outside" if i % 2 < 1 else "inside",
                        title="Empirical cdf F(x)" if i % 2 < 1 else None,
                        tickfont=dict(
                            size=16,
                        ),
                    ),
                }
            )

    fig.update_layout(
        template="simple_white",
        font_family="Open Sans",
        font_size=22,
        height=600,
        width=800,
        margin=dict(l=50, r=50, b=20, t=20, pad=4),
        legend=dict(yanchor="bottom", y=-0.38, xanchor="left", x=0, orientation="h"),
    )

    fig.update_annotations(font=dict(family="Open Sans", size=22))

    return fig


def plot_parallel_coordinates(
    results_df: pl.DataFrame,
    plot_cols: List[str],
    metric_col: str,
    pretty_metric: str,
):
    plot_df = results_df

    fig = px.parallel_coordinates(
        plot_df.select([metric_col] + list(plot_cols)).to_pandas(),
        dimensions=list(plot_cols) + [metric_col],
        color=metric_col,
        color_continuous_scale=RYG_COLORSCALE,
    )

    for i, (dim, name) in enumerate(
        zip(fig.data[0]["dimensions"], list(plot_cols) + [metric_col])
    ):
        try:
            label = NAME_2_LATEX[name]
            dim.label = ""
        except KeyError:
            dim.label = ""
            label = pretty_metric

        fig.add_annotation(
            text=label,
            xref="x domain",
            yref="y domain",
            x=0.25 * i,
            y=1.08 if label == pretty_metric else 1.05,
            # showarrow=True,
            # If axref is exactly the same as xref, then the text's position is
            # absolute and specified in the same coordinates as xref.
            axref="x domain",
            # The same is the case for yref and ayref, but here the coordinates are data
            # coordinates
            ayref="y domain",
            ax=0,
            ay=0,
        )

        base = 1 if name == metric_col else 0.01

        dim.range = [
            myround(results_df[name].min(), base),
            myround(results_df[name].max(), base),
        ]
        dim.tickvals = [str(v) for v in np.linspace(dim.range[0], dim.range[1], 6)]

    fig.update_coloraxes(showscale=False)

    fig.update_layout(
        template="simple_white",
        font_family="Open Sans",
        font_size=24,
        height=600,
        width=800,
        margin=dict(l=60, r=100, b=20, t=100, pad=4),
        # legend=dict(yanchor="bottom", y=1, xanchor="right", x=1, orientation="h")
    )
    fig.update_annotations(font=dict(family="Open Sans", size=24))
    return fig
