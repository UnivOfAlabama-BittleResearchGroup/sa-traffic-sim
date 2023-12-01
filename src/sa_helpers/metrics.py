from dataclasses import dataclass, MISSING
from pathlib import Path

import polars as pl

# mass to volume
SUMO_GASOLINE_GRAM_PER_LITER: float = 742.0  # gram / liter
SUMO_DIESEL_GRAM_PER_LITER: float = 836.0  # gram / liter

# mass to energy. From https://www.engineeringtoolbox.com/fossil-fuels-energy-content-d_1298.html
SUMO_GASOLINE_GRAM_TO_JOULE: float = (
    43.4 * 1e6 / 1e3
)  # MJ / kg * 1e6 J / MJ * 1 kg / 1e3 g = J / g
SUMO_DIESEL_GRAM_TO_JOULE: float = (
    42.8 * 1e6 / 1e3
)  # MJ / kg * 1e6 J / MJ * 1 kg / 1e3 g = J / g


def energy_2_fuel(
    df: pl.DataFrame,
) -> pl.DataFrame:
    return df.with_columns(
        total_fuel_l_cropped=pl.col("total_energy")
        / SUMO_GASOLINE_GRAM_TO_JOULE
        / SUMO_GASOLINE_GRAM_PER_LITER
    ).with_columns(
        cropped_per_vehicle_fuel=pl.col("total_fuel_l_cropped")
        / pl.col("total_vehicles_emissions")
    )


@dataclass
class SAObjectivesConfig:
    trip_info_file: Path
    detector_file: Path
    warmup_time: float
    total_fuel_l: float = MISSING
    average_fc: float = MISSING
    average_speed: float = MISSING
    average_delay: float = MISSING
    average_travel_time: float = MISSING
    delay_ratio: float = MISSING
    total_vehicles: int = MISSING


def fuel_to_j(
    trip_info_df: pl.DataFrame,
) -> float:
    return trip_info_df.with_columns(
        (
            pl.when(pl.col("vtype").str.contains("_D_"))
            .then(pl.lit(SUMO_DIESEL_GRAM_TO_JOULE))
            .otherwise(pl.lit(SUMO_GASOLINE_GRAM_TO_JOULE))
            * pl.col("fuel_abs").cast(float)
            * 1e-3  # mg -> g
        ).alias("fuel_j")
    )


def delay_ratio(config: SAObjectivesConfig) -> None:
    detector_df = (
        pl.read_parquet(config.detector_file)
        .filter(pl.col("id").str.contains("_logging"))
        .with_columns(
            pl.col("id").str.split("_").list.first().alias("tl"),
            pl.col("id")
            .str.split("_")
            .list.take(1)
            .list.first()
            .cast(int)
            .alias("detector"),
            pl.col("meanTimeLoss").cast(float),
        )
        .with_columns(
            pl.when(pl.col("detector").is_in([1, 2, 5, 6]))
            .then(pl.lit("mainline"))
            .otherwise(pl.lit("side_street"))
            .alias("dir")
        )
        .filter(pl.col("meanTimeLoss") >= 0)
    )

    config.delay_ratio = (
        detector_df.filter(pl.col("dir") == "mainline")["meanTimeLoss"].mean()
        / detector_df.filter(pl.col("dir") == "side_street")["meanTimeLoss"].mean()
    )


def compute_tripinfo_metrics(
    config: SAObjectivesConfig,
) -> None:
    trip_info_df = (
        pl.read_parquet(config.trip_info_file)
        .pipe(fuel_to_j)
        .with_columns(
            pl.col(["routeLength", "fuel_abs", "duration", "timeLoss", "depart"]).cast(
                float
            )
        )
        .filter(pl.col("depart") >= config.warmup_time)
    )

    config.total_fuel_l = (
        trip_info_df["fuel_j"].sum()
        / SUMO_GASOLINE_GRAM_TO_JOULE
        / SUMO_GASOLINE_GRAM_PER_LITER
    )
    config.average_fc = (
        (
            trip_info_df["fuel_j"].mean()
            / SUMO_GASOLINE_GRAM_TO_JOULE
            / SUMO_GASOLINE_GRAM_PER_LITER
        )
        / (trip_info_df["routeLength"].mean() / 1e3)  # m -> km
    ) * 100  # L / 100km

    config.average_speed = (
        trip_info_df["routeLength"] / trip_info_df["duration"]
    ).mean()

    config.average_delay = trip_info_df["timeLoss"].mean()

    config.average_travel_time = trip_info_df["duration"].mean()

    config.total_vehicles = trip_info_df["id"].n_unique()


def get_sa_results(
    config: SAObjectivesConfig,
    *args,
    **kwargs,
) -> None:
    # compute trip info metrics
    compute_tripinfo_metrics(config)

    # compute delay ratio
    delay_ratio(config)
