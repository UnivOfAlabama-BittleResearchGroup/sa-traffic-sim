import click
from pathlib import Path
from omegaconf import OmegaConf
import polars as pl
from sumo_pipelines.utils.file_helpers import walk_directory


@click.command()
@click.argument("experiment_root", type=click.Path(exists=True))
def main(experiment_root):
    experiment_root = Path(experiment_root)

    results_df_path = experiment_root / "results.parquet"

    if not results_df_path.exists():
        confs = list(
            walk_directory(
                experiment_root,
            )
        )
        confs.sort(key=lambda x: int(x.Metadata.run_id))

        results_df = (
            pl.from_records(
                [
                    {
                        "run_id": conf.Metadata.run_id,
                        "calibration_passed": (
                            conf.Pipeline.pipeline[0]
                            .consumers[7]
                            .config.calibration_passed
                        ),
                        **OmegaConf.to_container(
                            conf.Pipeline.pipeline[0].consumers[8].config
                        ),
                        **{
                            "total_vehicles_emissions": conf.Blocks.FuelTotalConfig.total_vehicles,
                            "total_energy": conf.Blocks.FuelTotalConfig.total_energy,
                        },
                    }
                    for conf in confs
                ]
            )
            .with_columns(
                pl.col("run_id").cast(pl.UInt32),
            )
            .drop(["detector_file", "trip_info_file", "warmup_time"])
        )

        # save the results
        results_df.write_parquet(experiment_root / "results.parquet")


if __name__ == "__main__":
    main()
