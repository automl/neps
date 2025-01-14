"""Plot a 3D landscape of learning curves for a given run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import (
    cm,
    pyplot as plt,
)
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Copied from plot.py
HERE = Path(__file__).parent.absolute()


@dataclass
class Plotter3D:
    """Plot a 3d landscape of learning curves for a given run."""

    objective_to_minimize_key: str = "Objective to minimize"
    fidelity_key: str = "epochs"
    run_path: str | Path | None = None
    scatter: bool = True
    footnote: bool = True
    alpha: float = 0.9
    scatter_size: float | int = 3
    bck_color_2d: tuple[float, float, float] = (0.8, 0.82, 0.8)
    view_angle: tuple[float, float] = (15, -70)

    def __post_init__(self) -> None:
        if self.run_path is not None:
            assert (
                Path(self.run_path).absolute().is_dir()
            ), f"Path {self.run_path} is not a directory"
            self.data_path = (
                Path(self.run_path).absolute() / "summary_csv" / "config_data.csv"
            )
            assert self.data_path.exists(), f"File {self.data_path} does not exist"
            self.df = pd.read_csv(  # type: ignore
                self.data_path,
                index_col=0,
                float_precision="round_trip",  # type: ignore
            )

            # Assigned at prep_df stage
            self.objective_to_minimize_range: tuple[float, float] | None = None
            self.epochs_range: tuple[float, float] | None = None

    @staticmethod
    def get_x(df: pd.DataFrame) -> np.ndarray:
        """Get the x-axis values for the plot."""
        return df["epochID"].to_numpy()  # type: ignore

    @staticmethod
    def get_y(df: pd.DataFrame) -> np.ndarray:
        """Get the y-axis values for the plot."""
        y_ = df["configID"].to_numpy()
        return np.ones_like(y_) * y_[0]  # type: ignore

    @staticmethod
    def get_z(df: pd.DataFrame) -> np.ndarray:
        """Get the z-axis values for the plot."""
        return df["result.objective_to_minimize"].to_numpy()  # type: ignore

    @staticmethod
    def get_color(df: pd.DataFrame) -> np.ndarray:
        """Get the color values for the plot."""
        return df.index.to_numpy()  # type: ignore

    def prep_df(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Prepare the dataframe for plotting."""
        df = self.df if df is None else df

        _fid_key = f"config.{self.fidelity_key}"
        self.objective_to_minimize_range = (
            df["result.objective_to_minimize"].min(),
            df["result.objective_to_minimize"].max(),
        )  # type: ignore
        self.epochs_range = (df[_fid_key].min(), df[_fid_key].max())  # type: ignore

        split_values = np.array([[*index.split("_")] for index in df.index])
        df[["configID", "epochID"]] = split_values
        df.configID = df.configID.astype(int)
        df.epochID = df.epochID.astype(int)
        if df.epochID.min() == 0:
            df.epochID += 1

        # indices become sampling order
        time_cols = ["metadata.time_started", "metadata.time_end"]
        return df.sort_values(by=time_cols).reset_index(drop=True)

    def plot3D(  # noqa: N802, PLR0915
        self,
        data: pd.DataFrame | None = None,
        save_path: str | Path | None = None,
        filename: str = "freeze_thaw",
    ) -> None:
        """Plot the 3D landscape of learning curves."""
        data = self.prep_df(data)

        # Create the figure and the axes for the plot
        fig, (ax3D, ax, cax) = plt.subplots(
            1, 3, figsize=(12, 5), width_ratios=(20, 20, 1)
        )

        # remove a 2D axis and replace with a 3D projection one
        ax3D.remove()
        ax3D = fig.add_subplot(131, projection="3d")

        # Create the normalizer to normalize the color values
        norm = Normalize(self.get_color(data).min(), self.get_color(data).max())

        # Counters to keep track of the configurations run for only a single fidelity
        n_lines = 0
        n_points = 0

        data_groups = data.groupby("configID", sort=False)

        for idx, (_configID, data_) in enumerate(data_groups):
            x = self.get_x(data_)
            y = self.get_y(data_)
            z = self.get_z(data_)

            y = np.ones_like(y) * idx
            color = self.get_color(data_)

            if len(x) < 2:
                n_points += 1
                if self.scatter:
                    # 3D points
                    ax3D.scatter(
                        y,
                        z,
                        s=self.scatter_size,
                        zs=0,
                        zdir="x",
                        c=color,
                        cmap="RdYlBu_r",
                        norm=norm,
                        alpha=self.alpha * 0.8,
                    )
                    # 2D points
                    ax.scatter(
                        x,
                        z,
                        s=self.scatter_size,
                        c=color,
                        cmap="RdYlBu_r",
                        norm=norm,
                        alpha=self.alpha * 0.8,
                    )
            else:
                n_lines += 1

                # Plot 3D
                # Get segments for all lines
                points3D = np.array([x, y, z]).T.reshape(-1, 1, 3)
                segments3D = np.concatenate([points3D[:-1], points3D[1:]], axis=1)

                # Construct lines from segments
                lc3D = Line3DCollection(
                    segments3D,  # type: ignore
                    cmap="RdYlBu_r",
                    norm=norm,
                    alpha=self.alpha,
                )
                lc3D.set_array(color)

                # Draw lines
                ax3D.add_collection3d(lc3D)  # type: ignore

                # Plot 2D
                # Get segments for all lines
                points = np.array([x, z]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Construct lines from segments
                lc = LineCollection(
                    segments,  # type: ignore
                    cmap="RdYlBu_r",
                    norm=norm,
                    alpha=self.alpha,  # type: ignore
                )
                lc.set_array(color)

                # Draw lines
                ax.add_collection(lc)

        assert self.objective_to_minimize_range is not None
        assert self.epochs_range is not None

        ax3D.axes.set_xlim3d(left=self.epochs_range[0], right=self.epochs_range[1])  # type: ignore
        ax3D.axes.set_ylim3d(bottom=0, top=data_groups.ngroups)  # type: ignore
        ax3D.axes.set_zlim3d(  # type: ignore
            bottom=self.objective_to_minimize_range[0],
            top=self.objective_to_minimize_range[1],
        )  # type: ignore

        ax3D.set_xlabel("Epochs")
        ax3D.set_ylabel("Iteration sampled")
        ax3D.set_zlabel(f"{self.objective_to_minimize_key}")  # type: ignore

        # set view angle
        ax3D.view_init(elev=self.view_angle[0], azim=self.view_angle[1])  # type: ignore

        ax.autoscale_view()
        ax.set_xlabel(self.fidelity_key)
        ax.set_ylabel(f"{self.objective_to_minimize_key}")
        ax.set_facecolor(self.bck_color_2d)
        fig.suptitle("ifBO run")

        if self.footnote:
            fig.text(
                0.01,
                0.02,
                f"Total {n_lines + n_points} configs evaluated; for multiple budgets: "
                f"{n_lines}, for single budget: {n_points}",
                ha="left",
                va="bottom",
                fontsize=10,
            )

        plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap="RdYlBu_r"),
            cax=cax,
            label="Iteration",
            use_gridspec=True,
            alpha=self.alpha,
        )
        fig.tight_layout()

        self.save(save_path, filename)
        plt.close(fig)

    def save(
        self,
        save_path: str | Path | None = None,
        filename: str = "freeze_thaw",
    ) -> None:
        """Save the plot to a file."""
        path = save_path if save_path is not None else self.run_path
        assert path is not None

        run_path = Path(path)
        run_path.mkdir(parents=True, exist_ok=True)
        assert run_path.is_dir()
        plot_path = run_path / f"Plot3D_{filename}.png"

        plt.savefig(plot_path, bbox_inches="tight")


if __name__ == "__main__":
    plotter = Plotter3D(run_path="./results", fidelity_key="epochs")
    plotter.plot3D()
