from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path
import multiprocessing as mp
from functools import partial

from argparse import ArgumentParser

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import matplotlib
matplotlib.use('TkAgg')

import itertools

from neps.status.status import get_run_summary_csv
import re
import pandas as pd
import numpy as np

from typing import Callable

# Copied from plot.py
HERE = Path(__file__).parent.absolute()
DEFAULT_RESULTS_PATH = HERE.parent / "results"


@dataclass
class Plotter3D:
    loss_key: str = "Loss"
    fidelity_key: str = "epochs"
    config_column: str | None = None
    run_path: str | Path | None = None
    base_results_path: str | Path = DEFAULT_RESULTS_PATH
    strict: bool = False
    get_x: Callable[[pd.DataFrame], np.array] | None = None
    get_y: Callable[[pd.DataFrame], np.array] | None = None
    get_z: Callable[[pd.DataFrame], np.array] | None = None
    get_color: Callable[[pd.DataFrame], np.array] | None = None
    scatter: bool = True
    footnote: bool = True
    alpha: float = 0.9
    scatter_size: float | int = 3
    bck_color_2d: tuple[float] = (0.8, 0.82, 0.8)
    view_angle: tuple[float | int] = (15, -70)

    def __post_init__(self):
        if self.run_path is not None:
            assert Path(self.run_path).absolute().is_dir(), \
                f"Path {self.run_path} is not a directory"
            self.data_path = Path(self.run_path).absolute() / "summary_csv" / "config_data.csv"
            assert self.data_path.exists(), f"File {self.data_path} does not exist"
            self.df = pd.read_csv(self.data_path, index_col=0, float_precision="round_trip")

            self.loss_range = (self.df["result.loss"].min(), self.df["result.loss"].max())
            _fid_key = f"config.{self.fidelity_key}"
            self.epochs_range = (self.df[_fid_key].min(), self.df[_fid_key].max())

    @staticmethod
    def get_x(df: pd.DataFrame) -> np.array:
        return df["epochID"].to_numpy()

    @staticmethod
    def get_y(df: pd.DataFrame) -> np.array:
        y_ = df["configID"].to_numpy()
        return np.ones_like(y_) * y_[0]

    @staticmethod
    def get_z(df: pd.DataFrame) -> np.array:
        return df["result.loss"].to_numpy()

    @staticmethod
    def get_color(df: pd.DataFrame) -> np.array:
        return df.index.to_numpy()

    def prep_df(self, df: pd.DataFrame = None) -> pd.DataFrame:
        df = self.df if df is None else df
        time_cols = ["metadata.time_started", "metadata.time_end"]
        df = df.sort_values(by=time_cols).reset_index(drop=True)
        split_values = np.array([[*index.split('_')] for index in self.df.index])
        df[['configID', 'epochID']] = split_values
        df.configID = df.configID.astype(int)
        df.epochID = df.epochID.astype(int)
        if df.epochID.min() == 0:
            df.epochID += 1
        return df

    def plot3D(
        self,
        data: pd.DataFrame = None,
        save_path: str | Path | None = None,
        filename: str = "freeze_thaw"
    ):
        data = self.prep_df(data)

        # Create the figure and the axes for the plot
        fig, (ax3D, ax, cax) = plt.subplots(1, 3, figsize=(12, 5), width_ratios=(20, 20, 1))

        # remove a 2D axis and replace with a 3D projection one
        ax3D.remove()
        ax3D = fig.add_subplot(131, projection='3d')

        # Create the normalizer to normalize the color values
        norm = Normalize(self.get_color(data).min(), self.get_color(data).max())

        # Counters to keep track of the configurations run for only a single fidelity
        n_lines = 0
        n_mins = 0

        data_groups = data.groupby("configID", sort=False)

        for idx, (configID, data_) in enumerate(data_groups):

            x = self.get_x(data_)
            y = self.get_y(data_)
            z = self.get_z(data_)

            y = np.ones_like(y) * idx
            color = self.get_color(data_)

            if len(x) < 2:
                n_mins += 1
                if self.scatter:
                    ax3D.scatter(
                        y,
                        z,
                        s=self.scatter_size, 
                        zs=0, 
                        zdir="x",
                        c=color,
                        cmap='RdYlBu_r',
                        norm=norm,
                        alpha=self.alpha * 0.8
                    )
                    ax.scatter(
                        x,
                        z,
                        s=self.scatter_size,
                        c=color,
                        cmap='RdYlBu_r',
                        norm=norm,
                        alpha=self.alpha * 0.8
                    )
            else:
                n_lines += 1

                # Plot 3D
                # Get segments for all lines
                points3D = np.array([x, y, z]).T.reshape(-1, 1, 3)
                segments3D = np.concatenate([points3D[:-1], points3D[1:]], axis=1)

                # Construct lines from segments
                lc3D = Line3DCollection(segments3D, cmap='RdYlBu_r', norm=norm, alpha=self.alpha)
                lc3D.set_array(color)

                # Draw lines
                ax3D.add_collection3d(lc3D)

                # Plot 2D
                # Get segments for all lines
                points = np.array([x, z]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Construct lines from segments
                lc = LineCollection(segments, cmap="RdYlBu_r", norm=norm, alpha=self.alpha)
                lc.set_array(color)

                # Draw lines
                ax.add_collection(lc)

        ax3D.axes.set_xlim3d(left=self.epochs_range[0], right=self.epochs_range[1])
        ax3D.axes.set_ylim3d(bottom=0, top=data_groups.ngroups)
        ax3D.axes.set_zlim3d(bottom=self.loss_range[0], top=self.loss_range[1])

        ax3D.set_xlabel('Epochs')
        ax3D.set_ylabel('Iteration sampled')
        ax3D.set_zlabel(f'{self.loss_key}')

        # set view angle
        ax3D.view_init(elev=self.view_angle[0], azim=self.view_angle[1])

        ax.autoscale_view()
        ax.set_xlabel(self.fidelity_key)
        ax.set_ylabel(f'{self.loss_key}')
        ax.set_facecolor(self.bck_color_2d)
        fig.suptitle("ifBO run")

        if self.footnote:
            fig.text(
                0.01, 0.02,
                f"Total {n_lines + n_mins} configs evaluated; for multiple budgets: "
                f"{n_lines}, for single budget: {n_mins}",
                ha='left',
                va="bottom",
                fontsize=10
            )

        plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap="RdYlBu_r"),
            cax=cax,
            label='Iteration',
            use_gridspec=True,
            alpha=self.alpha
        )
        fig.tight_layout()

        self.save(save_path, filename)
        plt.close(fig)

    def save(self, save_path: str | Path | None = None, filename: str = "freeze_thaw"):
        run_path = Path(save_path if save_path is not None else self.run_path)
        run_path.mkdir(parents=True, exist_ok=True)
        assert run_path.is_dir()
        plot_path = run_path / f"Plot3D_{filename}.png"
        
        plt.savefig(
            plot_path,
            bbox_inches='tight'
        )


if __name__ == "__main__":
    plotter = Plotter3D(
        run_path="./results",
        fidelity_key="epochs"
    )
    plotter.plot3D()
