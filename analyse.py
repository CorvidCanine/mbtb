import mbtb
import argparse
import hdf5storage
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def clean_save_for_pandas(chimaera_save):
    # Need a copy so that we can remove chimaera_grid entry from
    # only the cleaned dictionary and not the original
    clean_dict = chimaera_save.copy()
    del clean_dict["chimaera_grid"]
    clean_dict["uniform_cell_width"] = chimaera_save["chimaera_grid"][
        "uniform_cell_width"
    ]
    clean_dict["chimaera_name"] = chimaera_save["chimaera_grid"]["name"]
    for grid in chimaera_save["chimaera_grid"]["grids"]:
        if grid["name"] == "base":
            # I'm only using the first cell width here, so assuming
            # the grid is uniform. TODO should deal with non-uniform
            # grids as well.
            clean_dict["base_cell_width"] = grid["cell_widths"][0]
            # There's only one 'base' grid, can leave the loop now
            break

    return clean_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Multi-Block Test Bench - Analysis script",
        description="Analyse parameter sweeps from Multi-Block Test Bench",
    )
    parser.add_argument(
        "path",
        help="Path to a dictionary to look for MBTB save files",
        type=Path,
    )

    args = parser.parse_args()

    xr_list = list()
    diff_xr_list = list()
    df = pd.DataFrame()
    save_files = list(args.path.rglob("*.h5"))
    # save_files = save_files[0:5]
    for file_path in tqdm(save_files, unit="files"):
        data = hdf5storage.read(filename=file_path.as_posix())
        # print(data)

        solved_grid = data["chimaera_grid"]["cont_solution"]
        solved_grid_positions = data["chimaera_grid"]["chont_solution_pos"]

        solved_xr = xr.DataArray(
            solved_grid,
            coords=[solved_grid_positions, data["chimaera_grid"]["result"]["t"]],
            dims=["pos", "time"],
        )

        # for other_xr in xr_list:
        #     # This assumes that the new grid has a the smaller cell width
        #     diff_xr_list.append(solved_xr - other_xr.interp_like(solved_xr))

        # data["index"] = len(xr_list)
        xr_list.append(solved_xr)
        df = df.append(clean_save_for_pandas(data), ignore_index=True)

    print(df)

    # After updating pandas from 1.2 to 1.3, the columns must be created before
    # trying to assign values to them when using xarray functions
    df["peak_diff"] = df["mean_diff"] = np.nan
    if True:
        # comparison_xr = xr_list[
        #     df[df["chimaera_name"] == "no_overlap_periodic"]["base_cell_width"].idxmin()
        # ]
        comparison_xr = xr_list[
            df[df["chimaera_name"] == "single_overlap"]["base_cell_width"].idxmin()
        ]
    else:
        comparison_xr = xr_list[df["base_cell_width"].idxmin()]

    for index, grid in df.iterrows():
        diff = xr_list[index].interp_like(comparison_xr) - comparison_xr
        df.at[index, "peak_diff"] = diff.max()
        df.at[index, "mean_diff"] = diff.mean()
        diff_xr_list.append(diff.max(dim="pos"))

        # diff.sel(time=0.001, method="nearest").plot(label=grid["base_cell_width"])
    # plt.legend()
    # plt.show()

    df["chimaera_name"] = df["chimaera_name"].astype("category")

    peak_axes = df.plot.scatter(
        x="base_cell_width", y="peak_diff", c="chimaera_name", cmap="viridis"
    )

    df.plot.scatter(
        x="base_cell_width", y="mean_diff", c="chimaera_name", cmap="viridis"
    )
    plt.show()

    # diff = xr_list[9] - xr_list[0].interp_like(xr_list[9])
    # print(diff)
    # diff.plot()
    # diff.sel(time=0.001, method="nearest").plot()
    # plt.show()
