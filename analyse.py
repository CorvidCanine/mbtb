from pandas.io.formats.format import DataFrameFormatter
import mbtb
import argparse
import hdf5storage
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import linregress


def clean_save_for_pandas(chimaera_save):
    # Need a copy so that we can remove chimaera_grid entry from
    # only the cleaned dictionary and not the original
    clean_dict = chimaera_save.copy()
    del clean_dict["chimaera_grid"]
    # clean_dict["uniform_cell_width"] = chimaera_save["chimaera_grid"][
    #     "uniform_cell_width"
    # ]
    clean_dict["chimaera_name"] = chimaera_save["chimaera_grid"]["name"]
    clean_dict["solver_method"] = chimaera_save["chimaera_grid"]["solver_method"]
    clean_dict["solver_time"] = chimaera_save["chimaera_grid"]["solver_time"]
    clean_dict["end_time"] = chimaera_save["chimaera_grid"]["solver_time_span"][1]

    for grid in chimaera_save["chimaera_grid"]["grids"]:
        if grid["name"] == "base":
            # I'm only using the first cell width here, so assuming
            # the grid is uniform. TODO should deal with non-uniform
            # grids as well.
            clean_dict["base_cell_width"] = grid["cell_widths"][0]
            # There's only one 'base' grid, can leave the loop now
        elif grid["name"] == "overlap":
            clean_dict["overlap_cell_width"] = grid["cell_widths"][0]

    try:
        if len(chimaera_save["chimaera_grid"]["overlaps"]) > 0:
            clean_dict["interp_kind"] = chimaera_save["chimaera_grid"]["overlaps"][0][
                "interp_kind"
            ]
            clean_dict["interface_width"] = chimaera_save["chimaera_grid"]["overlaps"][
                0
            ]["interface_width"]

        if len(chimaera_save["chimaera_grid"]["overlaps"]) > 1:
            clean_dict["interp_kind_second"] = chimaera_save["chimaera_grid"][
                "overlaps"
            ][1]["interp_kind"]
            clean_dict["interface_width_second"] = chimaera_save["chimaera_grid"][
                "overlaps"
            ][1]["interface_width"]
    except KeyError:
        pass
    # for over in chimaera_save["chimaera_grid"]["overlaps"]:
    #     if over["name"] == "overlap":
    #         clean_dict["interp_kind"] = over["interp_kind"]
    #         clean_dict["interface_width"] = over["interface_width"]
    #     else:
    #         clean_dict["interp_kind_" + over["name"]] = over["interp_kind"]
    #         clean_dict["interface_width_" + over["name"]] = over["interface_width"]

    return clean_dict


def exp_bkg(x, a, b, c, d, l, k):
    return a * np.exp(d + (b * x)) + (x - l) / (k - l)


def exponential(x, a, b, c):
    # An exponential function function
    return a * np.exp(-b * x) + c


def other_exp(x, a, b):
    # An exponential function function
    return a * np.power(x, b)


def dexponential(x, a, b, c, q, w, e):
    # An exponential function function
    return (a * np.exp(b * x) + c) * (q * np.exp(w * x) + e)


def logistic(x, L, k, b, c):
    return L / (1 + np.exp(-k * (x - b))) + c


def fit_exponential(xdata, ydata):
    try:
        # Fit an exponential to the peaks
        popt, pcov = curve_fit(exponential, xdata, ydata)
        print(f"Exponential fit. a:{popt[0]}, b:{popt[1]}")
        pstd = np.sqrt(np.diag(pcov))
        print(f"Standard deviation. a:{pstd[0]}, b:{pstd[1]}")
        return popt, pcov
    except RuntimeError:
        print("Failed to fit exponential")
        return None


def fit_order(df, column_to_fit):

    fit = linregress(np.log10(df["base_cell_width"]), np.log10(df[column_to_fit]))
    df_cutoff = df[df["base_cell_width"] <= 0.1]
    fit_cutoff = linregress(
        np.log10(df_cutoff["base_cell_width"]), np.log10(df_cutoff[column_to_fit])
    )

    print(
        f"Full data order: {fit.slope: 6.3f}±{fit.stderr: 6.3f}, 10^-1 cutoff: {fit_cutoff.slope: 6.3f}±{fit_cutoff.stderr: 6.3f}"
    )
    return fit, fit_cutoff


def fit_all_solvers(df):
    for column in ["peak_diff_2", "peak_diff_10", "peak_diff_100"]:
        print("Fitting", column)
        for solver in df["solver_method"].unique():
            print(" - Fitting", solver)
            fit_order(df[df["solver_method"] == solver], column)


def fit_all_solvers_interp(df):
    for interp in df["interp_kind"].unique():
        print("Interp", interp)
        df_i = df[df["interp_kind"] == interp]
        for solver in df_i["solver_method"].unique():
            print(" - Fitting", solver)
            fit_order(df_i[df_i["solver_method"] == solver], "peak_diff_2")


def save_to_store(store_path, dataframe):
    if store_path.is_file():

        try:
            with pd.HDFStore(store_path) as store:
                store.append("table", dataframe)
        except ValueError:
            with pd.HDFStore(store_path) as store:
                existing_df = store.select("table")
            combined_df = dataframe.append(
                existing_df, verify_integrity=True, ignore_index=True
            )
            combined_df.to_hdf(store_path, "table", format="table", mode="w")

    else:
        # Create the directories/folders if they don't already exist
        store_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_hdf(store_path, "table", format="table")


def process_runs(args):
    xr_list = list()
    diff_xr_list = list()
    df = pd.DataFrame()
    save_files = list(args.path.rglob("*.h5"))
    # save_files = [
    #     "out/sweep_dx_overlap_interp/single_overlap_edges/2021-08-23T20-14-24.h5"
    # ]
    if len(save_files) == 0:
        print(f"There are no h5 files in the path {args.path}")
        raise SystemExit

    starting_condition = None
    big_count = 0
    not_solved_count = 0
    # save_files = save_files[0:5]
    for file_path in tqdm(save_files, unit="files"):

        # if file_path.stat().st_size < 8e8:
        #     big_count += 1
        #     continue

        data = hdf5storage.read(filename=file_path)

        if not data["chimaera_grid"]["solved_attempted"]:
            not_solved_count += 1
            continue

        if starting_condition is None:
            starting_condition = data["chimaera_grid"]["starting_condition"]
        elif starting_condition != data["chimaera_grid"]["starting_condition"]:
            # raise Warning(
            #     "Run {file_path} has a different starting condition, skipping"
            # )
            print(f"Run {file_path} has a different starting condition, skipping")
            continue

        # print(data)

        solved_grid = data["chimaera_grid"]["cont_solution"]
        # solved_grid = data["chimaera_grid"]["cont_solution_comp"]
        try:
            solved_grid_positions = data["chimaera_grid"]["cont_solution_pos"]
        except KeyError:
            solved_grid_positions = data["chimaera_grid"]["chont_solution_pos"]

        try:
            time = data["chimaera_grid"]["result"]["t"]
        except KeyError:
            try:
                time = data["chimaera_grid"]["time"]
            except KeyError:
                print(f"delete {file_path}, it has no time")
                continue

        solved_xr = xr.DataArray(
            solved_grid,
            coords=[solved_grid_positions, time],
            dims=["pos", "time"],
        )

        # solved_xr = xr.DataArray(
        #     solved_grid,
        #     coords=[solved_grid_positions, [0.01, 0.05, 0.1, 0.5, 1, 10, 100]],
        #     dims=["pos", "time"],
        # )

        # for other_xr in xr_list:
        #     # This assumes that the new grid has a the smaller cell width
        #     diff_xr_list.append(solved_xr - other_xr.interp_like(solved_xr))

        # data["index"] = len(xr_list)

        # xr_list.append(solved_xr)
        run_dict = clean_save_for_pandas(data)
        diff = solved_xr - mbtb.gaussian(
            solved_xr["pos"],
            starting_condition["height"],
            starting_condition["centre"],
            starting_condition["width"],
            starting_condition["base"],
            time=solved_xr["time"],
        )
        run_dict["peak_diff_2"] = np.float64(
            np.abs(diff.sel(time=2, method="nearest")).max()
        )
        run_dict["peak_diff_10"] = np.float64(
            np.abs(diff.sel(time=10, method="nearest")).max()
        )
        run_dict["peak_diff_100"] = np.float64(
            np.abs(diff.sel(time=100, method="nearest")).max()
        )
        run_dict["mean_diff"] = np.float64(diff.mean())
        df = df.append(run_dict, ignore_index=True)

    # # After updating pandas from 1.2 to 1.3, the columns must be created before
    # # trying to assign values to them when using xarray functions
    # df["peak_diff"] = df["mean_diff"] = np.nan

    # if True:
    #     comparison_xr = xr_list[
    #         df[df["chimaera_name"] == "no_overlap_periodic"]["base_cell_width"].idxmin()
    #     ]
    #     # comparison_xr = xr_list[
    #     #     df[df["chimaera_name"] == "single_overlap"]["base_cell_width"].idxmin()
    #     # ]
    # else:
    #     comparison_xr = xr_list[df["base_cell_width"].idxmin()]

    # comparison_xr = mbtb.gaussian()

    # for index, grid in df.iterrows():
    #     # diff = xr_list[index].interp_like(comparison_xr) - comparison_xr
    #     diff = xr_list[index] - mbtb.gaussian(
    #         xr_list[index]["pos"],
    #         starting_condition["height"],
    #         starting_condition["centre"],
    #         starting_condition["width"],
    #         starting_condition["base"],
    #         time=xr_list[index]["time"],
    #     )
    #     df.at[index, "peak_diff_2"] = np.abs(diff.sel(time=2, method="nearest")).max()
    #     df.at[index, "peak_diff_10"] = np.abs(diff.sel(time=10, method="nearest")).max()
    #     df.at[index, "peak_diff_100"] = np.abs(diff.sel(time=100, method="nearest")).max()
    #     df.at[index, "mean_diff"] = diff.mean()
    #     # diff_xr_list.append(diff.max(dim="pos"))

    # diff.sel(time=0.001, method="nearest").plot(label=grid["base_cell_width"])
    # plt.legend()
    # plt.show()
    df = df.sort_values(by=["base_cell_width"], ascending=True)
    df["chimaera_name"] = df["chimaera_name"].astype("category")
    df["solver_method"] = df["solver_method"].astype("category")
    if "interp_kind" in df:
        df["interp_kind"] = df["interp_kind"].astype("category")

    print(f"Files too large:  {big_count}, Files not solved: {not_solved_count}")

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Multi-Block Test Bench - Analysis script",
        description="Analyse parameter sweeps from Multi-Block Test Bench",
    )
    parser.add_argument(
        "path",
        help="Path to a dictionary to look for MBTB save files",
        type=Path,
        nargs="?",
    )
    parser.add_argument(
        "-store_path",
        help="Path to a HDF5 pandas storage file",
        type=Path,
        default=Path("storage/storage.h5"),
    )
    parser.add_argument(
        "-s",
        help="Save to storage",
        action="store_true",
        dest="save_storage",
    )

    args = parser.parse_args()

    if args.path is None:
        store = pd.HDFStore(args.store_path)
        df = store.select("table")
        df = df.sort_values(by=["base_cell_width"], ascending=True)
        df["chimaera_name"] = df["chimaera_name"].astype("category")
        df["solver_method"] = df["solver_method"].astype("category")
        df["interp_kind"] = df["interp_kind"].astype("category")
        df = df.drop_duplicates()
    else:
        df = process_runs(args)

        if args.save_storage:
            print(f"Saving to storage: {args.store_path}")
            save_to_store(args.store_path, df)

    peak_axes = df.plot.scatter(
        x="base_cell_width",
        y="peak_diff_2",
        c="chimaera_name",
        cmap="viridis",
        loglog=True,
    )
    peak_axes.set_xlabel("Cell width of base grid /m")
    peak_axes.set_ylabel("Max difference /K")
    colourbar_axes = plt.gcf().get_axes()[1]
    colourbar_axes.set_ylabel("Chimaera grid")

    peak_axes = df.plot.scatter(
        x="base_cell_width",
        y="peak_diff_2",
        c="solver_method",
        cmap="tab10",
        loglog=True,
    )
    peak_axes.set_xlabel("Cell width of base grid /m")
    peak_axes.set_ylabel("Max difference /K")
    colourbar_axes = plt.gcf().get_axes()[1]
    colourbar_axes.set_ylabel("Solver method")

    # Order solver methods by explicit, implic and convergance order
    df["solver_method"] = df["solver_method"].cat.reorder_categories(
        ["RK23", "RK45", "DOP853", "Radau", "BDF", "BDF_tol"], ordered=True
    )
    # reorder interpolation kinds by their order
    df["interp_kind"] = df["interp_kind"].cat.reorder_categories(
        ["zero", "linear", "quadratic", "cubic"], ordered=True
    )
    overlap_df = df[df["chimaera_name"] == "single_overlap_edges"]
    base_df = df[df["chimaera_name"] == "no_overlap_edges"]

    peak_axes = base_df.plot.scatter(
        x="base_cell_width",
        y="peak_diff_2",
        c="solver_method",
        cmap="tab10",
        s=8,
        loglog=True,
    )
    peak_axes.set_xlabel("Cell width of base grid /m")
    peak_axes.set_ylabel("Max difference /K")
    colourbar_axes = plt.gcf().get_axes()[1]
    colourbar_axes.set_ylabel("Solver method")

    peak_axes = overlap_df.plot.scatter(
        x="base_cell_width",
        y="peak_diff_2",
        c="solver_method",
        cmap="viridis",
        loglog=True,
    )
    peak_axes.set_xlabel("Cell width of base grid /m")
    peak_axes.set_ylabel("Max difference /K")
    colourbar_axes = plt.gcf().get_axes()[1]
    colourbar_axes.set_ylabel("Solver method")

    # peak_axes = overlap_df[overlap_df["interface_width"] == 0.03].plot.scatter(
    #     x="base_cell_width",
    #     y="peak_diff_2",
    #     c="solver_method",
    #     cmap="viridis",
    #     loglog=True,
    # )
    # peak_axes.set_xlabel("Cell width of base grid /m")
    # peak_axes.set_ylabel("Max difference /K")
    # colourbar_axes = plt.gcf().get_axes()[1]
    # colourbar_axes.set_ylabel("Solver method")

    plt.show()
    # If you breakpoint here, make sure to let the program finish when done to
    # correctly close the store file

    # diff = xr_list[9] - xr_list[0].interp_like(xr_list[9])
    # print(diff)
    # diff.plot()
    # diff.sel(time=0.001, method="nearest").plot()
    # plt.show()
