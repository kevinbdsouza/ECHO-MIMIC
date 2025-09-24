import os
import subprocess
import sys
from datetime import datetime
import logging
from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import geopandas as gpd
import matplotlib.patches as mpatches
import re
import numpy as np
from shapely.geometry import Polygon
import json


def delete_outputs(plots_dir):
    files = os.listdir(plots_dir)
    for file in files:
        if "output" in file or "scores" in file:
            file_path = os.path.join(plots_dir, file)
            os.remove(file_path)
        elif "plot" in file:
            dir_path = os.path.join(plots_dir, file)
            dir_files = os.listdir(dir_path)
            for dir_file in dir_files:
                if "output" in dir_file or "scores" in dir_file:
                    dir_file_path = os.path.join(dir_path, dir_file)
                    os.remove(dir_file_path)


class CommandOutputCapture:
    def __init__(self, log_dir="logs"):
        """Initialize the capture utility with a logging directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger("command_capture")
        self.logger.setLevel(logging.DEBUG)

        # Create a unique log file for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"command_run_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)

    def run_command(self, command, capture_stderr=True):
        """
        Run a command and capture its output.

        Args:
            command (str): The command to run
            capture_stderr (bool): Whether to capture stderr output

        Returns:
            tuple: (return_code, stdout, stderr)
        """
        self.logger.info(f"Running command: {command}")

        try:
            # Method 1: Using subprocess.Popen
            with subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE if capture_stderr else None,
                    universal_newlines=True
            ) as process:
                stdout, stderr = process.communicate()
                return_code = process.returncode

                if return_code:
                    # Log the outputs
                    self.logger.info(f"Command returned with code: {return_code}")
                    if stdout:
                        self.logger.debug(f"STDOUT:\n{stdout}")
                    if stderr:
                        self.logger.debug(f"STDERR:\n{stderr}")

                return return_code, stdout, stderr

        except Exception as e:
            self.logger.error(f"Error running command: {str(e)}")
            raise

    def run_python_script(self, script_path, args=None):
        """
        Run a Python script and capture its output.

        Args:
            script_path (str): Path to the Python script
            args (list): Optional list of arguments for the script

        Returns:
            tuple: (return_code, stdout, stderr)
        """
        if args is None:
            args = []

        command = f"{sys.executable} {script_path} {' '.join(args)}"
        return self.run_command(command)


def random_1():
    # Example usage
    capture = CommandOutputCapture()

    # Example 2: Run a Python script
    code, out, err = capture.run_python_script(
        "script.py",
        ["--arg1", "value1"]
    )


def line_plot(run=1):
    run_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", "farm_" + str(farm_id), "heuristics",
                           "run_" + str(run))
    score_file = os.path.join(run_dir, "fitness.txt")

    generations = []
    best_scores = []

    # Regex to capture the pattern: Generation - X : Best Score - Y
    pattern = re.compile(r'Generation\s*-\s*(\d+)\s*:\s*Best Score\s*-\s*([-\d\.]+)')

    with open(score_file, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                gen = int(match.group(1))
                score = float(match.group(2))
                generations.append(gen)
                best_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_scores, marker='o', linestyle='-', color='b')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score")
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "fitness_gen.png"))


def corr_plot(cfg, scatterplot=False, correlation=False, rf_regression=False, lin_regression=False, mutual_info=False,
              run=1):
    run_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", "farm_" + str(farm_id), "heuristics",
                           "run_" + str(run))
    figures_dir = os.path.join(run_dir, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    csv_file_path = os.path.join(run_dir, "metrics.csv")
    corr_file = os.path.join(run_dir, "column_correlations.txt")
    df = pd.read_csv(csv_file_path)
    df = df.fillna(0)

    columns_to_compare = [
        'loc', 'lloc', 'sloc', 'comment', 'multi', 'blank',
        'avg_cyclomatic_complexity', 'maintainability_index',
        'halstead_h1', 'halstead_h2', 'halstead_N1', 'halstead_N2',
        'halstead_vocabulary', 'halstead_length', 'halstead_volume',
        'halstead_difficulty', 'halstead_effort', 'halstead_time',
        'halstead_bugs'
    ]

    if scatterplot:
        # Number of columns you want per row in the grid
        ncols = 4
        # Calculate the number of rows needed
        nrows = int(np.ceil(len(columns_to_compare) / ncols))

        # Create a big figure with subplots
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=(4 * ncols, 4 * nrows),
                                 squeeze=False)

        # Flatten the axes array for easy indexing
        axes = axes.flatten()

        for idx, col in enumerate(columns_to_compare):
            # Current axis
            ax = axes[idx]

            # Create scatter plot on the current axis
            ax.scatter(df[col], df['fitness_score'], alpha=0.6)
            ax.set_title(f"{col} vs. fitness_score", fontsize=10)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel("fitness_score", fontsize=8)

        # If there are any unused subplots, hide them
        for i in range(len(columns_to_compare), nrows * ncols):
            axes[i].axis('off')

        plt.tight_layout()
        # Save the single big figure
        plot_filename = os.path.join(figures_dir, "scatter_all_in_one.png")
        plt.savefig(plot_filename)
        plt.close(fig)

    if correlation:
        correlation_results = []
        for col in columns_to_compare:
            corr_value = df['fitness_score'].corr(df[col])
            correlation_results.append(f"{col}: {corr_value:.4f}")

        with open(corr_file, "w") as f:
            f.write("Correlation of each column with fitness_score:\n\n")
            for result in correlation_results:
                f.write(result + "\n")

    X = df[columns_to_compare]
    y = df['fitness_score']
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    if rf_regression:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        importances = rf.feature_importances_
        feature_importance = list(zip(columns_to_compare, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for feat, imp in feature_importance:
            print(f"{feat}: {imp:.4f}")

    if lin_regression:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        lasso = Lasso(alpha=0.1)  # regularization strength
        lasso.fit(X_train, y_train)

        coefs = list(zip(columns_to_compare, lasso.coef_))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, coef in coefs:
            print(f"{feat}: {coef:.4f}")

    if mutual_info:
        mi = mutual_info_regression(X, y)
        mi = list(zip(columns_to_compare, mi))
        mi.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, score in mi:
            print(f"{col}: {score:.4f}")


def output_evolve_gif(start, end, run=1):
    farm_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", "farm_" + str(farm_id))
    run_dir = os.path.join(farm_dir, "heuristics", "run_" + str(run))
    output_gt_path = os.path.join(farm_dir, "output_gt.geojson")
    input_json = os.path.join(farm_dir, "input.geojson")
    run_input_json = os.path.join(run_dir, "input.geojson")
    gif_filename = os.path.join(run_dir, "comparison.gif")

    gdf_gt = gpd.read_file(output_gt_path)
    gdf_input = gpd.read_file(input_json)
    gdf_input_merged = gdf_input.merge(
        gdf_gt[['id', 'margin_intervention', 'habitat_conversion']],
        on='id',
        how='left'
    )
    gdf_input_merged = gdf_input_merged.fillna(0)

    frames = []
    os.chdir(run_dir)
    for x in range(start, end + 1):
        shutil.copyfile(input_json, run_input_json)
        _, _, _ = capture.run_python_script("best_heuristics_gem_gen_" + str(x) + ".py")

        output_path = os.path.join(run_dir, "output.geojson")
        if not os.path.exists(output_path):
            print(f"No output for heuristic {x}")
            continue

        gdf_out = gpd.read_file(output_path)
        gdf_out_merged = gdf_input.merge(
            gdf_out[['id', 'margin_intervention', 'habitat_conversion']],
            on='id',
            how='left'
        )
        gdf_out_merged = gdf_out_merged.fillna(0)

        # Create a side-by-side plot: ground truth (left) and output (right)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        patches = [
            mpatches.Patch(color='red', label='Margin Interventions'),
            mpatches.Patch(color='green', label='Habitat Conversions'),
            mpatches.Patch(color='blue', label='Existing Habitats')]

        # Left panel: Ground Truth
        gdf_input_merged.boundary.plot(ax=axes[0], color='grey', aspect=1)
        gdf_input_merged.plot(ax=axes[0], color='red', alpha=gdf_input_merged['margin_intervention'], aspect=1)
        gdf_input_merged.plot(ax=axes[0], color='green', alpha=gdf_input_merged['habitat_conversion'], aspect=1)
        hab_plots_gdf = gdf_input_merged[gdf_input_merged['type'] == 'hab_plots']
        hab_plots_gdf.plot(ax=axes[0], color='blue', alpha=0.5, aspect=1)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        # Right panel: Current output
        gdf_out_merged.boundary.plot(ax=axes[1], color='grey', aspect=1)
        gdf_out_merged.plot(ax=axes[1], color='red', alpha=gdf_out_merged['margin_intervention'], aspect=1)
        gdf_out_merged.plot(ax=axes[1], color='green', alpha=gdf_out_merged['habitat_conversion'], aspect=1)
        hab_plots_gdf = gdf_out_merged[gdf_out_merged['type'] == 'hab_plots']
        hab_plots_gdf.plot(ax=axes[1], color='blue', alpha=0.5, aspect=1)
        axes[1].set_title(f"Heuristic Output {x}")
        axes[1].axis("off")

        leg = axes[0].get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((1.1, 1.1))
        axes[0].legend(handles=patches)

        # Save this figure as a frame
        frame_filename = os.path.join(run_dir, "figures", f"frame_{x}.png")
        plt.savefig(frame_filename, bbox_inches='tight', dpi=150)
        plt.close(fig)

        frames.append(frame_filename)
        os.remove(output_path)

    if frames:
        images = [Image.open(f) for f in frames]
        images[0].save(
            gif_filename,
            save_all=True,
            append_images=images[1:],
            duration=1000,  # Duration in ms per frame
            loop=0,
            disposal=2
        )


def plot_directions():
    def plot_polygon(ax, poly, fill=False, color=None):
        """
        Plots a shapely Polygon or MultiPolygon.
         - If fill=True, fills the polygon with the given color.
         - Otherwise, draws the polygon boundary with the given color.
        """
        if poly.is_empty:
            return

        # If it’s a simple Polygon
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            if fill:
                ax.fill(x, y, color=color)
            else:
                ax.plot(x, y, color=color)
            # Plot any interior holes (rings)
            for ring in poly.interiors:
                rx, ry = ring.xy
                ax.plot(rx, ry, color=color)

        # If it’s a MultiPolygon
        elif poly.geom_type == 'MultiPolygon':
            for single_poly in poly.geoms:
                plot_polygon(ax, single_poly, fill=fill, color=color)

    def get_quadrant_polygon(plot_poly: Polygon, direction: str) -> Polygon:
        """
        Returns the bounding box quadrant polygon (north-west, north-east,
        south-west, south-east) intersected with the original plot polygon.
        """
        minx, miny, maxx, maxy = plot_poly.bounds
        x_center = (minx + maxx) / 2.0
        y_center = (miny + maxy) / 2.0

        if direction == "north-west":
            box = Polygon([
                (minx, y_center), (x_center, y_center),
                (x_center, maxy), (minx, maxy)
            ])
        elif direction == "north-east":
            box = Polygon([
                (x_center, y_center), (maxx, y_center),
                (maxx, maxy), (x_center, maxy)
            ])
        elif direction == "south-west":
            box = Polygon([
                (minx, miny), (x_center, miny),
                (x_center, y_center), (minx, y_center)
            ])
        elif direction == "south-east":
            box = Polygon([
                (x_center, miny), (maxx, miny),
                (maxx, y_center), (x_center, y_center)
            ])
        else:
            # If it's not one of the four quadrants,
            # return an empty polygon
            return Polygon()

        return box.intersection(plot_poly)

    def plot_line(ax, line, color=None):
        """
        Plots a shapely LineString or MultiLineString in the given color.
        """
        if line.is_empty:
            return

        if line.geom_type == 'LineString':
            x, y = line.xy
            ax.plot(x, y, color=color)
        elif line.geom_type == 'MultiLineString':
            for ls in line.geoms:
                x, y = ls.xy
                ax.plot(x, y, color=color)

    conn_dir = os.path.join(cfg.data_dir, "crop_inventory", "syn_farms", "farm_" + str(farm_id), "connectivity")
    heur_src_path = os.path.join(conn_dir, "generations", "best_heuristics_gem_gen_0.py")
    heur_dst_path = os.path.join(conn_dir, "best_heuristics_gem.py")
    shutil.copyfile(heur_src_path, heur_dst_path)

    input_path = os.path.join(conn_dir, "heuristics", "input.geojson")
    input_cp_path = os.path.join(conn_dir, "heuristics", "input_cp.geojson")
    shutil.copyfile(input_path, input_cp_path)

    os.chdir(conn_dir)
    _, _, _ = capture.run_python_script(heur_dst_path)

    output_path = os.path.join(conn_dir, "output.json")
    if not os.path.exists(output_path):
        print(f"No output for heuristic")
    else:
        gdf_input = gpd.read_file(input_cp_path)
        gdf_input['id'] = gdf_input['id'].astype(str)

        with open(output_path, 'r') as f:
            out_data = json.load(f)
        df_out = pd.DataFrame(out_data)
        if 'plot_id' in df_out.columns and 'id' not in df_out.columns:
            df_out.rename(columns={'plot_id': 'id'}, inplace=True)
        df_out['id'] = df_out['id'].astype(str)

        gdf_merged = gdf_input.merge(
            df_out[['id', 'margin_directions', 'habitat_directions']],
            on='id',
            how='left'
        )
        gdf_merged['margin_directions'] = gdf_merged['margin_directions'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        gdf_merged['habitat_directions'] = gdf_merged['habitat_directions'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        gdf_merged = gdf_merged.fillna(0)
        fig, ax = plt.subplots()

        for idx, row in gdf_merged.iterrows():
            plot_poly = row["geometry"]
            plot_polygon(ax, plot_poly, fill=False, color="grey")

            # 2) Margin directions are edges in those quadrants
            margin_directions = row["margin_directions"]
            for direction in margin_directions:
                # Get the quadrant polygon
                qpoly = get_quadrant_polygon(plot_poly, direction)
                # Intersect the plot boundary with that quadrant region
                boundary = plot_poly.boundary.intersection(qpoly)
                plot_line(ax, boundary, color='red')  # draws line with default settings

            # 3) Habitat directions are sub‐polygon areas in those quadrants
            habitat_directions = row["habitat_directions"]
            for direction in habitat_directions:
                qpoly = get_quadrant_polygon(plot_poly, direction)
                plot_polygon(ax, qpoly, fill=True, color='green')  # fill the sub‐quadrant area

        ax.set_aspect('equal', 'box')
        # Save the figure
        plt.savefig(os.path.join(conn_dir, "output_pred_directions.png"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    cfg = Config()
    capture = CommandOutputCapture()
    farm_id = 6

    # random_1()
    # line_plot(run=1)
    # corr_plot(cfg, scatterplot=True, correlation=True, rf_regression=True, lin_regression=True, mutual_info=True, run=1)
    # output_evolve_gif(start=0, end=41, run=1)

    plot_directions()
