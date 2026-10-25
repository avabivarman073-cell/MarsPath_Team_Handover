
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol

BASE = Path(r"D:\videos\Orbit Guardians\MarsPath")

SOURCE = (
    BASE / "03_processed_Data" / "Navigation"
    / "landing_area_navigation_cost_2km.tif"
)
OUTPUT = (
    BASE / "05_Backend_Exports" / "navigation_grid.json"
)

# Approximate projected landing coordinate
LANDING_X = 4590872.743
LANDING_Y = 1093304.013

BLOCK_SIZE = 5
BLOCKED = 255

with rasterio.open(SOURCE) as src:
    raster = src.read(1, masked=True)

    height, width = raster.shape
    rows = math.ceil(height / BLOCK_SIZE)
    cols = math.ceil(width / BLOCK_SIZE)

    grid = np.full((rows, cols), BLOCKED, dtype=np.uint8)

    # Accept only our defined cost classes
    values = raster.filled(np.nan)
    valid = (
        ~np.ma.getmaskarray(raster)
        & np.isin(values, [1, 5, 255])
    )

    for r in range(rows):
        for c in range(cols):
            r0 = r * BLOCK_SIZE
            c0 = c * BLOCK_SIZE
            r1 = min(r0 + BLOCK_SIZE, height)
            c1 = min(c0 + BLOCK_SIZE, width)

            block = values[r0:r1, c0:c1]
            block_valid = valid[r0:r1, c0:c1]

            # Block incomplete edge cells and unknown data
            if block.shape != (5, 5) or not block_valid.all():
                continue

            if np.any(block == 255):
                continue

            if np.any(block == 5):
                grid[r, c] = 5
            else:
                grid[r, c] = 1

    landing_row_1m, landing_col_1m = rowcol(
        src.transform, LANDING_X, LANDING_Y
    )

    landing_row = int(landing_row_1m // BLOCK_SIZE)
    landing_col = int(landing_col_1m // BLOCK_SIZE)

    if not (0 <= landing_row < rows and
            0 <= landing_col < cols):
        raise ValueError("Landing location is outside grid")

    landing_cost = int(grid[landing_row, landing_col])

    data = {
        "width": cols,
        "height": rows,
        "cell_size_m": 5,
        "crs_wkt": src.crs.to_wkt(),
        "source_transform": list(src.transform)[:6],
        "landing": {
            "x": float(LANDING_X),
            "y": float(LANDING_Y),
            "row": int(landing_row),
            "col": int(landing_col),
            "cost": int(landing_cost)
        },
        "blocked_value": BLOCKED,
        "costs": grid.tolist()
    }

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(data, f, separators=(",", ":"))

unique, counts = np.unique(grid, return_counts=True)

print("Navigation grid created:", OUTPUT)
print("Grid dimensions:", cols, "x", rows)
print("Landing cell:", landing_row, landing_col)
print("Landing cost:", landing_cost)
print("Cost counts:", dict(zip(
    unique.tolist(), counts.tolist()
)))
