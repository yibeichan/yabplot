#!/usr/bin/env python
"""
Build the schaefer100_cortical atlas for yabplot from a Schaefer-100
(7-network order) fs_LR 32k CIFTI dlabel file.

The shipped atlas was built from the original release,
Schaefer2018_100Parcels_7Networks_order.dlabel.nii in
https://github.com/ThomasYeoLab/CBIG (MIT licence; see
yabplot/data/atlases/schaefer100_cortical/README.md). The cortical part of the
4S156 atlas (atlas-4S156Parcels_space-fsLR_den-91k_dseg.dlabel.nii) gives
identical files.

Outputs two files to yabplot/data/atlases/schaefer100_cortical/:
  schaefer100_conte69.csv  — one parcel ID per line, full fsLR32k surface
                             (0 = medial wall / background; 1–100 = Schaefer parcels)
  schaefer100_LUT.txt      — id TAB name TAB R TAB G TAB B TAB 0

The atlas ships with the package; rerun this only to regenerate it.

Usage (from the repository root; needs nibabel and numpy):
  python tools/build_schaefer100_cortical.py \
      --dlabel <path>/Schaefer2018_100Parcels_7Networks_order.dlabel.nii
"""

import argparse
import os
import numpy as np
import nibabel as nib

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
N_CORTICAL = 100  # Schaefer-100 cortical parcels, IDs 1–100

OUT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "yabplot", "data", "atlases", "schaefer100_cortical",
    )
)
OUT_CSV = os.path.join(OUT_DIR, "schaefer100_conte69.csv")
OUT_LUT = os.path.join(OUT_DIR, "schaefer100_LUT.txt")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(dlabel_path: str) -> None:
    print(f"Loading CIFTI dlabel:\n  {dlabel_path}")
    if not os.path.exists(dlabel_path):
        raise FileNotFoundError(f"CIFTI dlabel not found: {dlabel_path}")

    img = nib.load(dlabel_path)
    label_axis = img.header.get_axis(0)   # LabelAxis  (rows = label maps)
    brain_axis = img.header.get_axis(1)   # BrainModelAxis

    # nvertices: dict of structure_name → total vertices on that surface
    nvertices = brain_axis.nvertices  # e.g. {'CIFTI_STRUCTURE_CORTEX_LEFT': 32492, ...}

    # data shape: (1, n_grayordinates) → squeeze to (n_grayordinates,)
    data = np.asarray(img.dataobj).squeeze()

    full_lh = full_rh = None

    for struct_name, slc, bm in brain_axis.iter_structures():
        if struct_name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            n_verts_lh = nvertices[struct_name]
            full_lh = np.zeros(n_verts_lh, dtype=int)
            full_lh[bm.vertex] = data[slc].astype(int)
            print(
                f"LH cortex: {len(bm.vertex):,} grayordinates "
                f"→ {n_verts_lh:,} vertices total"
            )
        elif struct_name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            n_verts_rh = nvertices[struct_name]
            full_rh = np.zeros(n_verts_rh, dtype=int)
            full_rh[bm.vertex] = data[slc].astype(int)
            print(
                f"RH cortex: {len(bm.vertex):,} grayordinates "
                f"→ {n_verts_rh:,} vertices total"
            )

    if full_lh is None or full_rh is None:
        raise RuntimeError(
            "Could not find CORTEX_LEFT or CORTEX_RIGHT in CIFTI brain model axes."
        )

    full_surface = np.concatenate([full_lh, full_rh])
    n_cortical_verts = int(np.sum(full_surface > 0))
    n_bg = int(np.sum(full_surface == 0))
    print(
        f"Full surface: {len(full_surface):,} vertices  "
        f"({n_cortical_verts:,} cortical, {n_bg:,} medial wall/background)"
    )

    # Sanity: no subcortical IDs (101–156) should appear in surface models
    unexpected = sorted(set(full_surface.tolist()) - set(range(0, N_CORTICAL + 1)))
    if unexpected:
        raise RuntimeError(
            f"Unexpected parcel IDs in cortical surface: {unexpected[:10]}. "
            "The CIFTI may have subcortical voxels assigned cortical IDs."
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Write conte69.csv ---
    np.savetxt(OUT_CSV, full_surface, fmt="%d")
    print(f"Wrote: {OUT_CSV}")

    # --- Write LUT from CIFTI label table ---
    # nibabel label_axis.label[0] is a dict: int → (name_str, rgba_tuple)
    label_table = label_axis.label[0]
    with open(OUT_LUT, "w") as f:
        for idx in range(1, N_CORTICAL + 1):
            if idx not in label_table:
                print(f"  WARNING: parcel ID {idx} missing from CIFTI label table")
                continue
            label_string, rgba = label_table[idx]
            r, g, b, a = (int(c * 255) for c in rgba)
            f.write(f"{idx}\t{label_string}\t{r}\t{g}\t{b}\t0\n")
    print(f"Wrote: {OUT_LUT}")

    print(f"\nDone. Atlas dir: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the schaefer100_cortical atlas from a Schaefer-100 CIFTI dlabel."
    )
    parser.add_argument(
        "--dlabel", required=True,
        help="Schaefer2018_100Parcels_7Networks_order.dlabel.nii (fs_LR 32k)",
    )
    build(parser.parse_args().dlabel)
