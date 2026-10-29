#!/usr/bin/env python
"""
Build yabplot subcortical atlas VTK meshes for the 4S atlas series.

All 4S atlases (4S156 through 4S1056) share the same 56 subcortical parcels
(CIT168 basal ganglia, HCP thalamus nuclei, hippocampus/amygdala, cerebellum)
with identical MNI coordinates. Only the integer parcel indices differ
(e.g. 101–156 for 4S156, 401–456 for 4S456). The resulting VTK meshes are
geometrically identical regardless of which CIFTI is used as source.

Therefore this script writes one shared directory ('4s_subcortical/') that
serves all 4S atlas variants.

The meshes are derived from the 4S atlas (CC BY-ND), so they are built locally
and not shipped with the package; yabplot/data/atlases/4s_subcortical/ is
gitignored.

Usage (from the repository root; needs yabplot's own dependencies plus nibabel
and pandas):

    # Any 4S version works; 4S156 is the default (smallest CIFTI, fastest)
    python tools/build_4s156_subcortical.py --atlas-root <dir holding atlas-4S*Parcels/>
    python tools/build_4s156_subcortical.py --atlas-root <dir> --parcellation atlas-4S456Parcels

    # Write into an installed yabplot instead of this source tree
    python tools/build_4s156_subcortical.py --atlas-root <dir> \
        --out-dir "$(python -c 'import os, yabplot; print(os.path.join(os.path.dirname(yabplot.__file__), "data", "atlases", "4s_subcortical"))')"

Each run replaces every .vtk file in the output directory with the meshes it
builds.
"""

import os
import sys
import re
import argparse
import glob
import shutil
import tempfile
import numpy as np
import nibabel as nib
import pandas as pd

# Add yabplot to path (dev install should handle this, but be safe)
YABPLOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, YABPLOT_DIR)
import yabplot as yab

# --- Config ---
# Default output dir (this source tree) — version-agnostic because the 56
# subcortical regions are anatomically identical across all 4S atlas variants.
DEFAULT_OUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "yabplot", "data", "atlases", "4s_subcortical"
))

# Known 4S atlas sizes for input validation
_VALID_SIZES = {156, 456, 556, 656, 756, 856, 956, 1056}
_N_SUBCORTICAL = 56  # always fixed in 4S series


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build shared 4S subcortical VTK meshes for yabplot."
    )
    parser.add_argument(
        "--atlas-root", required=True,
        help="Directory that holds the 4S atlas folders (e.g. atlas-4S156Parcels/).",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Directory to write the .vtk meshes to (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--parcellation",
        default="atlas-4S156Parcels",
        help=(
            "Source 4S atlas to extract subcortical volume from. "
            "Any version works since subcortical geometry is shared. "
            "Default: atlas-4S156Parcels (smallest CIFTI, recommended)."
        ),
    )
    parser.add_argument(
        "--smooth-i", type=int, default=15,
        help="Laplacian smoothing iterations (default: 15).",
    )
    parser.add_argument(
        "--smooth-f", type=float, default=0.6,
        help="Laplacian smoothing relaxation factor 0–1 (default: 0.6).",
    )
    return parser.parse_args()


def validate_parcellation(parcellation: str) -> int:
    """
    Validate parcellation string and return total parcel count.
    Accepts both short ('4S156') and full ('atlas-4S156Parcels') forms.
    """
    match = re.search(r"4[Ss](\d+)", parcellation)
    if not match:
        raise ValueError(
            f"Cannot parse 4S atlas size from {parcellation!r}. "
            f"Expected e.g. 'atlas-4S156Parcels' or '4S456'."
        )
    size = int(match.group(1))
    if size not in _VALID_SIZES:
        raise ValueError(
            f"Unknown 4S atlas size: {size}. "
            f"Valid sizes: {sorted(_VALID_SIZES)}"
        )
    return size


def get_atlas_paths(atlas_root: str, parcellation: str) -> tuple[str, str]:
    """Return (cifti_path, tsv_path) for the given parcellation."""
    # Normalize to full name
    if not parcellation.startswith("atlas-"):
        parcellation = f"atlas-{parcellation}"
    if not parcellation.endswith("Parcels"):
        parcellation = f"{parcellation}Parcels"

    atlas_dir = os.path.join(atlas_root, parcellation)
    cifti_path = os.path.join(
        atlas_dir, f"{parcellation}_space-fsLR_den-91k_dseg.dlabel.nii"
    )
    tsv_path = os.path.join(atlas_dir, f"{parcellation}_dseg.tsv")

    for path in (cifti_path, tsv_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Atlas file not found: {path}")
    return cifti_path, tsv_path


def extract_subcortical_volume(cifti_path: str, tmp_nii_path: str) -> None:
    """
    Extract the volumetric component from a CIFTI dlabel file and save as 3D NIfTI.

    The CIFTI 91k dlabel file stores both surface (64984 vertices) and
    volumetric grayordinates in a single 1D array. This function reconstructs
    the 3D volume for all subcortical structures.

    Uses brain_axis.volume_mask to select volumetric grayordinates directly.
    brain_axis.voxel is a (N, 3) array with NaN for surface elements;
    volume_mask = ~np.isnan(voxel).any(axis=-1) selects only real voxels.

    Args:
        cifti_path: Path to .dlabel.nii CIFTI file.
        tmp_nii_path: Output path for the 3D NIfTI volume.
    """
    print(f"  Loading CIFTI: {cifti_path}")
    img = nib.load(cifti_path)

    # For dlabel CIFTI: axis 0 = LabelAxis, axis 1 = BrainModelAxis
    brain_axis = img.header.get_axis(1)

    # Label data: shape (1, n_grayordinates) → squeeze to (n_grayordinates,)
    data = img.get_fdata(dtype=np.float32).squeeze()
    print(f"  Total grayordinates: {len(data)}")

    # volume_mask selects grayordinates that are voxels (not surface vertices).
    # brain_axis.voxel is (N, 3) with NaN for surface elements.
    volume_mask = brain_axis.volume_mask  # (N,) bool
    n_vol = int(volume_mask.sum())
    print(f"  Volumetric grayordinates: {n_vol}")

    if n_vol == 0:
        raise ValueError(
            "No volumetric grayordinates found. "
            "Check that the CIFTI file is a full 91k surface+volume file."
        )

    vol_voxels = brain_axis.voxel[volume_mask].astype(int)  # (n_vol, 3)
    vol_vals = np.round(data[volume_mask]).astype(np.int32)

    vol = np.zeros(brain_axis.volume_shape, dtype=np.int32)
    vol[vol_voxels[:, 0], vol_voxels[:, 1], vol_voxels[:, 2]] = vol_vals

    unique_vals = np.unique(vol[vol > 0])
    print(f"  Volume shape: {brain_axis.volume_shape}")
    print(f"  Parcel indices in volume: {unique_vals.tolist()}")
    nib.save(nib.Nifti1Image(vol, brain_axis.affine), tmp_nii_path)


def build_labels_dict(tsv_path: str, total_parcels: int) -> dict:
    """
    Build labels_dict for yab.build_subcortical_atlas from the TSV.

    Subcortical parcels are always the last 56 indices in any 4S atlas.
    Threshold = total_parcels - 56 (e.g. 100 for 4S156, 400 for 4S456).

    Returns {parcel_index: label_name} for subcortical parcels only.
    """
    threshold = total_parcels - _N_SUBCORTICAL
    tsv = pd.read_csv(tsv_path, sep="\t")
    subcort = tsv[tsv["index"] > threshold]
    labels_dict = {int(row["index"]): str(row["label"]) for _, row in subcort.iterrows()}
    print(f"  Subcortical parcels (index > {threshold}): {len(labels_dict)}")
    for idx, label in sorted(labels_dict.items()):
        print(f"    {idx}: {label}")
    return labels_dict


def main():
    args = parse_args()

    print("=" * 65)
    print("Building shared 4S subcortical atlas VTK meshes for yabplot")
    print("=" * 65)
    print(f"Source parcellation: {args.parcellation}")
    out_dir = os.path.abspath(args.out_dir)
    print(f"Output dir (shared): {out_dir}")
    print(f"Smoothing: {args.smooth_i} iterations, factor {args.smooth_f}")
    print()

    # Validate and resolve paths
    total_parcels = validate_parcellation(args.parcellation)
    print(f"Atlas size: {total_parcels} parcels total, {_N_SUBCORTICAL} subcortical")
    cifti_path, tsv_path = get_atlas_paths(args.atlas_root, args.parcellation)

    existing_vtk = glob.glob(os.path.join(out_dir, "*.vtk"))
    if existing_vtk:
        print(
            f"\nNOTE: {len(existing_vtk)} existing VTK file(s) in {out_dir} will be "
            "replaced by this build.\n"
        )

    # Build into a fresh directory next to out_dir, so a failed run leaves the
    # existing meshes alone and a successful one leaves no stale files behind.
    parent = os.path.dirname(out_dir)
    os.makedirs(parent, exist_ok=True)
    build_dir = tempfile.mkdtemp(prefix=".4s_subcortical_build_", dir=parent)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_nii_path = tmp.name

    try:
        print("Step 1: Extract subcortical volumetric component from CIFTI...")
        extract_subcortical_volume(cifti_path, tmp_nii_path)

        print("\nStep 2: Build labels dict from TSV...")
        labels_dict = build_labels_dict(tsv_path, total_parcels)

        print(f"\nStep 3: Build VTK meshes → {build_dir}")
        yab.build_subcortical_atlas(
            nii_path=tmp_nii_path,
            labels_dict=labels_dict,
            out_dir=build_dir,
            smooth_i=args.smooth_i,
            smooth_f=args.smooth_f,
        )

        built = sorted(glob.glob(os.path.join(build_dir, "*.vtk")))
        if not built:
            raise RuntimeError(f"No VTK meshes were built in {build_dir}.")
        os.makedirs(out_dir, exist_ok=True)
        for f in existing_vtk:
            os.remove(f)
        for f in built:
            shutil.move(f, os.path.join(out_dir, os.path.basename(f)))

    finally:
        if os.path.exists(tmp_nii_path):
            os.remove(tmp_nii_path)
        shutil.rmtree(build_dir, ignore_errors=True)

    print(f"\nGenerated {len(built)} VTK files in {out_dir}:")
    for f in built:
        print(f"  {os.path.basename(f)}")

    print(
        "\nNote: These VTK files are valid for ALL 4S atlas variants (4S156–4S1056).\n"
        "The 56 subcortical parcels share identical MNI coordinates across atlas sizes.\n"
    )


if __name__ == "__main__":
    main()
