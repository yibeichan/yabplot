import hashlib
import os

import pytest

import yabplot as yab

ATLAS_DIR = os.path.join(
    os.path.dirname(yab.__file__), "data", "atlases", "schaefer100_cortical"
)

# sha256 of the shipped files; regenerate with tools/build_schaefer100_cortical.py
EXPECTED = {
    "LICENSE": "0e38f0e8b304f13de2ce2911320faaa08ced178e031d3bbebc3e4769762e5f50",
    "schaefer100_conte69.csv": "72750adfea605adeab2604b489bb87e6588efb805f6c041981a1ba506f830706",
    "schaefer100_LUT.txt": "7f4c6a07880c125c471f91d10db1ba619a75352c2e27c215972bdef06f03e6a1",
}


def test_schaefer100_files_match():
    """The installed package carries the cortical atlas, unchanged."""
    assert set(EXPECTED) <= set(os.listdir(ATLAS_DIR))
    for name, digest in EXPECTED.items():
        with open(os.path.join(ATLAS_DIR, name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, name


def test_schaefer100_regions():
    """yabplot reads all 100 parcels from the bundled LUT."""
    regions = yab.get_atlas_regions(
        atlas="schaefer100_cortical",
        category="cortical",
        custom_atlas_path=ATLAS_DIR,
    )
    assert len(regions) == 100
    assert regions[0] == "7Networks_LH_Vis_1"


@pytest.mark.skipif(
    os.path.exists(os.path.join(os.path.dirname(yab.__file__), "..", "pyproject.toml")),
    reason="yabplot is imported from a source tree, where local builds may exist",
)
def test_no_subcortical_meshes_shipped():
    """4S-derived meshes are CC BY-ND and must not be packaged."""
    atlases = os.path.join(os.path.dirname(yab.__file__), "data", "atlases")
    assert sorted(os.listdir(atlases)) == ["schaefer100_cortical"]
