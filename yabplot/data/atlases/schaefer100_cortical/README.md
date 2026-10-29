# schaefer100_cortical

Schaefer-100 cortical parcellation (7-network order) on the fs_LR 32k surface
(Conte69), in yabplot's custom cortical atlas format.

| File | Content |
|---|---|
| `schaefer100_conte69.csv` | one parcel ID per surface vertex, left then right hemisphere (64,984 rows; 0 = medial wall) |
| `schaefer100_LUT.txt` | `id  name  R  G  B  0`, one line per parcel (1-100) |
| `LICENSE` | licence of the source parcellation |

## Source

Built with `tools/build_schaefer100_cortical.py` from
`Schaefer2018_100Parcels_7Networks_order.dlabel.nii` in the CBIG repository
(https://github.com/ThomasYeoLab/CBIG, commit 634f676,
`stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/`;
MD5 7bb3f447ca0c4e61ed0cc917341bbd51).

The parcellation is distributed under the MIT licence of the CBIG repository,
reproduced in `LICENSE`.

## Citation

Schaefer A, Kong R, Gordon EM, Laumann TO, Zuo XN, Holmes AJ, Eickhoff SB,
Yeo BTT. Local-Global parcellation of the human cerebral cortex from intrinsic
functional connectivity MRI. Cerebral Cortex 29:3095-3114, 2018.
