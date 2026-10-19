# Third party licensing scope

This inventory is not a legal clearance opinion or a replacement for complete license texts. It distinguishes shipped core components from external integrations. The root LICENSE.txt retains the custom VGGT terms; retained upstream copyrights must not be removed. Abu is not uniformly Apache licensed. Original VGGT authorship remains with the Oxford and Meta authors identified in the retained source headers.

## Bundled CPU core

The builder copies actual dependency license files into THIRD_PARTY_LICENSES and generates a CycloneDX inventory. Core source and collected notices accompany candidate assets. CPython, Tcl/Tk, NumPy, SciPy, Pillow, OpenCV, Trimesh, Viser, PyCOLMAP and their native dependencies have separate terms. Review the exact wheel build, embedded FFmpeg and native solver dependencies before a stable release; a top-level package license alone is insufficient. No model weights, ACE0 package, PFGS360 package, LichtFeld binary or NVIDIA GPU toolkit are included in the core.

## External integrations and models

| Component | Actual upstream terms and release treatment |
| --- | --- |
| Original VGGT code | [Custom VGGT license](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt), with acceptable-use terms. Preserve file-specific additional notices. |
| Original VGGT weights | [Original checkpoint](https://huggingface.co/facebook/VGGT-1B) and [commercial checkpoint](https://huggingface.co/facebook/VGGT-1B-Commercial) differ. Account approval is not automatically transferable to all users. |
| VGGT-Omega | [FAIR Noncommercial Research License](https://github.com/facebookresearch/vggt-omega/blob/main/LICENSE). This is separate from original VGGT's July 2025 change and restricts commercial use of materials and outputs. |
| VGG-T3 | [NVIDIA OneWay Noncommercial License](https://github.com/nv-dvl/vgg-ttt/blob/main/LICENSE), plus its third-party notices. |
| ACE0 | [Custom Niantic license](https://github.com/nianticlabs/acezero/blob/main/LICENSE): noncommercial use and source-format-only redistribution. A label does not authorize frozen binaries or client consulting use. |
| OpenSfM | [BSD-2-Clause style grant](https://github.com/mapillary/OpenSfM/blob/main/LICENSE); native dependencies require review. |
| 360DVO | [MIT code](https://github.com/chris1004336379/360DVO/blob/master/LICENSE); inherited code, native extensions and checkpoint redistribution must be checked separately. |
| COLMAP | [BSD license guidance](https://colmap.github.io/license.html), explicitly separate from dependencies of the selected build. |
| Nerfstudio and gsplat | Apache-2.0 top-level code; selected native extensions and optional models must be audited. |
| PFGS360 | [No complete top-level license grant established](https://github.com/zcq15/PFGS360). Some copied files have Apache notices; this does not license all original additions. Redistribution held pending clarification. |
| UniK3D | [CC-BY-NC-SA-4.0](https://github.com/lpiccinelli-eth/UniK3D/blob/main/LICENSE); model terms must also be established. |
| gsplat360 and Spherical Package | Apache-2.0 and BSD-3-Clause respectively; they do not override UniK3D or compiled CGAL terms. |
| NVIDIA 3DGRUT | [Apache-2.0 top level](https://github.com/nv-tlabs/3dgrut/blob/main/LICENSE), not generically research-only. SDKs and native dependencies still need review. |
| PyCuSFM | Apache-2.0 wrapper; cuSFM, TensorRT, cuDSS and native model distributions have separate terms. |
| LichtFeld Studio | [GPLv3 or later](https://github.com/MrNeRF/LichtFeld-Studio/blob/00a96b63c0ce45691d3dd9b4dcfd02069269c40c/LICENSE) plus the [GPU-library exception in the exact distributed revision](https://github.com/MrNeRF/LichtFeld-Studio/blob/00a96b63c0ce45691d3dd9b4dcfd02069269c40c/LICENSE-EXCEPTION.txt). Managed Windows CUDA candidate; matching corresponding source, dependency notices and source offer accompany its separate runtime archive. The older v0.5.4 tag lacked this exception. |
| RealityScan | Proprietary external application. No permission to repackage it is established by free eligibility. |
| SAM2 | [Apache-2.0 code and official checkpoints](https://github.com/facebookresearch/sam2#license); additional connected-component and font notices apply where included. |
| Manual learned matching | [LightGlue code and weights Apache-2.0](https://github.com/cvg/LightGlue#license), ALIKED BSD-3-Clause. SuperPoint has separate restrictive terms. |
| FFmpeg and GPU SDKs | Exact binary configurations and redistribution grants matter. Do not ship nonfree FFmpeg builds or arbitrary CUDA toolkit components. |

Free distribution does not make commercial use noncommercial. Downloading at first use is not a license loophole. Obtain qualified review of the final bill of materials and any rights-holder permissions before claiming unrestricted or commercial use of the whole suite. Attribution alone is not compliance with every license.
