import os
import torch
import torch.nn.functional as F
import utils3d
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

import open3d as o3d
import numpy as np


import torch
import numpy as np
import trellis.models as models

# Load once and reuse
_encoder = (
    models.from_pretrained(
        "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
    )
    .eval()
    .cuda()
)

def get_voxels(voxel_path):
    if not os.path.exists(voxel_path):
        raise FileNotFoundError(voxel_path)

    position = utils3d.io.read_ply(voxel_path)[0]
    coords = ((torch.tensor(position) + 0.5) * 64).int().contiguous()

    ss = torch.zeros(1, 64, 64, 64, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss

@torch.no_grad()
def encode_voxels(voxels: torch.Tensor) -> np.ndarray:
    """
    Encode voxel grids with the default TRELLIS shape encoder.

    Args:
        voxels: Tensor of shape (1, 64, 64, 64) or
                (B, 1, 64, 64, 64)

    Returns:
        NumPy array containing the latent mean.
    """
    if voxels.ndim == 4:
        voxels = voxels.unsqueeze(0)  # -> (1, 1, D, H, W)

    voxels = voxels.float().cuda()

    latent = _encoder(voxels, sample_posterior=False)

    if not torch.isfinite(latent).all():
        raise ValueError("Non-finite latent values produced")

    return latent[0].cpu().numpy()

    
def get_latent(path) -> np.array:
    latent_dmg = np.load(path)
    return latent_dmg

def get_latent_mask(path):
    # Mask
    points_dmg = utils3d.io.read_ply(path)[0]
    points_dmg = torch.tensor(points_dmg, dtype=torch.float32)
    idx = torch.floor((points_dmg + 0.5) * 64).long()
    voxel_idx = idx.clamp(0, 63)
    
    flat_idx = (
        voxel_idx[:, 0] * 64 * 64 +
        voxel_idx[:, 1] * 64 +
        voxel_idx[:, 2]
    )

    mask_flat = torch.zeros(64 * 64 * 64)
    mask_flat.scatter_(0, flat_idx, 1.0)
    mask64 = mask_flat.view(64, 64, 64)
    
    latent_mask = F.max_pool3d(
        mask64[None, None].float(),
        kernel_size=4,
        stride=4
    )[0, 0]
    
    return latent_mask
    
def save_normalized_ply(ply_path: str, output_ply_path: str):
    points = utils3d.io.read_ply(ply_path)[0].astype(np.float32)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    scale = (maxs - mins).max()
    if scale == 0:
        raise ValueError("Point cloud has zero extent")

    points_norm = (points - mins) / scale - 0.5

    # (x, y, z) -> (z, y, x)
    points_norm = points_norm[:, [2, 1, 0]]

    utils3d.io.write_ply(output_ply_path, points_norm)

    return points_norm

# Load a pipeline from a model folder
pipeline = TrellisImageTo3DPipeline.from_pretrained("./models/image-large-sk-pc")
pipeline.cuda()

# Load an image
# Could use a different full render for materials.
image_render = Image.open("./var_12.png") 

image = Image.open("./var_12.png")


voxels_path = "var.ply"
latent_dmg = encode_voxels(get_voxels(voxels_path))
print(latent_dmg.shape)

latent_mask = get_latent_mask(voxels_path)

output_name = 'sample_3_1'

save_normalized_ply(voxels_path, f'./{output_name}_gt.ply')


# Run the pipeline
outputs = pipeline.run_inpaint(
    image_render,
    image,
    latent_dmg,
    latent_mask,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 30,
        "cfg_strength": 5
    },
    slat_sampler_params={
        "steps": 10,
        "cfg_strength": 3,
    }
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave(f"{output_name}_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave(f"{output_name}_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave(f"{output_name}_mesh.mp4", video, fps=30)

glb_struct = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb_struct.export(f"{output_name}_struct.glb")


glb_material = postprocessing_utils.to_glb(
    outputs['radiance_field'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024,
    fill_holes=False
)

glb_material.export(f"{output_name}_material.glb")

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply(f"{output_name}.ply")
