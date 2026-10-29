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


# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("./models/image-large-sk")
pipeline.cuda()

# Load an image
image_render = Image.open("./var_2_4.png")

output_name = 'sample_2_6'


# Run the pipeline
outputs = pipeline.run(
    image_render,
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

# GLB files can be extracted from the outputs
# glb = postprocessing_utils.to_glb(
#     outputs['gaussian'][0],
#     outputs['mesh'][0],
#     # Optional parameters
#     simplify=0.95,          # Ratio of triangles to remove in the simplification process
#     texture_size=1024,      # Size of the texture used for the GLB
# )
# glb.export(f"{output_name}.glb")


glb_material = postprocessing_utils.to_glb(
    outputs['radiance_field'][0],   # use radiance field instead of gaussian
    outputs['mesh'][0],
    simplify=0.95,                   # slightly less simplification preserves UVs better
    texture_size=1024,              # higher resolution for better materials
    fill_holes=False
)

glb_material.export(f"{output_name}_material.glb")

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply(f"{output_name}.ply")
