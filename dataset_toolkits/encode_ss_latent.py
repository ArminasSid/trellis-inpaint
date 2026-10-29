import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis.models as models

torch.set_grad_enabled(False)


def is_damaged(instance):
    return instance.endswith("_dmg")


def get_voxels(instance):

    if is_damaged(instance):
        base_sha = instance[:-4]
        voxel_path = os.path.join(opt.output_dir, "voxels_dmg", f"{base_sha}.ply")
    else:
        voxel_path = os.path.join(opt.output_dir, "voxels", f"{instance}.ply")

    if not os.path.exists(voxel_path):
        raise FileNotFoundError(voxel_path)

    position = utils3d.io.read_ply(voxel_path)[0]
    coords = ((torch.tensor(position) + 0.5) * opt.resolution).int().contiguous()

    ss = torch.zeros(1, opt.resolution, opt.resolution, opt.resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


def latent_path(instance):

    if is_damaged(instance):
        sha = instance[:-4]
        return os.path.join(opt.output_dir, "ss_latents_dmg", latent_name, f"{sha}.npz")
    else:
        return os.path.join(opt.output_dir, "ss_latents", latent_name, f"{instance}.npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None)
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16')
    parser.add_argument('--model_root', type=str, default='results')
    parser.add_argument('--enc_model', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--instances', type=str, default=None)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    opt = parser.parse_args()
    opt = edict(vars(opt))

    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model}_{opt.ckpt}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')

    os.makedirs(os.path.join(opt.output_dir, "ss_latents", latent_name), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "ss_latents_dmg", latent_name), exist_ok=True)

    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')

    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['voxelized'] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    records = []

    sha256s = list(metadata['sha256'].values)

    instances = []

    for sha in sha256s:
        instances.append(sha)

        dmg_voxel = os.path.join(opt.output_dir, "voxels_dmg", f"{sha}.ply")
        if os.path.exists(dmg_voxel):
            instances.append(f"{sha}_dmg")

    filtered = []

    for inst in instances:
        path = latent_path(inst)

        if os.path.exists(path):
            records.append({
                'sha256': inst[:-4] if is_damaged(inst) else inst,
                f'ss_latent_{latent_name}': True
            })
        else:
            filtered.append(inst)

    instances = filtered

    print("Total instances:", len(instances))
    print("Damaged instances:", sum(is_damaged(i) for i in instances))

    load_queue = Queue(maxsize=4)

    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
             ThreadPoolExecutor(max_workers=32) as saver_executor:

            def loader(inst):
                try:
                    ss = get_voxels(inst)[None].float()
                    load_queue.put((inst, ss))
                except Exception as e:
                    print(f"Error loading {inst}: {e}")

            for inst in instances:
                loader_executor.submit(loader, inst)

            def saver(inst, pack):

                save_path = latent_path(inst)
                np.savez_compressed(save_path, **pack)

                records.append({
                    'sha256': inst[:-4] if is_damaged(inst) else inst,
                    f'ss_latent_{latent_name}': True
                })

            for _ in tqdm(range(len(instances)), desc="Extracting latents"):

                inst, ss = load_queue.get()

                ss = ss.cuda().float()
                latent = encoder(ss, sample_posterior=False)

                assert torch.isfinite(latent).all(), "Non-finite latent"

                pack = {
                    'mean': latent[0].cpu().numpy(),
                }

                saver_executor.submit(saver, inst, pack)

            saver_executor.shutdown(wait=True)

    except:
        print("Error happened during processing.")

    records = pd.DataFrame.from_records(records)
    records = records.drop_duplicates(subset="sha256", keep="first")
    records.to_csv(os.path.join(opt.output_dir, f'ss_latent_{latent_name}_{opt.rank}.csv'), index=False)
