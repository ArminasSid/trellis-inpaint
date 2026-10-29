import os
import copy
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from torchvision import transforms
from PIL import Image

torch.set_grad_enabled(False)


def get_data(frames, sha256, render_dir):
    with ThreadPoolExecutor(max_workers=16) as executor:

        def worker(view):
            image_path = os.path.join(
                opt.output_dir, render_dir, sha256, view['file_path']
            )

            if not os.path.exists(image_path):
                print(f"Missing image: {image_path}")
                return None

            try:
                image = Image.open(image_path)
            except Exception:
                print(f"Error loading image {image_path}")
                return None

            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            c2w = torch.tensor(view['transform_matrix'])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)

            fov = view['camera_angle_x']
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(
                torch.tensor(fov), torch.tensor(fov)
            )

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics
            }

        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None)
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg')
    parser.add_argument('--instances', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = edict(vars(parser.parse_args()))

    feature_name_clean = opt.model
    feature_name_dmg = f"{opt.model}_dmg"

    feature_col_clean = f"feature_{feature_name_clean}"
    feature_col_dmg = f"feature_{feature_name_dmg}"

    os.makedirs(os.path.join(opt.output_dir, 'features', feature_name_clean), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'features', feature_name_dmg), exist_ok=True)

    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dinov2_model.eval().cuda()

    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    n_patch = 518 // 14

    metadata_path = os.path.join(opt.output_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise ValueError("metadata.csv not found")

    metadata = pd.read_csv(metadata_path)

    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[
                metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score
            ]

        if feature_col_clean in metadata.columns:
            metadata = metadata[metadata[feature_col_clean] == False]

        if feature_col_dmg in metadata.columns:
            metadata = metadata[metadata[feature_col_dmg] == False]

        metadata = metadata[
            (metadata['voxelized'] == True) &
            (metadata['rendered'] == True)
        ]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    sha256s = list(metadata['sha256'].values)
    records = []

    def process_variant(render_dir, voxel_dir, feature_name, feature_col):

        feature_dir = os.path.join(opt.output_dir, 'features', feature_name)

        load_queue = Queue(maxsize=4)

        with ThreadPoolExecutor(max_workers=8) as loader_executor, \
             ThreadPoolExecutor(max_workers=8) as saver_executor:

            def loader(sha256):
                try:
                    transform_path = os.path.join(
                        opt.output_dir, render_dir, sha256, 'transforms.json'
                    )

                    if not os.path.exists(transform_path):
                        return

                    with open(transform_path, 'r') as f:
                        meta = json.load(f)

                    frames = meta['frames']
                    data = []

                    for datum in get_data(frames, sha256, render_dir):
                        datum['image'] = transform(datum['image'])
                        data.append(datum)

                    voxel_path = os.path.join(
                        opt.output_dir, voxel_dir, f'{sha256}.ply'
                    )

                    if not os.path.exists(voxel_path):
                        return

                    positions = utils3d.io.read_ply(voxel_path)[0]

                    load_queue.put((sha256, data, positions))

                except Exception as e:
                    print(f"Loader error {sha256}: {e}")

            loader_executor.map(loader, sha256s)

            def saver(sha256, pack, patchtokens, uv):
                try:
                    pack['patchtokens'] = F.grid_sample(
                        patchtokens,
                        uv.unsqueeze(1),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(2).permute(0, 2, 1).cpu().numpy()

                    pack['patchtokens'] = np.mean(
                        pack['patchtokens'], axis=0
                    ).astype(np.float16)

                    save_path = os.path.join(feature_dir, f'{sha256}.npz')
                    np.savez_compressed(save_path, **pack)

                    records.append({
                        'sha256': sha256,
                        feature_col: True
                    })

                except Exception as e:
                    print(f"Saver error {sha256}: {e}")

            for _ in tqdm(range(len(sha256s)), desc=f"Extracting {feature_name}"):

                sha256, data, positions = load_queue.get()

                positions = torch.from_numpy(positions).float().cuda()
                indices = ((positions + 0.5) * 64).long()

                if not (torch.all(indices >= 0) and torch.all(indices < 64)):
                    print(f"Out-of-bounds indices: {sha256}")
                    continue

                pack = {
                    'indices': indices.cpu().numpy().astype(np.uint8),
                }

                patchtokens_lst = []
                uv_lst = []

                for i in range(0, len(data), opt.batch_size):

                    batch = data[i:i+opt.batch_size]
                    bs = len(batch)

                    images = torch.stack([d['image'] for d in batch]).cuda()
                    extrinsics = torch.stack([d['extrinsics'] for d in batch]).cuda()
                    intrinsics = torch.stack([d['intrinsics'] for d in batch]).cuda()

                    features = dinov2_model(images, is_training=True)

                    uv = utils3d.torch.project_cv(
                        positions, extrinsics, intrinsics
                    )[0] * 2 - 1

                    patchtokens = features['x_prenorm'][
                        :, dinov2_model.num_register_tokens + 1:
                    ].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)

                    patchtokens_lst.append(patchtokens)
                    uv_lst.append(uv)

                patchtokens = torch.cat(patchtokens_lst, dim=0)
                uv = torch.cat(uv_lst, dim=0)

                saver_executor.submit(saver, sha256, pack, patchtokens, uv)

            saver_executor.shutdown(wait=True)

    process_variant(
        render_dir='renders',
        voxel_dir='voxels',
        feature_name=feature_name_clean,
        feature_col=feature_col_clean
    )

    process_variant(
        render_dir='renders_dmg',
        voxel_dir='voxels_dmg',
        feature_name=feature_name_dmg,
        feature_col=feature_col_dmg
    )

    records_df = pd.DataFrame.from_records(records)

    records_df = records_df[['sha256', feature_col_clean]]

    records_df = records_df.drop_duplicates(subset='sha256', keep='first')

    merged_dir = os.path.join(opt.output_dir, "merged_records")
    os.makedirs(merged_dir, exist_ok=True)

    timestamp = int(__import__("time").time())

    csv_path = os.path.join(
        merged_dir,
        f"{timestamp}_feature_{opt.rank}.csv"
    )

    records_df.to_csv(csv_path, index=False)