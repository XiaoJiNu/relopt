"""ZC self-collected dataset adapters (7Scenes-style layout), for eval_zc.py.

Layout (correct extrinsic chain; old data/zc chain deleted 2026-10-10):
    data/zc_fix/<scene>/<seq>/frame-XXXXXX.color.png   (undistorted 640x480 RGB)
    data/zc_fix/<scene>/<seq>/frame-XXXXXX.pose.txt    (4x4 c2w, map frame, metric)
"""
import os
import copy

import numpy as np

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2
from reloc3r.image_retrieval.topk_retrieval import PREPROCESS_FOLDER

ZC_PAIR_INFO_MASK = 'zc_{}_q-retrieval_db-step={}_topk={}.npy'

DATA_ROOT = './data/zc_fix'
# from in-bag camera_info (constant across bags, verified)
INTRINSICS = np.array([[366.0964660644531, 0, 321.33489990234375],
                       [0, 366.00811767578125, 242.17283630371094],
                       [0, 0, 1]], dtype=np.float32)


def label_to_str(label):
    return '_'.join(label)


def list_frames(folder):
    names = sorted(n for n in os.listdir(folder) if n.endswith('.color.png'))
    return [os.path.join(folder, n) for n in names]


class ZcRetrieval:
    """Database / query image list for NetVLAD retrieval (same interface as SevenScenesRetrieval)."""

    def __init__(self, scene, split):
        assert split in ('train', 'test')
        self.root_folder = DATA_ROOT
        self.scene = scene
        self.split = split
        self.names_color = list_frames('{}/{}/seq-01'.format(self.root_folder, self.scene))

    def load_image(self, name, device):
        color = imread_cv2(name)  # HxWx3 uint8 RGB
        import torch
        data = {'image': torch.from_numpy(color.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).to(device)}
        return data


class ZcRelpose(BaseStereoViewDataset):
    """(db, query) pairs from a retrieval pair-info file; db/query poses come from .pose.txt.

    query_parity: 'all' | 'even' | 'odd' — split queries by frame index parity so finetune
    train/eval use disjoint query frames (db frames intentionally shared, they are the map).
    data_root: optional override of DATA_ROOT (default ./data/zc_fix); 0923 bags still
    resolve via absolute/relative paths stored inside the pair npy.
    """

    def __init__(self, scene, pairs_info_file_mask=PREPROCESS_FOLDER + '/' + ZC_PAIR_INFO_MASK,
                 db_step=1, topk=10, pair_id=None, query_scene=None, query_parity='all',
                 data_root=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_folder = data_root or DATA_ROOT
        self.scene = scene
        # query images live in a different scene folder than the db (scene=db bag, query_scene=query bag)
        self.query_scene = query_scene or scene
        self.intrinsics = INTRINSICS
        self.pairs_path = pairs_info_file_mask.format(self.query_scene, db_step, topk)
        self.loaded_data = {}
        pairs_info = np.load(self.pairs_path, allow_pickle=True)
        for pairs in pairs_info:
            self.loaded_data[self._map_query_path(pairs[0]['query_path'])] = \
                [p['db_path'] for p in pairs]
        if query_parity in ('even', 'odd'):
            p0 = 0 if query_parity == 'even' else 1
            self.loaded_data = {q: d for q, d in self.loaded_data.items()
                                if int(q.split('frame-')[-1][:6]) % 2 == p0}
        self.queries = list(self.loaded_data)
        self.pair_id = pair_id
        self._pose_chain_guard()

    def _pose_chain_guard(self):
        """Fail fast if pose.txt comes from the deprecated chain (missing R_BL).

        Finalized chain: camera optical axis (z_cam = pose row 2) points ~10 deg below
        horizon => z_cam . map_up (pose[2,2]) in (-0.5, 0). Old glm chain gives +0.2 (camera
        flipped) / any other layout will show |.|>=0.5 -> treat as wrong chain. Samples 20 frames.
        """
        fs = [q.replace('.color.png', '.pose.txt') for q in self.queries[:20]]
        vals = []
        for f in fs:
            if os.path.exists(f):
                vals.append(float(np.loadtxt(f)[2, 2]))
        if not vals:
            return  # paths not readable here (e.g. retrieval-relative); eval scripts guard separately
        med = float(np.median(vals))
        assert -0.5 < med < 0.0, (
            f'ZC pose chain guard FAILED for {self.scene}: median cam-z.map-up={med:+.2f} '
            f'(finalized chain expects -0.5..0). Likely re-extracted with the old glm chain '
            f'(missing R_BL, ~2m pose error). Fix eval_zc.py extract / re-run fixpos-era check.')

    def _map_query_path(self, path):
        """Retrieval stored query paths of scene=db; rewrite scene dir to the query bag."""
        return path.replace('/{}/'.format(self.scene), '/{}/'.format(self.query_scene), 1)

    def __len__(self):
        return len(self.queries)

    def _get_views(self, idx, resolution, rng):
        image_path1 = self.queries[idx]
        image_paths2 = self.loaded_data[image_path1]
        if self.pair_id is None:
            image_path2 = image_paths2[rng.integers(0, len(image_paths2))]
        else:
            image_path2 = image_paths2[self.pair_id]

        views = []
        for view_path in [image_path2, image_path1]:  # [database, query]
            view_idx_splits = view_path.split('/')
            color_image = imread_cv2(view_path)
            pose = np.loadtxt(view_path.replace('.color.png', '.pose.txt')).astype(np.float32)
            color_image, intrinsics = self._crop_resize_if_necessary(
                color_image, copy.deepcopy(self.intrinsics), resolution, rng=rng)
            views.append(dict(
                img=color_image,
                camera_intrinsics=intrinsics,
                camera_pose=pose,
                dataset='ZC-' + self.query_scene,
                label=label_to_str(view_idx_splits[:-1]),
                instance=view_idx_splits[-1],
            ))
        return views
