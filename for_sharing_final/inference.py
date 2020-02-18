import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator


def main(img_path, weights_path='best_fpn_fpn_mobilenet_content0.5_feature0.006_adv0.001.h5', out_dir='predicted/'):
    imgs = sorted(glob(img_path))
    names = sorted([os.path.basename(x) for x in glob(img_path)])
    with open('config/deblur_solver.yaml') as cfg:
        config = yaml.load(cfg)
    model = get_generator(config['model'])
    model.load_state_dict(torch.load(weights_path)['model'])
    model = model.cuda()
    model.train(True)

    os.makedirs(out_dir, exist_ok=True)
    for name, img in tqdm(zip(names, imgs), total=len(names)):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img, _ = get_normalize()(img, img)
        mask = np.ones_like(img, dtype=np.float32)
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        img = np.pad(img, **pad_params)
        mask = np.pad(mask, **pad_params)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        mask = torch.from_numpy(np.expand_dims(np.transpose(mask, (2, 0, 1)), 0))
        with torch.no_grad():
            inputs = [img.cuda()]
            pred = model(*inputs)
        pred, = pred
        pred = pred.detach().cpu().float().numpy()
        pred = (np.transpose(pred, (1, 2, 0)) + 1) / 2.0 * 255.0
        pred =  pred.astype('uint8')[:h, :w, :]

        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, name), pred)
        