import os
import random
import json
import yaml
import time
import numpy as np
import torch
from types import SimpleNamespace
import matplotlib.pyplot as plt


from torchmetrics import JaccardIndex, Dice
from torchdiffeq import odeint

from datasets.config_dl import config_dl
from models import unet_segdiff

def set_seed(SEED=0):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def show_images(images, vmin=None, vmax=None, save_name="", overlay=None, cmap='gray'):
    alpha = 0.6 if overlay is not None else 1.0
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if overlay is not None:
        overlay = overlay.detach().cpu().numpy()
    if vmin is None:
        vmin = images.min().item()
    if vmax is None:
        vmax = images.max().item()

    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** 0.5)
    cols = round(len(images) / rows)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(images):
                continue
            fig.add_subplot(rows, cols, idx + 1)
            if overlay is not None:
                plt.imshow(overlay[idx][0], cmap="gray")
                mask = np.ma.masked_where(images[idx][0] == 0, images[idx][0])
                plt.imshow(mask, alpha=alpha)
            else:
                if images.shape[1] == 1:
                    plt.imshow(images[idx][0], cmap=cmap, vmin=vmin, vmax=vmax)
                else:
                    plt.imshow(images[idx].transpose(1, 2, 0), vmin=vmin, vmax=vmax)
            plt.axis("off")
            idx += 1
    plt.savefig(save_name, bbox_inches="tight", dpi=250)
    plt.close()


def compute_metrics(m, m_gt, thresh, corr_mode, num_classes):
    if corr_mode == 'sdf':
        m_thresh = (m <= 3 * thresh).type(torch.int8).squeeze().cpu()
        m_gt_thresh = (m_gt <= 0.).type(torch.int8).squeeze().cpu()
    elif corr_mode == 'binary':
        m_thresh = (m >= 0.5).type(torch.int8).squeeze().cpu()
        m_gt_thresh = (m_gt >= 0.5).type(torch.int8).squeeze().cpu()

    if num_classes > 1:
        m_thresh = argmax_multiclass_sdf(m_thresh)
        m_gt_thresh = argmax_multiclass_sdf(m_gt_thresh)
        jaccard = JaccardIndex(task='multiclass', num_classes=num_classes+1)
        dice = Dice(average="macro", num_classes=num_classes+1)
        iou, dice_score = jaccard(m_thresh, m_gt_thresh), dice(m_thresh, m_gt_thresh)
        return iou, dice_score
    else:
        jaccard = JaccardIndex(task="binary")
        dice = Dice(task='binary', average='macro', num_classes=2, ignore_index=0)
        iou, dice_score = jaccard(m_thresh, m_gt_thresh), dice(m_thresh, m_gt_thresh)
        return iou, dice_score


def argmax_multiclass_sdf(m_thresh):
    m_thresh = torch.cat((torch.zeros_like(m_thresh)[:, 0][:, None], m_thresh), 1)
    return torch.argmax(m_thresh, 1, True)


class Sampling:
    def __init__(self, net, sigma_min, ode_steps, device, load_path, sz, img_cond, corr_mode, thresh, save_images=True):
        self.net = net
        self.device = device
        self.load_path = load_path
        self.sz = sz
        self.img_cond = img_cond
        self.save_images = save_images
        self.sigma_min = sigma_min
        self.ode_steps = ode_steps
        self.thresh = thresh
        self.corr_mode = corr_mode

    def sample(self, x, m_gt, n_samples):
        m0 = torch.randn_like(m_gt)

        def func_conditional(t, m):
            t_curr = torch.ones(m.shape[0], device=m.device) * t
            m_ipt = (1 - (1 - self.sigma_min) * t) * m0 + t * m
            return self.net(m_ipt, t_curr.reshape(m.shape[0], -1), img_cond=x)

        start_time = time.time()
        with torch.no_grad():
            traj = odeint(
                func_conditional,
                y0=m0,
                t=torch.linspace(0, 1, self.ode_steps, device=x.device),
                method='euler',
                atol=1e-5,
                rtol=1e-5
            )
        print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

        if self.save_images:
            x_plot = (x + 1.) / 2. if x.shape[1] > 1 else x
            show_images(x_plot, save_name=self.load_path + "/samples/x.png")
            show_images(traj[-1], save_name=self.load_path + "/samples/m.png")

            if self.corr_mode == 'sdf':
                m_thresh = (traj[-1] <= 3. * self.thresh).type(torch.int8)
            else:
                m_thresh = (traj[-1] >= 0.5).type(torch.int8)

            show_images(m_thresh, save_name=self.load_path + "/samples/m_thresh.png")
            show_images(m_thresh, save_name=self.load_path + "/samples/m_thresh_overlay.png", overlay=x_plot)

            m_gt_thresh = (m_gt <= 0.).type(torch.int8) if self.corr_mode == 'sdf' else (m_gt >= 0.5).type(torch.int8)
            show_images(m_gt_thresh, save_name=self.load_path + "/samples/m_gt.png")
            show_images(m_gt, save_name=self.load_path + "/samples/m_gt_sdf.png")

        return traj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg/glas.yaml", type=str, help="path to .yaml config")
    parser.add_argument("--seed", default=0, type=int, help="seed for reproducibility")
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = json.loads(json.dumps(yaml.safe_load(file)), object_hook=lambda d: SimpleNamespace(**d))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed)

    train_dl, test_dl = config_dl(cfg)

    model = unet_segdiff.UNetModel(
        in_channels=cfg.model.n_cin,
        model_channels=cfg.model.n_fm,
        out_channels=cfg.model.n_cin,
        num_res_blocks=3,
        attention_resolutions=(16, 8),
        channel_mult=tuple(cfg.model.mults),
        dims=2,
        rrdb_blocks=12,
    ).to(device)

    load_path = os.path.join(os.getcwd(), "runs", cfg.general.modality)

    exp_name = sorted(os.listdir(load_path))[-1] if cfg.inference.latest else cfg.inference.load_exp
    load_path = os.path.join(load_path, exp_name)
    model_dir = os.path.join(load_path, cfg.inference.load_model_dir)

    print(f"Loading model from {model_dir}")
    os.makedirs(os.path.join(load_path, "samples"), exist_ok=True)

    cp_name = sorted(f for f in os.listdir(model_dir) if f.endswith(".pt"))[-1] \
        if cfg.inference.load_cp == 'None' else cfg.inference.load_cp

    model.load_state_dict(torch.load(os.path.join(model_dir, cp_name), map_location=device)["state_dict"])
    model.eval()

    if cfg.general.corr_mode in ["sdf", "binary"]:
        sampler = Sampling(
            net=model,
            sigma_min=cfg.fm.sigma_min,
            ode_steps=cfg.inference.ode_steps,
            device=device,
            load_path=load_path,
            sz=cfg.general.sz,
            img_cond=cfg.general.img_cond,
            corr_mode=cfg.general.corr_mode,
            thresh=cfg.inference.thresh,
            save_images=True
        )

        batch = next(iter(test_dl))
        x = batch['image'].to(device) if cfg.general.img_cond else None
        m_gt = batch['mask'].to(device) if cfg.general.img_cond else None

        m_seeds = [sampler.sample(x, m_gt, cfg.inference.n_samples)[-1].detach().cpu().numpy()
                   for _ in range(cfg.inference.n_eval)]

        m_pred = torch.from_numpy(np.mean(np.array(m_seeds), axis=0)) if cfg.inference.n_eval > 1 else torch.tensor(m_seeds[0])

        iou, dice = compute_metrics(m_pred, m_gt.cpu(), cfg.inference.thresh, cfg.general.corr_mode, cfg.model.n_cin)
        print(f"\nFinal metrics: IoU {iou:.4f}, Dice {dice:.4f}")
