import argparse
import importlib
import torch
import os
import cv2
from tqdm import tqdm

import torch as th
import torchvision as tv
from natsort import natsorted
import glob
import numpy as np

from guided_diffusion.sample_utils import get_sample
from guided_diffusion.vis_util import tensor2label, tensor2im


device = "cuda" if torch.cuda.is_available() else "cpu"


def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--save_dir", help="Path to save results.", type=str, default="./results")
    parser.add_argument("--input_dir", help="Path to input images.", type=str, default="./assets/test_masks")
    parser.add_argument("--num_classes", help="num_classes", type=int, default=16)
    parser.add_argument("--image_size", help="image_size", type=int, default=256)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--num_samples", help="num_samples", type=int, default=16)
    parser.add_argument("--use_rm", help="Model module.", type=bool, default=False)
    parser.add_argument("--diffusion_steps", help="diffusion_steps", type=int, default=None)
    parser.add_argument("--clip_denoised", help="clip_denoised.", type=bool, default=False)
    parser.add_argument(
        "--eta",
        help="Amount of random noise in clipping sampling mode(recommended non-zero values only for not distilled model).",
        type=float,
        default=0,
    )
    return parser


def init_model(args):
    print(args.checkpoint)
    if os.path.isfile(args.checkpoint):
        n_timesteps = 1024
        if args.checkpoint.endswith(".pt"):
            ckpt = torch.load(args.checkpoint)
            time_scale = ckpt["time_scale"]
            teacher, teacher_diffusion = make_model(
                diffusion_steps=n_timesteps, time_scale=time_scale, num_classes=args.num_classes, image_size=args.image_size
            )
            teacher.eval()
            teacher.load_state_dict(ckpt["G"])
            del ckpt
        elif args.checkpoint.endswith(".safetensors"):
            from safetensors import safe_open

            ckpt = {}
            with safe_open(args.checkpoint, framework="pt", device=device) as f:
                for key in f.keys():
                    ckpt[key] = f.get_tensor(key)
            time_scale = float(os.path.basename(args.checkpoint).split("_")[-1].split(".")[0])
            teacher, teacher_diffusion = make_model(
                diffusion_steps=n_timesteps, time_scale=time_scale, num_classes=args.num_classes, image_size=args.image_size
            )
            teacher.eval()
            teacher.load_state_dict(ckpt)
        print("Model loaded.")

    else:
        raise ValueError("Checkpoint not found.")
    print(f"Num timesteps: {1024//time_scale}, time scale: {time_scale}.")

    print("Model loaded.")

    return teacher, teacher_diffusion


def sample(teacher, teacher_diffusion, cond, args, save_dir):
    path = cond["path"]
    samples, model_kwargs = get_sample(
        teacher, teacher_diffusion, cond, num_classes=args.num_classes, clip_denoised=args.clip_denoised, use_rm=args.use_rm, use_fp16=False
    )
    labels = np.array(
        [
            tensor2label(model_kwargs["label"][i], args.num_classes, imtype=np.uint8, tile=False, last2white=True)
            for i in range(model_kwargs["label"].shape[0])
        ]
    )
    samples = tensor2im(samples)
    print("samples", samples.shape, "labels", labels.shape)

    for i, (label, sample) in enumerate(zip(labels, samples)):
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(save_dir, "labels", os.path.basename(path[i])), label)
        # cv2.imwrite(os.path.join(save_dir, "samples", os.path.basename(path[i])), sample)
        result = np.concatenate([label, sample], axis=1)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(path[i])), result)


def sample_images(args, make_model, make_dataset, diffusion_steps=None):
    # device = torch.device("cuda")
    teacher, teacher_diffusion = init_model(args)

    mask_root = args.input_dir
    all_masks = natsorted(glob.glob(os.path.join(mask_root, "*.png")))
    print("total masks = ", len(all_masks))

    # bacth_size = 4
    all_masks = [all_masks[i : i + args.batch_size] for i in range(0, len(all_masks), args.batch_size)]

    print("len dataset = ", len(all_masks))
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)
    # os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

    for i, mask_path in enumerate(tqdm(all_masks)):
        masks = np.array(
            [
                cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
                for path in mask_path
            ]
        )
        masks = torch.from_numpy(masks).unsqueeze(1)
        print("masks", masks.shape, masks.dtype)
        cond = {"label": masks, "path": mask_path}
        sample(teacher, teacher_diffusion, cond, args, save_dir)
    print(f"created {len(all_masks)} samples")


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()

    from guided_diffusion.model_utils import make_model, make_dataset

    sample_images(args, make_model, make_dataset, diffusion_steps=args.diffusion_steps)
    print("sampling complete")
