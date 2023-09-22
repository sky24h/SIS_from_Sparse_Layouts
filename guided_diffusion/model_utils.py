import os
import sys
import torch

from . import dist_util, logger
from .image_datasets import load_data, ImageDataset, _list_image_files_recursively
from .resample import create_named_schedule_sampler
from .script_util import create_model_and_diffusion
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset


def make_model(diffusion_steps=1024, time_scale=1.0, num_classes=184, image_size=256, use_fp16=False):
    import torch

    print("torch version: ", torch.__version__)
    print(torch.cuda.is_available())

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        image_size=image_size,
        class_cond=True,
        learn_sigma=False,  # False to 3 channels, True to 6 channels, need to be investigated
        num_classes=num_classes,
        no_instance=True,
        num_channels=256,
        num_res_blocks=2,
        channel_mult="",
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32, 16, 8",
        dropout=0,
        diffusion_steps=diffusion_steps,
        noise_schedule="cosine",  # "linear","cosine",
        timestep_respacing="",
        time_scale=time_scale,
        use_kl=False,
        predict_xstart=False,
        predict_v=True,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=True,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=use_fp16,
        use_new_attention_order=False,
    )

    model.to(dist_util.dev())

    return model, diffusion


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="",
        dataset_mode="real_landscapes",
        image_size=256,
        class_cond=True,
        is_train=True,
        batch_size=1,
    ):
        logger.log("creating data loader...")

        if dataset_mode == "real_landscapes":
            all_files = _list_image_files_recursively(os.path.join(data_dir, "train_img" if is_train else "test_img"))
            classes = _list_image_files_recursively(os.path.join(data_dir, "train_mask" if is_train else "test_mask"))
            instances = None  # _list_image_files_recursively(os.path.join(data_dir, 'train_mask' if is_train else 'test_mask'))
        else:
            raise NotImplementedError("{} not implemented".format(dataset_mode))

        dataset_ = ImageDataset(
            dataset_mode,
            image_size,
            all_files,
            classes=classes,
            instances=instances,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=True,
            random_flip=True,
            is_train=is_train,
        )

        self.dataset = DataLoader(dataset_, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    def __getitem__(self, item):
        return next(iter(self.dataset))

    def __len__(self):
        return len(self.dataset)


def make_dataset(data_dir="", image_size=256, batch_size=1, is_train=True):
    return Dataset(data_dir=data_dir, image_size=image_size, batch_size=batch_size, is_train=is_train)


if __name__ == "__main__":
    model = make_model()
    data = make_dataset()
