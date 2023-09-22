import torch
import importlib
import torch.distributed as dist

# use_preprocess = 's1'
# P = importlib.import_module("guided_diffusion.preprocess."+use_preprocess)
# random_draw = getattr(P, 'random_draw')
# random_pick = getattr(P, 'random_pick')


def get_edges(t):
    edge = torch.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def preprocess_input(data, num_classes=184, use_rm=False):
    # move to GPU and change data types
    if use_rm:
        # if using random mask
        data["label"] = random_draw(random_pick(data["label"]))
    data["label"] = data["label"].long()

    # create one-hot label map
    if len(data["label"].size()) == 5:
        label_map = data["label"][0]
    else:
        label_map = data["label"]
    bs, _, h, w = label_map.size()
    input_label = torch.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if "instance" in data:
        inst_map = data["instance"]
        instance_edge_map = get_edges(inst_map)
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

    return {"y": input_semantics.cuda(), "label": data["label"]}


def get_sample(model, diffusion, cond, num_classes=184, clip_denoised=False, during_training=False, noise=None, use_rm=False, use_fp16=False):
    with torch.no_grad():
        if use_rm:
            model_kwargs = preprocess_input(cond, num_classes=num_classes, use_rm=True)
        else:
            model_kwargs = preprocess_input(cond, num_classes=num_classes)

        model_kwargs["s"] = 1.5  # * 2
        sample_fn = diffusion.p_sample_loop

        if during_training:
            model_kwargs["y"] = model_kwargs["y"][:1]
            cond = model_kwargs["y"]
        else:
            cond = model_kwargs["y"]

        if noise is None:
            noise = torch.randn((cond.shape[0], 3, cond.shape[2], cond.shape[3]), device=cond.device)
        if use_fp16:
            noise = noise.half()
            model_kwargs["y"] = model_kwargs["y"].half()
            model = model.half()

        sample = sample_fn(
            model,
            (cond.shape[0], 3, cond.shape[2], cond.shape[3]),
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
        )

        # gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
    return sample, model_kwargs
