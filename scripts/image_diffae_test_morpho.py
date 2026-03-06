import sys
sys.path.append("../")

import argparse
import os
import math

import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.utils import save_image

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.nn import reparameterize
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def save_grid(x: th.Tensor, path: str, nrow: int = 4):
    """
    Save a tensor batch as an image grid.
    x: [B, C, H, W]
    """
    x = x.detach().cpu()
    save_image(x, path, nrow=nrow, normalize=True, value_range=(-1, 1))


def infer_attr_stats(data, num_batches=10):
    """
    Peek a few batches to estimate min/max for cond['c'].
    Returns:
        mins: [D]
        maxs: [D]
    """
    all_c = []
    for _ in range(num_batches):
        batch, cond = next(data)
        if "c" not in cond:
            raise KeyError("cond does not contain key 'c'. Please check your dataset loader.")
        all_c.append(cond["c"].float().cpu())
    all_c = th.cat(all_c, dim=0)
    mins = all_c.min(dim=0).values
    maxs = all_c.max(dim=0).values
    return mins, maxs


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        split="test",
    )

    # ---- inspect one batch first ----
    batch0, cond0 = next(data)
    if "c" not in cond0:
        raise KeyError("cond['c'] not found. Your MorphoMNIST loader must provide cond['c'].")

    logger.log(f"batch shape: {tuple(batch0.shape)}")
    logger.log(f"cond keys: {list(cond0.keys())}")
    logger.log(f"cond['c'] shape: {tuple(cond0['c'].shape)}")
    print("Example cond['c'][:8]:")
    print(cond0["c"][:8])

    attr_dim = cond0["c"].shape[1]
    if attr_dim != 2:
        print(f"[Warning] Expected 2 attributes for MorphoMNIST, but got {attr_dim}.")

    if not (0 <= args.intervene_idx < attr_dim):
        raise ValueError(
            f"intervene_idx={args.intervene_idx} is out of range for cond['c'] with dim={attr_dim}"
        )

    # Re-create loader because we already consumed one batch above
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        split="test",
    )

    # Estimate attribute ranges from data if user did not specify manual min/max
    logger.log("estimating attribute range...")
    mins, maxs = infer_attr_stats(data, num_batches=args.stat_batches)
    print("Estimated cond['c'] mins:", mins)
    print("Estimated cond['c'] maxs:", maxs)

    # Re-create loader again because infer_attr_stats consumed some batches
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        split="test",
    )

    os.makedirs(args.out_dir, exist_ok=True)

    logger.log("sampling counterfactuals...")
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    saved = False
    count = 0

    while count < args.num_batches:
        batch, cond = next(data)
        batch = batch.to(dist_util.dev())

        # ---- encode to latent ----
        with th.no_grad():
            mu, var = model.rep_emb.encode(batch)
            # keep variance tiny, same spirit as your original code
            var = th.ones_like(mu) * args.latent_var_scale

            # ---- choose intervention value ----
            labels = cond["c"].clone().float()

            if args.intervene_value is not None:
                target_value = float(args.intervene_value)
            else:
                low = float(mins[args.intervene_idx].item())
                high = float(maxs[args.intervene_idx].item())
                target_value = np.random.uniform(low, high)

            labels[:, args.intervene_idx] = target_value

            # ---- normalize the chosen attribute to [roughly] latent control scale ----
            low = float(mins[args.intervene_idx].item())
            high = float(maxs[args.intervene_idx].item())

            if math.isclose(high, low):
                raise ValueError(
                    f"Attribute {args.intervene_idx} has zero range: min=max={low}"
                )

            target_norm = (target_value - low) / (high - low)

            # ---- intervene latent block ----
            # For MorphoMNIST in your original code, latent_dim=512 and one variable
            # was controlled by mu[:, 256:512]. We keep this behavior configurable.
            mu[:, args.latent_start:args.latent_end] = target_norm

            z = reparameterize(mu, var)

            # ---- partial noising from original image ----
            t = th.ones((batch.shape[0],), dtype=th.int64, device=dist_util.dev()) * args.timestep_respacing_test
            noise = th.randn_like(batch)
            x_t = diffusion.q_sample(batch, t, noise=noise)

            # ---- build cond for reverse diffusion ----
            model_cond = {}
            for k, v in cond.items():
                if th.is_tensor(v):
                    model_cond[k] = v.to(dist_util.dev())
                else:
                    model_cond[k] = v

            model_cond["z"] = z
            model_cond["c"] = labels.to(dist_util.dev())

            if "y" in model_cond and th.is_tensor(model_cond["y"]):
                model_cond["y"] = model_cond["y"].to(dist_util.dev())

            sample = sample_fn(
                model,
                (args.batch_size, args.in_channels, args.image_size, args.image_size),
                noise=x_t,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_cond,
            )

        # ---- save first batch only ----
        if not saved:
            save_grid(batch[:args.save_n], os.path.join(args.out_dir, "original_grid.png"))
            save_grid(sample[:args.save_n], os.path.join(args.out_dir, "counterfactual_grid.png"))

            np.save(
                os.path.join(args.out_dir, "original_batch.npy"),
                batch[:args.save_n].detach().cpu().numpy()
            )
            np.save(
                os.path.join(args.out_dir, "counterfactual_batch.npy"),
                sample[:args.save_n].detach().cpu().numpy()
            )

            print(f"Saved original grid to: {os.path.join(args.out_dir, 'original_grid.png')}")
            print(f"Saved counterfactual grid to: {os.path.join(args.out_dir, 'counterfactual_grid.png')}")
            print(f"Intervened attribute index: {args.intervene_idx}")
            print(f"Intervened target value: {target_value}")
            print(f"Latent block: [{args.latent_start}:{args.latent_end}]")
            saved = True

        count += 1

    dist.barrier()
    logger.log("done.")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir="",
        rep_cond=True,
        class_cond=True,
        in_channels=1,
        n_vars=2,

        # testing / saving
        num_batches=1,
        save_n=16,
        out_dir="./morphomnist_cf_test",

        # intervention
        intervene_idx=0,
        intervene_value=None,

        # latent block to modify
        latent_start=256,
        latent_end=512,

        # q_sample timestep for editing
        timestep_respacing_test=249,

        # tiny variance for reparameterization
        latent_var_scale=0.001,

        # how many batches to inspect for attr min/max
        stat_batches=10,
    )
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()