"""Two-stage learning with deterministic validation and resumable local checkpoints."""
import csv
import time
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import TemperatureDataset, check_dataset_metadata
from .prior import DiffusionPrior
from .unfolding import HQSUnfold
from .losses import reconstruction_loss, metrics
from .utils import seed_everything, seed_worker, file_hash, write_json, environment, load_checkpoint

SCHEMA = "sair-unfold-v1"

def check_data(data_dir, config):
    manifest = check_dataset_metadata(data_dir, config)
    for split in ("train", "val"):
        if file_hash(Path(data_dir) / f"{split}.npy") != manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} data changed since preparation")
    return file_hash(Path(data_dir) / "manifest.json")

def check_prior(state, config):
    if state.get("schema") != SCHEMA or state.get("kind") != "prior_ve":
        raise ValueError("Expected this project's VE prior checkpoint; old DDPM checkpoints are incompatible")
    for key in ("image_size", "t_min", "t_max", "base_channels", "channel_mult", "sigma_min", "sigma_max"):
        if state["config"][key] != getattr(config, key):
            raise ValueError(f"Prior checkpoint {key} differs from unfolding config")

def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])

@torch.no_grad()
def validate(model, dataset, config, device, physics=None):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(config.seed + 9000)
    totals, count = {}, 0
    for x in DataLoader(dataset, batch_size=config.batch_size, num_workers=0):
        x = x.to(device)
        if physics is None:
            values = {"val_dsm": float(model.dsm_loss(x, rng))}
        else:
            y = physics.simulate(x, rng)
            predicted = model(y, physics)
            _, terms = reconstruction_loss(predicted, x, y, physics, config)
            values = {"val_image_mse": terms["image_mse"],
                      **{f"val_{k}": v for k, v in metrics(predicted, x, y, physics, config).items()}}
        for key, value in values.items():
            totals[key] = totals.get(key, 0.) + value * len(x)
        count += len(x)
    return {key: value / count for key, value in totals.items()}

def train(config, data_dir, output, device, max_steps, validate_every=100,
          physics=None, physics_source=None, prior_checkpoint=None, finetune_prior=False,
          prior_kind="diffusion", projection="soft", resume=None, amp=False, log_every=100):
    config.validate()
    if min(max_steps, validate_every, log_every) < 1:
        raise ValueError("max_steps, validate_every and log_every must be positive")
    kind = "prior_ve" if physics is None else "unfold"
    if physics is not None and prior_kind == "diffusion" and prior_checkpoint is None and resume is None:
        raise ValueError("Pretrain a diffusion prior and supply --prior-checkpoint")
    if projection == "hard" and physics is not None:
        physics._hard_basis = physics.hard_basis()  # Fail before training if unsupported.
    seed_everything(config.seed)
    data_id = check_data(data_dir, config)
    training_data = TemperatureDataset(Path(data_dir) / "train.npy", config)
    validation_data = TemperatureDataset(Path(data_dir) / "val.npy", config)
    model = (DiffusionPrior(config) if physics is None else HQSUnfold(config, prior_kind, projection)).to(device)
    if prior_checkpoint and physics is not None and prior_kind == "diffusion" and not resume:
        state = load_checkpoint(prior_checkpoint, device)
        check_prior(state, config)
        model.prior.load_state_dict(state["model"])
    if physics is not None and prior_kind == "diffusion" and not finetune_prior:
        # Freeze weights, retain d D(z)/d z so stage parameters can learn.
        model.prior.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.)
    use_amp = bool(amp and device.type == "cuda" and physics is None)
    if amp and physics is not None:
        raise ValueError("Unfolding uses float32 complex physics/CG; AMP currently supported only in prior training")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    identity = {"prior_kind": prior_kind, "projection": projection,
                "finetune_prior": finetune_prior, "amp": use_amp}
    source = physics_source if physics is not None else None
    step, epoch, batch_offset, best_val = 0, 0, 0, float("inf")
    if resume:
        state = load_checkpoint(resume, device)
        if (state.get("schema") != SCHEMA or state["kind"] != kind or state["config"] != config.to_dict()
                or state["data_id"] != data_id or state["identity"] != identity or state["physics_source"] != source):
            raise ValueError("Resume configuration, data, physics, or training mode differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, epoch, batch_offset, best_val = state["step"], state["epoch"], state["batch_offset"], state["best_val"]
        restore_rng(state["rng"])
    if step >= max_steps:
        raise ValueError("max_steps must exceed checkpoint step")
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError("Training output already exists; choose a new output or --resume")
    output.mkdir(parents=True, exist_ok=bool(resume))
    config.save(output / "config.json")
    write_json(output / "environment.json", {**environment(), "device": str(device),
        "data_id": data_id, "kind": kind, "amp": use_amp,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "prior_checkpoint": str(prior_checkpoint) if prior_checkpoint else None,
        "prior_checkpoint_sha256": file_hash(prior_checkpoint) if prior_checkpoint else None})
    if physics is not None:
        write_json(output / "physics.json", {**physics.descriptor(), "source": source})

    def save(name):
        state = {"schema": SCHEMA, "kind": kind, "config": config.to_dict(),
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "step": step, "epoch": epoch,
                 "batch_offset": batch_offset, "best_val": best_val,
                 "rng": rng_state(), "data_id": data_id, "identity": identity,
                 "physics_source": source, "physics": physics.descriptor() if physics is not None else None}
        temporary = output / f"{name}.tmp"
        torch.save(state, temporary)
        temporary.replace(output / name)

    started = time.perf_counter()
    print(f"{kind}: start step={step}/{max_steps} device={device} "
          f"total_params={sum(p.numel() for p in model.parameters()):,} "
          f"trainable_params={sum(p.numel() for p in parameters):,} "
          f"batch_size={config.batch_size} train_images={len(training_data)} "
          f"steps_per_epoch={(len(training_data)+config.batch_size-1)//config.batch_size}"
          + (f" stages={config.stages} cg_iters={config.cg_iters}" if physics is not None else ""),
          flush=True)
    log_path = output / "training.csv"
    with log_path.open("a" if resume else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step","epoch","train_loss","validation","elapsed_s"])
        if not resume:
            writer.writeheader()
            handle.flush()
        while step < max_steps:
            loader_rng = torch.Generator().manual_seed(config.seed + epoch)
            loader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True,
                                generator=loader_rng, num_workers=config.num_workers,
                                worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
            for batch_index, x in enumerate(loader):
                if batch_index < batch_offset:
                    continue
                step_started = time.perf_counter()
                model.train()
                if physics is not None and prior_kind == "diffusion" and not finetune_prior:
                    model.prior.eval()
                x = x.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if physics is None:
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        loss = model.dsm_loss(x)
                else:
                    y = physics.simulate(x)
                    predicted = model(y, physics)
                    loss, _ = reconstruction_loss(predicted, x, y, physics, config)
                    if prior_kind == "diffusion" and finetune_prior and config.dsm_weight:
                        loss = loss + config.dsm_weight * model.prior.dsm_loss(x)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=not use_amp)
                scaler.step(optimizer)
                scaler.update()
                train_loss_value = float(loss.detach())
                step_seconds = time.perf_counter() - step_started
                step += 1
                batch_offset = batch_index + 1
                val = None
                if step % validate_every == 0 or step == max_steps:
                    val = validate(model, validation_data, config, device, physics)
                    selection = val["val_dsm" if physics is None else "val_image_mse"]
                    if selection < best_val:
                        best_val = selection
                        save("best.pt")
                    save("last.pt")
                if step == 1 or step % log_every == 0 or val is not None:
                    val_text = f" val={selection:.6f}" if val is not None else ""
                    memory_text = (f" gpu_peak_MiB={torch.cuda.max_memory_allocated(device)/2**20:.0f}"
                                   if device.type == "cuda" else "")
                    print(f"{kind}: step={step}/{max_steps} epoch={epoch+1} "
                          f"batch={batch_index+1}/{len(loader)} train={train_loss_value:.6f}"
                          f"{val_text} step_s={step_seconds:.3f}{memory_text}", flush=True)
                writer.writerow({"step":step,"epoch":epoch,"train_loss":train_loss_value,
                                 "validation":str(val) if val else "", "elapsed_s":time.perf_counter()-started})
                handle.flush()
                if step >= max_steps:
                    break
            if step < max_steps:
                epoch += 1
                batch_offset = 0
    return output / "best.pt"
