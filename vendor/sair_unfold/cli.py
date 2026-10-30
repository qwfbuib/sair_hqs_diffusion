from pathlib import Path
import json
import numpy as np
import torch
from .operators import CrossGeometry, load_operator
from .physics import Physics
from .utils import file_hash

def add_physics_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--geometry", help="Cross-array geometry JSON")
    group.add_argument("--matrix", help="Complex G.npy or G.npz [M,H*W]")
    parser.add_argument("--matrix-key", default="G")
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--noise-sigma", type=float, help="sqrt(E|n|^2), in visibility units")
    noise.add_argument("--noise-vector", help="npy [M], per-visibility noise sigma")

def build_physics(args, config, device, checkpoint=None):
    expected = checkpoint["physics_source"] if checkpoint else None
    matrix, geometry = args.matrix, args.geometry
    matrix_key = args.matrix_key
    if checkpoint and not matrix and not geometry and expected["kind"] == "matrix":
        matrix_key = expected["matrix_key"]
    if checkpoint and not matrix and not geometry:
        if expected["kind"] == "matrix":
            matrix = expected["path"]
        else:
            op = load_operator(config.image_size, device)
            from .operators import CrossSAIROperator
            op = CrossSAIROperator(CrossGeometry(**expected["geometry"])).to(device)
    if matrix or geometry or not checkpoint:
        op = load_operator(config.image_size, device, geometry, matrix, matrix_key)
    if matrix:
        source = {"kind":"matrix", "path":str(Path(matrix).resolve()), "sha256":file_hash(matrix),
                  "matrix_key":matrix_key, "image_size":config.image_size}
    else:
        source = {"kind":"cross_spherical", "geometry":op.metadata()["geometry"]}
    if checkpoint:
        # A moved copy of the same matrix is valid; identity rests on bytes and key.
        comparable = lambda s: {k:v for k,v in s.items() if k != "path"}
        if comparable(source) != comparable(expected):
            raise ValueError("Operator differs from the unfolding training operator")
    sigma = np.load(args.noise_vector, allow_pickle=False) if args.noise_vector else args.noise_sigma
    if sigma is None and checkpoint:
        sigma = checkpoint["physics"]["noise_sigma"]
    physics = Physics(op, config, sigma).to(device)
    if checkpoint:
        saved = torch.tensor(checkpoint["physics"]["noise_sigma"], device=device)
        if not torch.equal(saved, physics.sigma):
            raise ValueError("Noise model differs from the unfolding training noise model")
        physics.spectral_scale.fill_(checkpoint["physics"]["spectral_scale"])
    return physics, source

def dataset_argument(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", help="Prepared Kelvin arrays and manifest.json")
    group.add_argument("--image-dir", help="UCMerced_LandUse/Images, or one category directory")
    parser.add_argument("--prepared-dir", help="New Kelvin data directory; default: <output>_data")
    parser.add_argument("--allow-resize", action="store_true")

def resolve_dataset(args, config):
    if args.data_dir:
        return args.data_dir
    if not args.prepared_dir:
        output = Path(args.output)
        args.prepared_dir = str(output.with_name(output.name + "_data"))
    from prepare_dataset import prepare
    prepare(args.image_dir, args.prepared_dir, config, args.allow_resize)
    return args.prepared_dir
