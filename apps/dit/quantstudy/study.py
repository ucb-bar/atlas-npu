"""Accuracy of NVFP4, MXFP4 and FP8 against FP32 on a pretrained DINOv3 encoder.

    python -m quantstudy.study run   OUT --model vits16 --data DATA [--hf-cache DIR]
    python -m quantstudy.study check OUT --model vits16 --data DATA [--hf-cache DIR]

DATA holds imagenette2-320/ and imagewoof2-320/ (fast.ai's 10-class ImageNet subsets, 320 px) and
umi.mp4, the file videos/observation.image/chunk-000/file-000.mp4 of the LeRobot dataset
lerobot/umi_cup_in_the_wild (a gripper-mounted fisheye camera).

run writes OUT/<model>.json and prints its tables. check runs each emulated matmul and the
bit-exact model of the same matmul (Atlas mxu1, Blackwell FP8, the Atlas NVFP4 MXU) on real layer
inputs and writes OUT/<model>-check.json. docs/nvfp4-accuracy.md describes the method.
"""
import argparse
import json
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from ditflow import numerics, quant
from npu_model.configs import fp8_mma, nvfp4
from . import formats as Fm
from . import vit as V

BF16 = V.Fmt(Fm.bf16, Fm.bf16, Fm.mm)
ATLAS = V.Fmt(partial(Fm.fp8, flush=True), partial(Fm.fp8, margin=2.0, flush=True), Fm.mm_ipt)
FP8 = V.Fmt(Fm.fp8, partial(Fm.fp8, margin=2.0), Fm.mm)
FP8_FLUSH = V.Fmt(ATLAS.weight, ATLAS.act, Fm.mm)
MXFP8 = V.Fmt(Fm.mxfp8, Fm.mxfp8, Fm.mm)
NVFP4 = V.Fmt(Fm.nvfp4, Fm.nvfp4, Fm.mm)
NVFP4_POW2 = V.Fmt(partial(Fm.nvfp4, pow2=True), partial(Fm.nvfp4, pow2=True), Fm.mm)
W4A8 = V.Fmt(Fm.mxfp4, partial(Fm.fp8, margin=2.0), Fm.mm)
W4A8_MX = V.Fmt(Fm.mxfp4, Fm.mxfp8, Fm.mm)
W4A16 = V.Fmt(Fm.nvfp4, Fm.bf16, Fm.mm)
W8 = V.Fmt(ATLAS.weight, Fm.exact, Fm.mm_ipt)  # FP8 weights, exact activations


def every(fmt: V.Fmt, **kinds) -> dict:
    """fmt for every kind of linear but those named."""
    return {**dict.fromkeys(V.KINDS, fmt), **kinds}


# name: (the linears' formats by kind, the attention matmuls' format or None for FP32)
CONFIGS = {
    "fp32": ({}, None),
    "bf16": (every(BF16), None),
    "atlas-fp8": (every(ATLAS), None),
    "fp8-fp32acc": (every(FP8), None),
    "fp8-fp32acc, flush": (every(FP8_FLUSH), None),
    "mxfp8": (every(MXFP8), None),
    "nvfp4": (every(NVFP4), None),
    "nvfp4-pow2": (every(NVFP4_POW2), None),
    "w4a8-mxfp4-fp8": (every(W4A8), None),
    "w4a8-mxfp4-mxfp8": (every(W4A8_MX), None),
    "w4a16-nvfp4": (every(W4A16), None),
    "atlas-fp8, pe bf16": (every(ATLAS, pe=BF16), None),
    "atlas-fp8, pe weights only": (every(ATLAS, pe=W8), None),
    "nvfp4, pe fp8": (every(NVFP4, pe=FP8), None),
    "nvfp4 mlp, fp8 rest": (every(FP8, up=NVFP4, down=NVFP4), None),
    "atlas-fp8+attn": (every(ATLAS), ATLAS),
    "fp8-fp32acc+attn": (every(FP8), FP8),
    "nvfp4+attn": (every(NVFP4), FP8),
    "w4a8-mxfp4-fp8+attn": (every(W4A8), FP8),
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
SHIFT = 3  # patches between the two crops of a correspondence pair


def folder(root: Path, size: int, per_class: int, skip: int = 0, seed: int = 0):
    """uint8 center crops and labels: per_class images of each class, a seeded draw after the first skip."""
    ds = ImageFolder(str(root), transform=T.Compose([T.Resize(size), T.CenterCrop(size), T.PILToTensor()]))
    targets = torch.tensor(ds.targets)
    g = torch.Generator().manual_seed(seed)
    pick = []
    for c in range(len(ds.classes)):
        members = (targets == c).nonzero().flatten()
        pick += members[torch.randperm(len(members), generator=g)][skip:skip + per_class].tolist()
    x, y = zip(*DataLoader(Subset(ds, pick), batch_size=64, num_workers=4))
    return torch.cat(x), torch.cat(y)


def umi(path: Path, pairs: int, calib: int):
    """Frames t and t + 1 at evenly spaced t, and calibration frames between them."""
    n = av.open(str(path)).streams.video[0].frames
    t = np.linspace(0, n - 2, pairs).astype(int)
    c = np.linspace(0, n - 2, calib, endpoint=False).astype(int) + (n - 2) // calib // 2
    want = {i: None for i in np.concatenate([t, t + 1, c])}
    for i, frame in enumerate(av.open(str(path)).decode(video=0)):
        if i in want:
            want[i] = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
    pick = lambda idx: torch.stack([want[i] for i in idx])
    return pick(t), pick(t + 1), pick(c)


def pixels(x: torch.Tensor) -> torch.Tensor:
    return (x.cuda().float() - MEAN.cuda()) / STD.cuda()


@torch.no_grad()
def tokens(vit, x: torch.Tensor, bs: int = 64) -> torch.Tensor:
    """CLS and patch tokens after the final norm (the registers dropped), FP32."""
    r = vit.config.num_register_tokens
    out = []
    for i in range(0, len(x), bs):
        h = vit(pixel_values=pixels(x[i:i + bs])).last_hidden_state
        out.append(torch.cat([h[:, :1], h[:, 1 + r:]], 1))
    return torch.cat(out)


def drift(t: torch.Tensor, ref: torch.Tensor, chunk: int = 256) -> dict:
    """Mean over images of the tokens' rel. L2 error, and the mean cosine of patch and CLS tokens."""
    rel, cos = [], []
    for a, b in zip(t.split(chunk), ref.split(chunk)):
        b = b.to(a.device)
        rel.append((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1))
        cos.append(F.cosine_similarity(a, b, dim=-1))
    cos = torch.cat(cos)
    return {"rel_l2": float(torch.cat(rel).mean()), "cos_patch": float(cos[:, 1:].mean()),
            "cos_cls": float(cos[:, 0].mean())}


def summary(t: torch.Tensor) -> torch.Tensor:
    return torch.cat([t[:, 0], t[:, 1:].mean(1)], -1)


def fit_probe(x: torch.Tensor, y: torch.Tensor, wd: float = 1e-3):
    """Multinomial logistic regression on standardized features."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    w = torch.zeros(x.shape[1], int(y.max()) + 1, device=x.device, requires_grad=True)
    b = torch.zeros(w.shape[1], device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")
    z = (x - mu) / sd

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z @ w + b, y) + wd * w.square().sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return lambda f: (((f - mu) / sd) @ w + b).argmax(-1)


def fit_knn(bank: torch.Tensor, labels: torch.Tensor, k: int = 20, temp: float = 0.07):
    """DINO's weighted k-NN classifier on cosine similarity."""
    bank = F.normalize(bank, dim=-1)

    def predict(f):
        sim, i = (F.normalize(f, dim=-1) @ bank.T).topk(k)
        votes = torch.zeros(len(f), int(labels.max()) + 1, device=f.device)
        return votes.scatter_add_(1, labels[i], (sim / temp).exp()).argmax(-1)
    return predict


def nearest(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """For each patch token of a, the index of the most similar patch token of b (cosine)."""
    return (F.normalize(a[:, 1:], dim=-1) @ F.normalize(b[:, 1:], dim=-1).mT).argmax(-1)


def truth(grid: int):
    """Patches of crop A that crop B (SHIFT patches down and right) contains, and their index in B."""
    i, j = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    valid = (i >= SHIFT) & (j >= SHIFT)
    return valid.flatten().cuda(), ((i - SHIFT) * grid + (j - SHIFT)).flatten().cuda()


def scale_census(t: torch.Tensor, amax: float, pow2: bool) -> torch.Tensor:
    """NVFP4 block scales: blocks, saturated (amax_b / 6g > 448), E4M3 subnormal, zero for a nonzero block."""
    raw = t.unflatten(-1, (-1, 16)).abs().amax(-1) / (Fm.E2M1_MAX * Fm.global_scale(amax, pow2))
    s = Fm.e4m3(raw)
    return torch.tensor([raw.numel(), int((raw > Fm.E4M3_MAX).sum()),
                         int(((s > 0) & (s < 2 ** -6)).sum()), int(((s == 0) & (raw > 0)).sum())])


def per_linear(vit, run: V.Run, x: torch.Tensor) -> tuple[dict, dict]:
    """Each linear's output error (no bias) against FP32 on the FP32 model's own inputs, and the
    NVFP4 block-scale census of the activations and weights."""
    err, ref = defaultdict(float), defaultdict(float)
    census = defaultdict(lambda: torch.zeros(4, dtype=torch.int64))

    def probe(m, a):
        a = a.flatten(0, -2)
        y = a @ m.weight.T
        ref[m.name] += float(y.square().sum())
        for name, (fmts, attn) in CONFIGS.items():
            fmt = fmts.get(m.kind)
            if fmt and attn is None:
                err[name, m.name, m.kind] += float((fmt.mm(fmt.act(a, run.amax[m.act]), m.wq(fmt)) - y).square().sum())
        for pow2 in (False, True):
            census["act", pow2] += scale_census(a, run.amax[m.act], pow2)

    run.probe = probe
    tokens(vit, x)
    run.probe = None
    for m in V.linears(vit):
        for pow2 in (False, True):
            census["weight", pow2] += scale_census(m.weight, m.wmax, pow2)
    kinds = defaultdict(lambda: defaultdict(list))
    for (name, layer, kind), e in err.items():
        kinds[name][kind].append((e / ref[layer]) ** 0.5)
        kinds[name]["all"].append((e / ref[layer]) ** 0.5)
    rel = {name: {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in d.items()}
           for name, d in kinds.items()}
    blocks = {f"{which}{'-pow2' if pow2 else ''}": dict(zip(("blocks", "saturated", "subnormal", "zero"),
                                                         c.tolist())) for (which, pow2), c in census.items()}
    return rel, blocks


def run_study(args) -> dict:
    vit = V.load(args.model, args.hf_cache)
    run = V.wrap(vit)
    data = args.data
    sets = {}
    for key, root in (("nette", data / "imagenette2-320"), ("woof", data / "imagewoof2-320")):
        sets[key] = {"train": folder(root / "train", 224, args.train_per_class),
                     "calib": folder(root / "train", 224, args.calib_per_class, skip=args.train_per_class)[0],
                     "val": folder(root / "val", 224, args.val_per_class)}
    pairs = folder(data / "imagenette2-320" / "val", 224 + 16 * SHIFT, args.pairs_per_class, seed=1)[0]
    crop_a, crop_b = pairs[..., :224, :224], pairs[..., 16 * SHIFT:, 16 * SHIFT:]
    u0, u1, ucal = umi(data / "umi.mp4", args.umi_pairs, args.umi_calib)
    batches = lambda x: (pixels(x[i:i + 64]) for i in range(0, len(x), 64))
    amax = {key: V.calibrate(vit, run, batches(s["calib"])) for key, s in sets.items()}
    amax["umi"] = V.calibrate(vit, run, batches(ucal))
    valid, target = truth(14)
    evalsets = {"nette": (sets["nette"]["val"][0], "nette"), "woof": (sets["woof"]["val"][0], "woof"),
                "a": (crop_a, "nette"), "b": (crop_b, "nette"), "u0": (u0, "umi"), "u1": (u1, "umi")}

    def out(key):
        x, domain = evalsets[key]
        run.amax = amax[domain]
        return tokens(vit, x)

    heads = {}
    for key in ("nette", "woof"):
        f, y = summary(tokens(vit, sets[key]["train"][0])), sets[key]["train"][1].cuda()
        heads[key] = {"probe": fit_probe(f, y), "knn": fit_knn(f, y)}
    ref = {key: out(key) for key in evalsets}
    ref_pred = {(key, h): fn(summary(ref[key])) for key, d in heads.items() for h, fn in d.items()}
    ref_nn = nearest(ref["a"], ref["b"]), nearest(ref["u0"], ref["u1"])
    ref = {key: ref[key].cpu() for key in ("nette", "woof", "u0")}

    results = {}
    for name, (fmts, attn) in CONFIGS.items():
        t0 = time.time()
        run.fmt, run.attn = fmts, attn
        r = {}
        for key in ("nette", "woof"):
            t, y = out(key), sets[key]["val"][1].cuda()
            r[key] = drift(t, ref[key])
            for h, fn in heads[key].items():
                p = fn(summary(t))
                r[key][f"{h}_acc"] = float((p == y).float().mean())
                r[key][f"{h}_agree"] = float((p == ref_pred[key, h]).float().mean())
        nn = nearest(out("a"), out("b"))
        r["pairs"] = {"acc": float((nn == target)[:, valid].float().mean()),
                      "agree": float((nn == ref_nn[0])[:, valid].float().mean())}
        t = out("u0")
        r["umi"] = drift(t, ref["u0"])
        r["umi"]["nn_agree"] = float((nearest(t, out("u1")) == ref_nn[1]).float().mean())
        r["seconds"] = time.time() - t0
        results[name] = r
        print(name, json.dumps(r), flush=True)
    run.fmt, run.attn = {}, None
    lin = {}
    for key, x in (("nette", sets["nette"]["val"][0]), ("umi", u0)):
        run.amax = amax[key]
        lin[key] = per_linear(vit, run, x[::len(x) // 32][:32])
    return {"model": V.MODELS[args.model], "args": {k: str(v) for k, v in vars(args).items()},
            "sizes": {k: len(v[0]) for k, v in evalsets.items()} | {f"{k}_train": len(s["train"][0])
                                                                     for k, s in sets.items()},
            "fp32": {key: {h: float((ref_pred[key, h] == sets[key]["val"][1].cuda()).float().mean())
                           for h in ("probe", "knn")} for key in ("nette", "woof")},
            "configs": results, "per_linear": {k: v[0] for k, v in lin.items()},
            "nvfp4_scales": {k: v[1] for k, v in lin.items()}}


def table(res: dict) -> str:
    rows = ["| config | linear out. rel. L2 (mean / max) | features rel. L2, cos (ImageNet) | features rel. L2 (UMI) "
            "| k-NN acc. / agree (Imagewoof) | probe acc. / agree (Imagewoof) | correspondence acc. | UMI NN agree |",
            "|---|---|---|---|---|---|---|---|"]
    lin = res["per_linear"]["nette"]
    for name, r in res["configs"].items():
        l = lin.get(name.split("+")[0])  # +attn: the same linears
        l = f"{l['all']['mean']:.2%} / {l['all']['max']:.2%}" if l else "-"
        n, w = r["nette"], r["woof"]
        rows.append(f"| {name} | {l} | {n['rel_l2']:.2%}, {n['cos_patch']:.4f} | {r['umi']['rel_l2']:.2%} "
                    f"| {w['knn_acc']:.1%} / {w['knn_agree']:.1%} | {w['probe_acc']:.1%} / {w['probe_agree']:.1%} "
                    f"| {r['pairs']['acc']:.1%} | {r['umi']['nn_agree']:.1%} |")
    kinds = ["pe", "qkv", "o", "up", "down"]
    rows += ["", "| config | " + " | ".join(kinds) + " |", "|---" * (len(kinds) + 1) + "|"]
    for name, d in lin.items():
        rows.append(f"| {name} | " + " | ".join(f"{d[k]['mean']:.2%}" for k in kinds) + " |")
    rows += ["", "| NVFP4 block scales | blocks | saturated | E4M3 subnormal | zero |", "|---|---|---|---|---|"]
    for domain, d in res["nvfp4_scales"].items():
        for k, c in d.items():
            rows.append(f"| {domain} {k} | {c['blocks']} | {c['saturated'] / c['blocks']:.2e} "
                        f"| {c['subnormal'] / c['blocks']:.2e} | {c['zero'] / c['blocks']:.2e} |")
    return "\n".join(rows)


def e2m1_codes(q: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(q.abs(), torch.tensor(Fm.E2M1, device=q.device)) | ((q < 0).long() << 3)


def nv_operand(x: torch.Tensor, amax: float):
    """E2M1 codes, E4M3 scale bytes and the global scale of x, as formats.nvfp4 rounds them."""
    g = Fm.global_scale(amax, False)
    s = Fm.nvfp4_scales(x, g)
    sg = (s * g)[..., None]
    q = torch.where(sg == 0, 0.0, Fm.e2m1(x.unflatten(-1, (-1, 16)) / torch.where(sg == 0, 1.0, sg)))
    return e2m1_codes(q.flatten(-2)).tolist(), s.to(torch.float8_e4m3fn).view(torch.uint8).tolist(), g


def gap(fast: torch.Tensor, exact: torch.Tensor, y: torch.Tensor) -> dict:
    """How far an emulation is from the bit-exact model, next to the format's own error."""
    return {"identical": float((fast == exact).float().mean()),
            "rel_l2_vs_exact": float((fast - exact).norm() / exact.norm()),
            "rel_l2_format": float((exact - y).norm() / y.norm())}


@torch.no_grad()
def check(args) -> dict:
    """The study's matmul emulations against the bit-exact models, on real inputs of four layers."""
    vit = V.load(args.model, args.hf_cache)
    run = V.wrap(vit)
    calib, _ = folder(args.data / "imagenette2-320" / "train", 224, 2)
    x, _ = folder(args.data / "imagenette2-320" / "val", 224, 1)
    run.amax = V.calibrate(vit, run, [pixels(calib)])
    last = len(vit.model.layer) - 1
    layers = {"pe", "0.q", f"{last // 2}.up", f"{last}.down"}
    seen = {}
    run.probe = lambda m, a: seen.setdefault(m.name, (m, a.flatten(0, -2)[:args.rows]))
    tokens(vit, x[:2])
    run.probe = None
    out = {}
    for name in sorted(layers):
        m, a = seen[name]
        w, amax = m.weight, run.amax[m.act]
        y = (a @ w.T).cpu()
        ea, ew = quant.exponent_for(amax, 2.0), quant.exponent_for(m.wmax)
        k = a.shape[1]

        ca, cw = quant.to_fp8(a, ea).cpu(), quant.to_fp8(w, ew).cpu()
        acc = torch.zeros(len(a), len(w), dtype=torch.bfloat16)
        for i in range(0, k, 32):
            acc = numerics.ipt_matmul(ca[:, i:i + 32].view(torch.float8_e4m3fn),
                                      cw[:, i:i + 32].view(torch.float8_e4m3fn), acc)
        fast = ATLAS.mm(ATLAS.act(a, amax), ATLAS.weight(w, m.wmax)).cpu()
        out[name, "atlas-ipt"] = gap(fast, acc.float() * 2.0 ** (ea + ew), y)

        ca = (a * 2.0 ** -ea).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
        cw = (w * 2.0 ** -ew).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
        acc = np.zeros((len(ca), len(cw)), np.uint32)
        for i in range(0, k, 32):
            acc = fp8_mma.dot(ca[:, None, i:i + 32], cw[None, :, i:i + 32], acc)
        exact = torch.from_numpy(acc.view(np.float32).copy()) * 2.0 ** (ea + ew)
        out[name, "blackwell-fp8"] = gap(FP8.mm(FP8.act(a, amax), FP8.weight(w, m.wmax)).cpu(), exact, y)

        rows, cols = slice(0, args.nv_rows), slice(0, args.nv_cols)
        qa, sa, ga = nv_operand(a[rows], amax)
        qw, sw, gw = nv_operand(w[cols], m.wmax)
        acc = [[0] * len(qw) for _ in qa]
        for i in range(k // 64):
            acc = [[nvfp4.dot(qa[r][64 * i:64 * i + 64], qw[j][64 * i:64 * i + 64],
                              sa[r][4 * i:4 * i + 4], sw[j][4 * i:4 * i + 4], acc[r][j])
                    for j in range(len(qw))] for r in range(len(qa))]
        exact = torch.tensor(acc, dtype=torch.int64).to(torch.int32).view(torch.float32) * torch.tensor(ga * gw)
        fast = NVFP4.mm(NVFP4.act(a[rows], amax), NVFP4.weight(w[cols], m.wmax)).cpu()
        out[name, "atlas-nvfp4"] = gap(fast, exact, y[rows, cols])
    return {f"{layer} {kind}": v for (layer, kind), v in out.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "check"):
        p = sub.add_parser(cmd)
        p.add_argument("out", type=Path)
        p.add_argument("--model", default="vits16", choices=sorted(V.MODELS))
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--hf-cache")
    r = sub.choices["run"]
    r.add_argument("--train-per-class", type=int, default=600)
    r.add_argument("--calib-per-class", type=int, default=13)
    r.add_argument("--val-per-class", type=int, default=1000, help="all of them")
    r.add_argument("--pairs-per-class", type=int, default=100)
    r.add_argument("--umi-pairs", type=int, default=1024)
    r.add_argument("--umi-calib", type=int, default=128)
    c = sub.choices["check"]
    c.add_argument("--rows", type=int, default=64)
    c.add_argument("--nv-rows", type=int, default=8)
    c.add_argument("--nv-cols", type=int, default=64)
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "run":
        res = run_study(args)
        (args.out / f"{args.model}.json").write_text(json.dumps(res, indent=1))
        print(table(res))
    else:
        res = check(args)
        (args.out / f"{args.model}-check.json").write_text(json.dumps(res, indent=1))
        for k, v in res.items():
            print(k, json.dumps(v))


if __name__ == "__main__":
    main()
