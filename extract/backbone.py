"""
STAGE A -- backbone + multi-layer hooks  [runs on RunPod GPU]

Loads a CIFAR ResNet and hooks MULTIPLE layers simultaneously (stage1/2/3 +
penultimate), so the layer-sweep can measure where each of the 7 axes is best
(approach (c): don't assume penultimate; let data decide single vs multi-layer).

Design for pod:
  - data root and weights are ARGS, never hardcoded (fixes the mount-path
    mismatch pain: pass --data-root to match wherever the volume mounted)
  - minimal deps: torch + torchvision only (no robustbench/autoattack, which
    caused the dependency conflicts; not needed for extraction)
  - hooks capture activations from named modules; pooled to (B, C) per layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_backbone(arch="cifar10_resnet20", device="cuda"):
    """Load a CIFAR-native ResNet. Falls back through a couple of sources.
    Returns (model, info). No robustbench (dependency conflicts) by default."""
    model, src = None, None
    try:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", arch, pretrained=True)
        src = f"chenyaofo {arch}"
    except Exception as e:
        print("hub load failed:", repr(e)[:100])
    if model is None:
        raise RuntimeError("could not load backbone; check network / arch name")
    model.eval().to(device)
    return model, src


class MultiLayerHooks:
    """Hook several layers at once. For a CIFAR ResNet (stem -> layer1/2/3 -> fc),
    we capture the OUTPUT of each residual stage plus the penultimate feature
    (input to the final Linear). Each captured map is global-avg-pooled to (B, C)."""

    def __init__(self, model):
        self.model = model
        self.acts = {}
        self.handles = []
        self.layer_names = []
        self._register(model)

    def _register(self, model):
        # residual stages: modules named layer1/layer2/layer3 (torchvision-style)
        for name, mod in model.named_children():
            if name in ("layer1", "layer2", "layer3", "layer4"):
                self.handles.append(
                    mod.register_forward_hook(self._make_hook(name)))
                self.layer_names.append(name)
        # penultimate feature = input to the final Linear
        final_fc = None
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                final_fc = mod
        if final_fc is not None:
            def _penult_hook(_, inp, __):
                self.acts["penult"] = inp[0].detach()
            self.handles.append(final_fc.register_forward_hook(_penult_hook))
            self.layer_names.append("penult")

    def _make_hook(self, name):
        def hook(_, __, output):
            # output is a feature map (B, C, H, W); global-avg-pool -> (B, C)
            if output.dim() == 4:
                self.acts[name] = F.adaptive_avg_pool2d(output, 1).flatten(1).detach()
            else:
                self.acts[name] = output.detach()
        return hook

    def forward(self, x):
        """Run model, return {layer_name: (B, C) activations} for all hooked layers."""
        self.acts = {}
        with torch.no_grad():
            logits = self.model(x)
        feats = {k: v.float() for k, v in self.acts.items()}
        return feats, logits.float()

    def close(self):
        for h in self.handles:
            h.remove()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cifar10_resnet20")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, src = load_backbone(args.arch, args.device)
    print("MODEL:", src, "| device:", args.device)
    hooks = MultiLayerHooks(model)
    print("hooked layers:", hooks.layer_names)

    # probe shapes with a dummy CIFAR-sized input
    x = torch.randn(4, 3, 32, 32).to(args.device)
    feats, logits = hooks.forward(x)
    print("layer feature dims:")
    for name in hooks.layer_names:
        print(f"  {name:8s}: {tuple(feats[name].shape)}")
    print("logits:", tuple(logits.shape))
    hooks.close()
