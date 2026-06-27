"""imagenet_density.py -- density-acceleration with SEMANTIC classes (the MEANING test).
Spike a class to 40% of the batch at t>=10, measure peak acceleration of the predicted-class
distribution. Safety-critical (ambulance/fire_engine/police_van) vs mundane (cat/dog/mug).

KEY RESULT: all spikes 5-6x stable (ambulance 0.859 vs stable 0.141); critical (0.781) ~ mundane
(0.688). The gate detects ABRUPTNESS, not class identity -> escalate is two-layer: (1) class-
agnostic abruptness detector + (2) policy map deciding which classes warrant human escalation.

Usage: python session_experiments/imagenet_density.py --volume /workspace --n 30000
"""
import argparse, glob, io
import numpy as np
import torch, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as T
from PIL import Image
import pyarrow.parquet as pq

prep=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

CRITICAL={'ambulance':407,'fire_engine':555,'police_van':734}
MUNDANE ={'tabby_cat':281,'golden_retriever':207,'coffee_mug':504}

def main(args):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    net=models.resnet50(weights="IMAGENET1K_V2").to(dev).eval()
    print("loading images + predicting classes for stream sampling...")
    files=sorted(glob.glob(f"{args.volume}/datasets/imagenet_val/data/*.parquet"))
    all_imgs,all_pred,bi=[],[],[]; nd=0
    def flush():
        nonlocal bi
        if not bi: return
        with torch.no_grad():
            pr=F.softmax(net(torch.stack(bi).to(dev)),1).argmax(1).cpu().numpy()
        for t,p in zip(bi,pr): all_imgs.append(t); all_pred.append(int(p))
        bi=[]
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            im=batch.column('image')
            for i in range(len(im)):
                try:
                    img=Image.open(io.BytesIO(im[i].as_py()['bytes'])).convert('RGB')
                    bi.append(prep(img))
                except: continue
                if len(bi)>=64: flush(); nd+=64
            if nd>=args.n: break
        if nd>=args.n: break
    flush()
    all_pred=np.array(all_pred); pools={}
    for idx,p in enumerate(all_pred): pools.setdefault(p,[]).append(idx)
    print(f"  {len(all_imgs)} images, {len(pools)} predicted classes")

    def pred_dist_of(indices):
        with torch.no_grad():
            pr=F.softmax(net(torch.stack([all_imgs[i] for i in indices]).to(dev)),1).argmax(1).cpu().numpy()
        h=np.bincount(pr,minlength=1000).astype(float); return h/h.sum()
    base_pool=np.arange(len(all_imgs))
    def sample_batch(spike_class=None, spike_frac=0.0, bs=128):
        n_spike=int(bs*spike_frac); idx=list(np.random.choice(base_pool,bs-n_spike))
        if spike_class is not None and spike_class in pools and n_spike>0:
            idx+=list(np.random.choice(pools[spike_class],n_spike,replace=True))
        return idx
    def stream_accel(spike_class, jump_at=10, T_=20, peak_frac=0.4):
        dists=[]
        for t in range(T_):
            frac=peak_frac if (spike_class is not None and t>=jump_at) else 0.0
            dists.append(pred_dist_of(sample_batch(spike_class,frac)))
        dists=np.array(dists); vel=np.abs(np.diff(dists,axis=0)).sum(1); acc=np.abs(np.diff(vel))
        return vel.mean(), acc.max()

    print("\n"+"="*72); print("DENSITY ACCELERATION on ImageNet -- abrupt spike = escalate?"); print("="*72)
    print(f"  {'scenario':28s} {'idx':>6s} {'peak_accel':>11s}")
    v0,a0=stream_accel(None); print(f"  {'stable (no spike)':28s} {'-':>6s} {a0:>11.3f}")
    print(f"  --- safety-critical ---"); crit=[]
    for name,ci in CRITICAL.items():
        _,a=stream_accel(ci); crit.append(a); print(f"  {'spike '+name:28s} {ci:>6d} {a:>11.3f}")
    print(f"  --- mundane (control) ---"); mund=[]
    for name,ci in MUNDANE.items():
        _,a=stream_accel(ci); mund.append(a); print(f"  {'spike '+name:28s} {ci:>6d} {a:>11.3f}")
    print(f"\n  stable {a0:.3f} | critical mean {np.mean(crit):.3f} | mundane mean {np.mean(mund):.3f}")
    print(f"  -> spikes >> stable = abruptness detected; critical~mundane = class-agnostic gate")
    print(f"     (policy map, a separate layer, decides which classes escalate)")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--volume",default="/workspace")
    ap.add_argument("--n",type=int,default=30000); main(ap.parse_args())
