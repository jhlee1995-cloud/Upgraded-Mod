"""imagenet_valley_check.py -- is the ImageNet valley separation 1.14 a sample-count artifact?
Re-extract more samples, sweep min-samples-per-class threshold, recompute separation.

KEY RESULT: sep_ratio flat at 1.13-1.14 across min/class 3->10->20->30 -> NOT an artifact;
valleys are genuinely shallow at ImageNet scale (1000 classes packed in 2048d, clusters overlap).

Usage: python session_experiments/imagenet_valley_check.py --volume /workspace --n 30000
"""
import argparse, glob, io, os
import numpy as np
import torch
import torchvision.models as models, torchvision.transforms as T
from PIL import Image
import pyarrow.parquet as pq

prep=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def main(args):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    net=models.resnet50(weights="IMAGENET1K_V2").to(dev).eval()
    penult={}
    net.avgpool.register_forward_hook(lambda m,i,o: penult.__setitem__('z',o.squeeze(-1).squeeze(-1)))
    cache=f"{args.volume}/cache/imagenet"; os.makedirs(cache,exist_ok=True)
    fp=f"{cache}/penult_{args.n}.npz"
    if os.path.exists(fp):
        d=np.load(fp); Fc,lab=d['F'],d['lab']; print(f"loaded {len(Fc)} cached")
    else:
        files=sorted(glob.glob(f"{args.volume}/datasets/imagenet_val/data/*.parquet"))
        feats,labs,bi,bl=[],[],[],[]; nd=0
        def flush():
            nonlocal feats,labs,bi,bl
            if not bi: return
            with torch.no_grad(): net(torch.stack(bi).to(dev)); feats.append(penult['z'].cpu().numpy())
            labs.extend(bl); bi,bl=[],[]
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
                im=batch.column('image'); lb=batch.column('label')
                for i in range(len(lb)):
                    try:
                        img=Image.open(io.BytesIO(im[i].as_py()['bytes'])).convert('RGB')
                        bi.append(prep(img)); bl.append(lb[i].as_py())
                    except: continue
                    if len(bi)>=64: flush(); nd+=64
                if nd>=args.n: break
            if nd>=args.n: break
        flush(); Fc=np.vstack(feats); lab=np.array(labs[:len(Fc)])
        np.savez(fp,F=Fc,lab=lab,pred=np.zeros(len(Fc)),conf=np.zeros(len(Fc)))
        print(f"extracted {len(Fc)}")

    print(f"\n{len(Fc)} samples, {len(np.unique(lab))} classes, avg {len(Fc)/len(np.unique(lab)):.1f}/class\n")
    print(f"{'min/class':>10s} {'#classes':>9s} {'avg n':>7s} {'sep_ratio':>10s}")
    for minn in [3,10,20,30,40]:
        cls=[c for c in np.unique(lab) if (lab==c).sum()>=minn]
        if len(cls)<2: continue
        cents=np.array([Fc[lab==c].mean(0) for c in cls])
        within=np.array([np.linalg.norm(Fc[lab==c]-cents[i],axis=1).mean() for i,c in enumerate(cls)])
        bw=np.linalg.norm(cents[:,None]-cents[None],axis=2); np.fill_diagonal(bw,np.nan)
        sr=np.nanmean(bw)/(within.mean()+1e-9)
        avgn=np.mean([(lab==c).sum() for c in cls])
        print(f"{minn:>10d} {len(cls):>9d} {avgn:>7.1f} {sr:>10.2f}")
    print(f"\n  -> rising = sample artifact; flat ~1.3 = genuinely shallow at ImageNet scale")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--volume",default="/workspace")
    ap.add_argument("--n",type=int,default=30000); main(ap.parse_args())
