"""imagenet_extract.py -- extract ResNet50 penult (2048d) from ImageNet val parquet, then:
  [1] valley separation (do valleys exist at ImageNet scale? CIFAR was 2.87)
  [2] type-b per-sample: margin vs cluster-distance vs energy
Caches penult to /workspace/cache/imagenet/penult_N.npz.

KEY RESULT: valley sep 1.14 (shallow, vs CIFAR 2.87); margin 0.800 best (rank preserved
margin>cluster>energy). Absolute distance degrades with shallow valleys; relative margin survives.

Usage: python session_experiments/imagenet_extract.py --volume /workspace --n 10000
Data: non-gated HF mirror benjamin-paine/imagenet-1k-256x256, downloaded separately via
huggingface_hub to /workspace/datasets/imagenet_val/data/*.parquet (NOT the datasets library).
"""
import argparse, glob, io, os
import numpy as np
import torch, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as T
from PIL import Image
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

def dir_auc(a,b):
    y=np.r_[np.zeros(len(a)),np.ones(len(b))]; s=np.r_[a,b]
    if len(np.unique(s))<2: return 0.5
    return max(roc_auc_score(y,s),1-roc_auc_score(y,s))

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
        d=np.load(fp); Fc,lab,pred,conf=d['F'],d['lab'],d['pred'],d['conf']
        print(f"loaded cached {fp}")
    else:
        files=sorted(glob.glob(f"{args.volume}/datasets/imagenet_val/data/*.parquet"))
        feats,labs,preds,confs,bi,bl=[],[],[],[],[],[]; nd=0
        def flush():
            nonlocal feats,labs,preds,confs,bi,bl
            if not bi: return
            with torch.no_grad():
                lg=net(torch.stack(bi).to(dev)); z=penult['z']; p=F.softmax(lg,1)
            feats.append(z.cpu().numpy()); preds.append(p.argmax(1).cpu().numpy())
            confs.append(p.max(1).values.cpu().numpy()); labs.extend(bl); bi,bl=[],[]
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
        flush()
        Fc=np.vstack(feats); lab=np.array(labs[:len(Fc)])
        pred=np.concatenate(preds); conf=np.concatenate(confs)
        np.savez(fp,F=Fc,lab=lab,pred=pred,conf=conf)
        print(f"extracted {len(Fc)} -> {fp}")

    acc=(pred==lab).mean()
    print(f"\n{'='*72}\nImageNet val: {len(Fc)} samples, penult {Fc.shape[1]}d, acc {acc:.3f}\n{'='*72}")
    cls=np.unique(lab); centers={c:Fc[lab==c].mean(0) for c in cls if (lab==c).sum()>=3}
    cl=np.array(list(centers)); C=np.array([centers[c] for c in cl])
    within=np.array([np.linalg.norm(Fc[lab==c]-centers[c],axis=1).mean() for c in cl])
    bw=np.linalg.norm(C[:,None]-C[None],axis=2); np.fill_diagonal(bw,np.nan)
    sr=np.nanmean(bw)/(within.mean()+1e-9)
    print(f"\n[1] VALLEY separation ratio: {sr:.2f}  (CIFAR was 2.87)")

    wrong=pred!=lab; cw=wrong&(conf>0.5); correct=pred==lab
    print(f"\n[2] TYPE-B: {wrong.sum()} wrong, {cw.sum()} conf-wrong(>0.5), {correct.sum()} correct")
    def nd_(X): return np.linalg.norm(X[:,None]-C[None],axis=2).min(1)
    def mg_(X):
        d=np.linalg.norm(X[:,None]-C[None],axis=2); ds=np.sort(d,axis=1); return ds[:,1]-ds[:,0]
    ci=np.where(correct)[0]; np.random.shuffle(ci); ci=ci[:cw.sum()]
    print(f"  per-sample CLUSTER_DISTANCE AUC: {dir_auc(nd_(Fc[ci]),nd_(Fc[cw])):.3f}")
    print(f"  per-sample MARGIN AUC:           {dir_auc(mg_(Fc[ci]),mg_(Fc[cw])):.3f}")
    print(f"  per-sample energy AUC:           {dir_auc((Fc[ci]**2).mean(1),(Fc[cw]**2).mean(1)):.3f}")
    print(f"  (CIFAR: CLUSTER 0.763, MARGIN 0.895, energy 0.757)")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--volume",default="/workspace")
    ap.add_argument("--n",type=int,default=10000); main(ap.parse_args())
