"""sample_and_scale.py -- two checks in one run:
  (A) PER-SAMPLE type-b: AUC at the individual-sample level (no batch averaging) -- batch-mean
      hits a 1.00 ceiling; per-sample shows true difficulty and which axis is best.
  (B) MODEL SCALE: repeat on ResNet56 vs ResNet20 -- do valleys, type-b detection hold?

KEY RESULT: per-sample margin best (CIFAR 0.895), holds/strengthens at scale (ResNet56 0.905,
valley sep 2.87->4.12). CAVEAT: ResNet56 also higher accuracy -> "improves" is confounded;
solid claim is "method does not break at scale".

Usage: python session_experiments/sample_and_scale.py --volume /workspace
"""
import argparse, sys
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, "extract")
from backbone import load_backbone, MultiLayerHooks
from axis_registry import AxisRef
from sklearn.metrics import roc_auc_score
import torchvision, torchvision.transforms as T

def dir_auc(a, b):
    y=np.r_[np.zeros(len(a)),np.ones(len(b))]; s=np.r_[a,b]
    if len(np.unique(s))<2: return 0.5
    return max(roc_auc_score(y,s),1-roc_auc_score(y,s))
def tf():
    return T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])

def collect(arch, volume, dev):
    model,src=load_backbone(arch, dev)
    hooks=MultiLayerHooks(model)
    ds=torchvision.datasets.CIFAR10(f"{volume}/datasets", train=False, download=True, transform=tf())
    loader=torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    feats,labs,preds,confs=[],[],[],[]
    with torch.no_grad():
        for x,y in loader:
            f,lg=hooks.forward(x.to(dev))
            feats.append(f["penult"].cpu().numpy())
            p=F.softmax(lg,1).cpu().numpy()
            preds.append(p.argmax(1)); confs.append(p.max(1)); labs.append(y.numpy())
    return src, np.vstack(feats), np.concatenate(labs), np.concatenate(preds), np.concatenate(confs)

def analyze(arch, volume, dev):
    src,F_,lab,pred,conf=collect(arch, volume, dev)
    ref=AxisRef(F_, lab, n_classes=10); acc=(pred==lab).mean()
    print(f"\n{'='*72}\nMODEL: {src} | feat_dim {F_.shape[1]} | acc {acc:.3f}\n{'='*72}")
    nc=10; centers=np.zeros((nc,F_.shape[1])); within=np.zeros(nc)
    for c in range(nc):
        Xc=F_[lab==c]; centers[c]=Xc.mean(0); within[c]=np.linalg.norm(Xc-centers[c],axis=1).mean()
    between=np.linalg.norm(centers[:,None]-centers[None],axis=2); np.fill_diagonal(between,np.nan)
    sr=np.nanmean(between)/(within.mean()+1e-9)
    print(f"  valley separation ratio: {sr:.2f}")
    wrong=pred!=lab; cw=wrong&(conf>0.7); correct=pred==lab
    print(f"  {wrong.sum()} wrong, {cw.sum()} conf-wrong, {correct.sum()} correct")
    def cluster(X): return ref.per_subnet_nearest(X).mean(1)
    def margin(X):
        d=np.zeros((len(X),nc))
        for c in range(nc): d[:,c]=np.linalg.norm(X-centers[c],axis=1)
        ds=np.sort(d,axis=1); return ds[:,1]-ds[:,0]
    cd=dir_auc(cluster(F_[correct]), cluster(F_[cw]))
    en=dir_auc((F_[correct]**2).mean(1), (F_[cw]**2).mean(1))
    mg=dir_auc(margin(F_[correct]), margin(F_[cw]))
    print(f"  PER-SAMPLE CLUSTER_DISTANCE AUC: {cd:.3f}")
    print(f"  PER-SAMPLE energy AUC:           {en:.3f}")
    print(f"  PER-SAMPLE margin AUC:           {mg:.3f}")
    return sr, mg

def main(args):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    results={}
    for arch in ["cifar10_resnet20","cifar10_resnet56"]:
        try:
            sr,mg=analyze(arch, args.volume, dev); results[arch]=(sr,mg)
        except Exception as e:
            print(f"\n{arch} failed: {repr(e)[:80]}")
    print(f"\n{'='*72}\nSCALE COMPARISON\n{'='*72}")
    print(f"  {'model':22s} {'valley_sep':>11s} {'typeb_margin_auc':>17s}")
    for a,(sr,mg) in results.items():
        print(f"  {a:22s} {sr:11.2f} {mg:17.3f}")
    print("  -> similar across scale = generalizes; diverge = note scale-fragility")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--volume",default="/workspace"); main(ap.parse_args())
