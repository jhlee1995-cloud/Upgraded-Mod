"""mahalanobis_typeb.py -- two experiments in one run:
  (A) MAHALANOBIS variant of CLUSTER_DISTANCE (clusters are elongated, aniso 7.4).
  (B) TYPE-B at batch=32 (786 wrong / 32 = ~24 batches) to look for structure-axis split.

KEY RESULT: Mahalanobis helps fog/contrast (contrast 0.51->0.65, fog 0.51->0.62) where CLD was
weakest, but hurts elsewhere and is worse on type-b -> targeted variant, not blanket. Type-b at
batch 32 still all axes 1.00 (batch ceiling); per-sample analysis (sample_and_scale.py) needed.

Usage: python session_experiments/mahalanobis_typeb.py --volume /workspace
"""
import argparse, os, sys
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, "extract")
from backbone import load_backbone, MultiLayerHooks
from axis_registry import AxisRef, SINGLE_BATCH_AXES
from sklearn.metrics import roc_auc_score
import torchvision, torchvision.transforms as T

MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)
def tf(): return T.Compose([T.ToTensor(), T.Normalize(MEAN,STD)])
def dir_auc(a,b):
    y=np.r_[np.zeros(len(a)),np.ones(len(b))]; s=np.r_[a,b]
    if len(np.unique(s))<2: return 0.5
    return max(roc_auc_score(y,s),1-roc_auc_score(y,s))

def main(args):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model,src=load_backbone("cifar10_resnet20",dev); model.eval()
    hooks=MultiLayerHooks(model)
    root=f"{args.volume}/datasets"
    ds=torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf())
    loader=torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
    feats,labs,preds,confs=[],[],[],[]
    with torch.no_grad():
        for x,y in loader:
            f,lg=hooks.forward(x.to(dev))
            feats.append(f["penult"].cpu().numpy())
            p=F.softmax(lg,1).cpu().numpy()
            preds.append(p.argmax(1)); confs.append(p.max(1)); labs.append(y.numpy())
    F_=np.vstack(feats); lab=np.concatenate(labs); pred=np.concatenate(preds); conf=np.concatenate(confs)
    ref=AxisRef(F_, lab, n_classes=10)
    nc=10; centers=np.array([F_[lab==c].mean(0) for c in range(nc)])
    cov=sum(((F_[lab==c]-centers[c]).T@(F_[lab==c]-centers[c])) for c in range(nc))/len(F_)
    inv=np.linalg.inv(cov+np.eye(F_.shape[1])*1e-3)

    def maha(X):
        d=np.zeros((len(X),nc))
        for c in range(nc):
            diff=X-centers[c]; d[:,c]=np.sqrt(np.einsum('ij,jk,ik->i',diff,inv,diff))
        return d.min(1)
    def eucl(X): return ref.per_subnet_nearest(X).mean(1)

    print("="*72); print("[A] MAHALANOBIS vs EUCLIDEAN (severity 1)"); print("="*72)
    def bmeans(Xp, fn, nb=40, bs=50): return np.array([fn(Xp[i*bs:(i+1)*bs]).mean() for i in range(nb)])
    clean_e=bmeans(F_[:2000], eucl); clean_m=bmeans(F_[:2000], maha)
    print(f"  {'corruption':16s} {'EUCL':>7s} {'MAHA':>7s} {'delta':>7s}")
    c10c=f"{root}/cifar10c"
    for corr in ["contrast","fog","defocus_blur","brightness","gaussian_noise"]:
        path=f"{c10c}/{corr}.npy"
        if not os.path.exists(path): continue
        sev1=np.load(path)[:10000]
        xs=torch.stack([tf()(im) for im in sev1[:2000]])
        ff=[]
        for i in range(0,len(xs),128):
            with torch.no_grad(): f,_=hooks.forward(xs[i:i+128].to(dev))
            ff.append(f["penult"].cpu().numpy())
        cX=np.vstack(ff)
        ae=dir_auc(clean_e, bmeans(cX, eucl)); am=dir_auc(clean_m, bmeans(cX, maha))
        print(f"  {corr:16s} {ae:7.2f} {am:7.2f} {am-ae:+7.2f}")

    print("\n"+"="*72); print("[B] TYPE-B at batch=32"); print("="*72)
    wrong=pred!=lab; cw=wrong&(conf>0.7); correct=pred==lab
    print(f"  {wrong.sum()} wrong, {cw.sum()} conf-wrong, {correct.sum()} correct")
    def axis_batches(mask, bs=32, maxb=30):
        idx=np.where(mask)[0]; np.random.shuffle(idx); rows=[]
        for b in range(min(maxb, len(idx)//bs)):
            Xb=F_[idx[b*bs:(b+1)*bs]]
            row=[SINGLE_BATCH_AXES[a](Xb, ref).mean() for a in SINGLE_BATCH_AXES]
            row.append(maha(Xb).mean()); rows.append(row)
        return np.array(rows)
    cor=axis_batches(correct); cwb=axis_batches(cw)
    names=["DEVIATION","CONSENSUS","CLUSTER_DIST","SUBNET_CONS","MAHA_DIST"]
    print(f"  correct {len(cor)} batches, conf-wrong {len(cwb)} batches")
    aucs={}
    for i,nm in enumerate(names):
        a=dir_auc(cor[:,i], cwb[:,i]); aucs[nm]=a
        print(f"  {nm:14s} AUC {a:.2f}")
    struct=[aucs["CONSENSUS"],aucs["CLUSTER_DIST"],aucs["SUBNET_CONS"]]
    print(f"  structure-axis spread: {max(struct)-min(struct):.2f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--volume",default="/workspace"); main(ap.parse_args())
