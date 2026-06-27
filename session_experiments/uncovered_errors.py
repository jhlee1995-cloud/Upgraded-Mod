"""uncovered_errors.py -- three under-explored error types, each swept over a RANGE.
[1] ADVERSARIAL FGSM eps sweep  [2] PARTIAL CORRUPTION area sweep  [3] DISTRIBUTION SHIFT skew sweep

KEY RESULT: adversarial caught at every eps (batch AUC 1.00), partial caught down to 10% area,
but DISTRIBUTION SHIFT is flat ~0.5 across the full skew range = structural blind spot. This
motivated the third gate (density acceleration). Individuals stay normal (correct cluster),
only the set-level density changes -> point-wise axes are blind by design.

Usage: python session_experiments/uncovered_errors.py --volume /workspace
"""
import argparse, sys
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

def batch_axes(X, ref, centers, bs=64, nb=30):
    rows=[]
    for b in range(min(nb, len(X)//bs)):
        Xb=X[b*bs:(b+1)*bs]
        row=[SINGLE_BATCH_AXES[a](Xb,ref).mean() for a in SINGLE_BATCH_AXES]
        d=np.linalg.norm(Xb[:,None]-centers[None],axis=2); dsf=np.sort(d,axis=1)
        row.append((dsf[:,1]-dsf[:,0]).mean())
        rows.append(row)
    return np.array(rows)

def main(args):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model,src=load_backbone("cifar10_resnet20",dev); model.eval()
    hooks=MultiLayerHooks(model)
    ds=torchvision.datasets.CIFAR10(f"{args.volume}/datasets",train=False,download=True,transform=tf())
    loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False)
    feats,labs,preds,confs,imgs=[],[],[],[],[]
    with torch.no_grad():
        for x,y in loader:
            f,lg=hooks.forward(x.to(dev))
            feats.append(f["penult"].cpu().numpy())
            p=F.softmax(lg,1).cpu().numpy()
            preds.append(p.argmax(1)); confs.append(p.max(1)); labs.append(y.numpy()); imgs.append(x.numpy())
    Fc=np.vstack(feats); lab=np.concatenate(labs); pred=np.concatenate(preds)
    conf=np.concatenate(confs); IMG=np.vstack(imgs)
    ref=AxisRef(Fc,lab,n_classes=10)
    nc=10; centers=np.array([Fc[lab==c].mean(0) for c in range(nc)])
    AXES=list(SINGLE_BATCH_AXES)+["MARGIN"]
    clean_b=batch_axes(Fc[pred==lab], ref, centers)

    def feats_of(imgs_np):
        out=[]; xs=torch.tensor(imgs_np)
        with torch.no_grad():
            for i in range(0,len(xs),128):
                f,_=hooks.forward(xs[i:i+128].to(dev)); out.append(f["penult"].cpu().numpy())
        return np.vstack(out)

    # [1] ADVERSARIAL FGSM via direct model grad
    print("="*72); print("[1] ADVERSARIAL FGSM -- epsilon sweep"); print("="*72)
    corr_idx=np.where(pred==lab)[0][:2000]
    x0=torch.tensor(IMG[corr_idx]).to(dev).requires_grad_(True)
    y0=torch.tensor(lab[corr_idx]).to(dev)
    logits=model(x0); loss=F.cross_entropy(logits,y0)
    model.zero_grad(); loss.backward()
    grad_sign=x0.grad.detach().sign().cpu().numpy()
    print(f"  {'eps/255':>8s} " + " ".join(f"{a[:7]:>8s}" for a in AXES) + "   flip")
    for eps255 in [0.5,1,2,4,8,16]:
        eps=eps255/255.0/np.mean(STD)
        adv=(IMG[corr_idx]+eps*grad_sign).astype(np.float32)
        Fa=feats_of(adv)
        with torch.no_grad(): lg2=model(torch.tensor(adv).to(dev))
        flip=(lg2.argmax(1).cpu().numpy()!=lab[corr_idx]).mean()
        adv_b=batch_axes(Fa, ref, centers)
        aucs=[dir_auc(clean_b[:,i],adv_b[:,i]) for i in range(len(AXES))]
        print(f"  {eps255:8.1f} " + " ".join(f"{a:8.2f}" for a in aucs) + f"   {flip:.0%}")

    # [2] PARTIAL CORRUPTION
    print("\n"+"="*72); print("[2] PARTIAL CORRUPTION -- area-fraction sweep"); print("="*72)
    base=IMG[corr_idx].copy(); H=base.shape[2]
    print(f"  {'area':>8s} " + " ".join(f"{a[:7]:>8s}" for a in AXES))
    for frac in [0.0,0.1,0.25,0.5,0.75,1.0]:
        side=int(round(H*np.sqrt(frac))); cor=base.copy()
        if side>0:
            o=(H-side)//2
            cor[:,:,o:o+side,o:o+side]+=np.random.randn(*cor[:,:,o:o+side,o:o+side].shape).astype(np.float32)*0.5
        Fp=feats_of(cor.astype(np.float32)); part_b=batch_axes(Fp, ref, centers)
        aucs=[dir_auc(clean_b[:,i],part_b[:,i]) for i in range(len(AXES))]
        print(f"  {frac:8.2f} " + " ".join(f"{a:8.2f}" for a in aucs))

    # [3] DISTRIBUTION SHIFT
    print("\n"+"="*72); print("[3] DISTRIBUTION SHIFT -- class-imbalance sweep (blind-spot test)"); print("="*72)
    print(f"  {'skew':>8s} " + " ".join(f"{a[:7]:>8s}" for a in AXES) + "   (0=uniform,1=single)")
    cm=pred==lab
    for skew in [0.0,0.25,0.5,0.75,1.0]:
        rows=[]
        for b in range(30):
            tc=b%10; bs=64; nt=int(bs*(0.1+0.9*skew))
            pt=np.where(cm&(lab==tc))[0]; po=np.where(cm&(lab!=tc))[0]
            pick=np.r_[np.random.choice(pt,nt),np.random.choice(po,bs-nt)]
            Xb=Fc[pick]
            row=[SINGLE_BATCH_AXES[a](Xb,ref).mean() for a in SINGLE_BATCH_AXES]
            d=np.linalg.norm(Xb[:,None]-centers[None],axis=2); dsf=np.sort(d,axis=1)
            row.append((dsf[:,1]-dsf[:,0]).mean()); rows.append(row)
        sb=np.array(rows)
        aucs=[dir_auc(clean_b[:,i],sb[:,i]) for i in range(len(AXES))]
        print(f"  {skew:8.2f} " + " ".join(f"{a:8.2f}" for a in aucs))
    print("  -> FLAT ~0.5 = structural BLIND SPOT (individuals normal, axes blind)")
    print("\n"+"="*72); print("DONE"); print("="*72)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--volume",default="/workspace"); main(ap.parse_args())
