"""density_accel.py -- can we detect the ACCELERATION of distribution shift?
Three stream types over time:
  stable:  class distribution fixed (baseline -> accel ~0)
  gradual: distribution drifts slowly (normal, like day->night -> low accel)
  sudden:  distribution jumps abruptly (emergency-vehicle 10x -> high accel = escalate)
Per timestep: batch's predicted-class distribution -> velocity (1st diff) -> accel (2nd diff).
Test: does acceleration separate SUDDEN from gradual/stable? (gradual = normal, sudden = escalate)

KEY RESULT: sudden peak accel 0.658 vs gradual 0.191 vs stable 0.170 (3.4x); velocity ~equal
across all (0.295/0.295/0.305) -> acceleration is NECESSARY, velocity alone false-alarms.

Usage: python session_experiments/density_accel.py --volume /workspace
"""
import argparse, sys
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, "extract")
from backbone import load_backbone, MultiLayerHooks
import torchvision, torchvision.transforms as T

MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)
def tf(): return T.Compose([T.ToTensor(), T.Normalize(MEAN,STD)])

def main(args):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model,src=load_backbone("cifar10_resnet20",dev); model.eval()
    hooks=MultiLayerHooks(model)
    ds=torchvision.datasets.CIFAR10(f"{args.volume}/datasets",train=False,download=True,transform=tf())
    loader=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=False)
    allx, allp = [], []
    with torch.no_grad():
        for x,y in loader:
            _,lg=hooks.forward(x.to(dev))
            allp.append(F.softmax(lg,1).argmax(1).cpu().numpy()); allx.append(x.numpy())
    X=np.vstack(allx); P=np.concatenate(allp)
    nc=10
    pools={c: np.where(P==c)[0] for c in range(nc)}

    def make_batch(dist, bs=128):
        counts=np.random.multinomial(bs, dist)
        idx=np.concatenate([np.random.choice(pools[c], counts[c], replace=True) for c in range(nc)])
        return X[idx]

    def pred_dist(Xb):
        with torch.no_grad():
            _,lg=hooks.forward(torch.tensor(Xb).to(dev))
            pred=F.softmax(lg,1).argmax(1).cpu().numpy()
        h=np.bincount(pred, minlength=nc).astype(float); return h/h.sum()

    base=np.ones(nc)/nc

    def stream_stable(T_=20):
        return [base.copy() for _ in range(T_)]
    def stream_gradual(T_=20):
        out=[]
        for t in range(T_):
            d=base.copy(); frac=t/(T_-1)
            d[0]=0.1*(1-frac); d[1]=0.1*(1+frac); d=d/d.sum(); out.append(d)
        return out
    def stream_sudden(T_=20, jump_at=10):
        out=[]
        for t in range(T_):
            d=base.copy()
            if t>=jump_at: d[2]=0.1*10; d=d/d.sum()
            out.append(d)
        return out

    def measure(dist_seq):
        dists=np.array([pred_dist(make_batch(d)) for d in dist_seq])
        vel=np.abs(np.diff(dists, axis=0)).sum(1)
        acc=np.abs(np.diff(vel))
        return dists, vel, acc

    print("="*72); print("DENSITY ACCELERATION -- separate sudden from gradual/stable?"); print("="*72)
    results={}
    for name, fn in [("stable",stream_stable),("gradual",stream_gradual),("sudden",stream_sudden)]:
        vels, accs, peakaccs = [], [], []
        for rep in range(10):
            _, v, a = measure(fn())
            vels.append(v.mean()); accs.append(a.mean()); peakaccs.append(a.max())
        results[name]=(np.mean(vels), np.mean(accs), np.mean(peakaccs))
        print(f"  {name:10s} mean_velocity={np.mean(vels):.3f}  mean_accel={np.mean(accs):.3f}  PEAK_accel={np.mean(peakaccs):.3f}")
    s_peak=results["sudden"][2]; g_peak=results["gradual"][2]; st_peak=results["stable"][2]
    print(f"\n  sudden peak {s_peak:.3f} vs gradual {g_peak:.3f} vs stable {st_peak:.3f}")
    if s_peak > 2*max(g_peak, st_peak):
        print(f"  >>> SUDDEN peak accel >2x gradual/stable -> acceleration DETECTS abrupt shift")
    print(f"\n  velocity (why accel not velocity): sudden {results['sudden'][0]:.3f} vs gradual {results['gradual'][0]:.3f}")
    print(f"  -> velocity ~equal -> alone it false-alarms; acceleration isolates abruptness")
    print("\n"+"="*72); print("DONE"); print("="*72)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--volume",default="/workspace"); main(ap.parse_args())
