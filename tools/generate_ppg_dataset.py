#!/usr/bin/env python3
"""Generate MAX30100-like IR/RED PPG and labeled systolic-peak datasets.

Output: one .npz containing independent train/val/test CHW windows.
  <split>_raw: uint16 [N,2,256], IR channel 0, RED channel 1
  <split>_q: int8 [N,2,256], integer-preprocessed Noodle2 inputs
  <split>_y: float32 [N,1,256], Gaussian peak target heatmaps
  <split>_mask: float32 [N,1,256], ignored/boundary regions are zero
  <split>_peaks: int16 [N,12], padded by -1, reliable reference peaks
  <split>_bpm: float32 [N], generator's nominal heart rate (not labels)
  <split>_artifact: uint8 [N], whether a major disturbance was introduced

Quick start:
  python generate_ppg_dataset.py --train 6000 --val 1000 --test 1000
  python generate_ppg_dataset.py --train 64 --val 16 --test 16 --preview
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

# These integer preprocessing constants must be mirrored in Noodle2 firmware.
WINDOW = 256
INPUT_SCALE = 1.0 / 32.0
MAD_FLOOR = 16


def raw_to_int8(raw):
    """Accept shape (2,256) uint16/raw integers, return (2,256) int8 CHW."""
    x = np.asarray(raw, dtype=np.int64)
    if x.shape != (2, WINDOW):
        raise ValueError(f"expected (2,{WINDOW}), got {x.shape}")
    if np.any((x < 0) | (x > 65535)):
        raise ValueError("MAX30100 raw counts must be unsigned 16-bit")
    mean = x.sum(axis=1, keepdims=True) // WINDOW
    delta = x - mean
    mad = np.abs(delta).sum(axis=1, keepdims=True) // WINDOW
    den = np.maximum(mad, MAD_FLOOR)
    num = delta * 32
    # Exact division, ties away from zero; no float or NumPy banker rounding.
    mag = (2 * np.abs(num) + den) // (2 * den)
    q = np.sign(num) * mag
    return np.clip(q, -128, 127).astype(np.int8)


def int8_to_model_float(q):
    """PyTorch input, equal to dequantized signed INT8 with fixed 1/32 scale."""
    return np.asarray(q, dtype=np.int8).astype(np.float32) * INPUT_SCALE


def quality_summary(raw):
    x = np.asarray(raw, dtype=np.int64)
    mean = x.mean(axis=1)
    mad = np.abs(x - x.mean(axis=1, keepdims=True)).mean(axis=1)
    # A screening cue, not a ground-truth label / medical quality score.
    usable = bool((mean[0] > 1000) and (mean[1] > 1000) and
                  (mad[0] >= 16) and (mad[1] >= 16))
    return {"mean_ir": float(mean[0]), "mean_red": float(mean[1]),
            "mad_ir": float(mad[0]), "mad_red": float(mad[1]),
            "screen_usable": usable}


# -------------------------------------------------------------
# Physiology-inspired two-channel PPG waveform generator.
# -------------------------------------------------------------
from dataclasses import dataclass
import numpy as np


FS = 100
RECEPTIVE_RADIUS = 13  # 9+6+6+6 -> 27-sample receptive field


@dataclass
class SyntheticWindow:
    raw: np.ndarray     # uint16 [2,256]
    q: np.ndarray       # int8 [2,256]
    y: np.ndarray       # float32 [256] Gaussian targets
    mask: np.ndarray    # float32 [256], 0 where artifact/boundary invalid
    peaks: np.ndarray   # true local sample indices, in this window
    bpm: float
    artifact: bool


def _pulse(t, onset, rise, tail, notch_strength, reflect_delay):
    u = (t - onset) / rise
    # Asymmetric pulse, rapid rise and slower fall; zero before foot.
    primary = np.zeros_like(t)
    ok = (u > 0) & (u < 14)
    v = u[ok]
    primary[ok] = v*v * np.exp(2.0*(1.0-v))
    # Dicrotic notch and reflected secondary hump: not a second beat.
    notch = -notch_strength * np.exp(-0.5*((t-onset-rise-0.115)/0.033)**2)
    secondary = (0.07+0.10*notch_strength) * np.exp(
        -0.5*((t-onset-rise-reflect_delay)/tail)**2)
    return primary + (notch + secondary) * (t >= onset+rise)


def generate_stream(rng, seconds=5.2, fs=FS, clean=False, no_contact=False, no_pulse=False):
    """Generate raw physical-looking recordings plus latent clean peak indices.

    clean=True disables motion for reproducible diagrams/validation checks.
    no_contact=True produces low/no-light sensor readings with NO labels.
    """
    n = int(round(seconds*fs))
    t = np.arange(n, dtype=np.float64)/fs
    if no_contact:
        raw = rng.integers(0, 7, size=(2,n), dtype=np.uint16)
        return raw, np.empty(0, np.int32), np.ones(n, dtype=np.float32), 0.0, False

    bpm = float(rng.uniform(50, 130))
    rr0 = 60.0/bpm
    beats = []
    foot = float(-rng.uniform(.05, .75)*rr0)
    respiration_hz = rng.uniform(.12,.36)
    while foot < seconds + 0.2:
        beats.append(foot)
        inst_rr = rr0 * (1.0 + 0.045*np.sin(2*np.pi*respiration_hz*foot))
        inst_rr *= float(np.clip(1 + rng.normal(0,.035), 0.78,1.25))
        foot += inst_rr
    ir_dc = rng.uniform(38500,47500)
    red_dc = rng.uniform(22000,29200)
    # Real session01 baseline slowly settled downward over tens of seconds.
    drift_ir = rng.uniform(-95,55)
    drift_red = drift_ir * rng.uniform(.45,1.10)
    baseline = np.vstack([
        ir_dc + drift_ir*t + rng.uniform(40,160)*np.sin(2*np.pi*respiration_hz*t + rng.uniform(-3,3)),
        red_dc + drift_red*t + rng.uniform(30,125)*np.sin(2*np.pi*respiration_hz*t + rng.uniform(-3,3)),
    ])
    # Wide domain: clean, low-SNR and nearly lost pulses (seen in the real data).
    low_snr = (rng.random() < .30) and not clean
    a_ir = rng.uniform(18,155) if low_snr else rng.uniform(100,1300)
    a_red = a_ir * rng.uniform(.52,1.25)
    signal = baseline.copy()
    # Smooth random contact-pressure drift; especially useful as a hard negative.
    knots = np.linspace(0,seconds,max(5,int(seconds*2)))
    colored = np.interp(t,knots,rng.normal(0,rng.uniform(15,150),len(knots)))
    signal[0]+=colored
    signal[1]+=colored*rng.uniform(.5,1.1)
    peak_samples = []
    for onset in ([] if no_pulse else beats):
        rise = rng.uniform(.14,.245)
        amp_jitter = rng.uniform(.8,1.23)
        notch = rng.uniform(.02,.22)
        ref = rng.uniform(.18,.31)
        # IR and RED share pulse events but have modest shape/delay differences.
        p_ir = _pulse(t, onset, rise, rng.uniform(.07,.12),notch,ref)
        red_delay = rng.uniform(-.008,.025)
        p_red = _pulse(t, onset+red_delay, rise*rng.uniform(.95,1.10),
                       rng.uniform(.07,.13), notch*rng.uniform(.8,1.3),ref)
        signal[0] += a_ir * amp_jitter * p_ir
        signal[1] += a_red * amp_jitter * p_red
        ideal = int(round((onset+rise)*fs))
        if 0 <= ideal < n:
            # If pulse overlaps its predecessor, its true clean IR maximum
            # can shift by a few samples. Find max in a narrow local range.
            lo=max(0,ideal-6); hi=min(n,ideal+7)
            peak_samples.append(int(lo + np.argmax(signal[0,lo:hi])))

    shared = rng.normal(0,rng.uniform(4,25),n)
    signal[0] += shared+rng.normal(0,rng.uniform(6,27),n)
    signal[1] += shared*rng.uniform(.5,1.3)+rng.normal(0,rng.uniform(7,32),n)
    # Slowly varying contact pressure and optional 50-Hz mains alias.
    signal += rng.uniform(2,22)*np.sin(2*np.pi*rng.uniform(.5,2.2)*t+rng.uniform(0,6))
    signal += rng.uniform(0,10)*np.sin(2*np.pi*rng.uniform(48,52)*t)

    mask = np.ones(n,dtype=np.float32)
    artifact = False
    if not clean and rng.random() < .38:
        artifact=True
        count = int(rng.integers(1,4))
        for _ in range(count):
            start = int(rng.integers(0,max(1,n-25)))
            size = int(rng.integers(9,58))
            stop = min(n,start+size)
            seg=stop-start
            shock = rng.normal(0,rng.uniform(180,1400))
            signal[:,start:stop] += shock*np.exp(-np.arange(seg)/rng.uniform(5,32))
            signal[0,start:stop] += rng.normal(0, rng.uniform(80,650),seg)
            signal[1,start:stop] += rng.normal(0, rng.uniform(80,650),seg)
            # Do not call peaks 'wrong' in severely corrupted regions.
            mask[max(0,start-8):min(n,stop+8)] = 0
    # Small fraction of intermittent low-light/contact-loss samples.
    if not clean and rng.random() < .08:
        artifact=True
        start=int(rng.integers(0,max(1,n-30)))
        stop=min(n,start+int(rng.integers(12,65)))
        signal[:,start:stop] = rng.integers(0,10,size=(2,stop-start))
        mask[max(0,start-8):min(n,stop+8)] = 0

    raw = np.clip(np.rint(signal),0,65535).astype(np.uint16)
    return raw, np.asarray(sorted(set(peak_samples)),dtype=np.int32),mask,(0.0 if no_pulse else bpm),artifact


def make_window(rng, clean=False, no_contact_prob=.07):
    nc = rng.random() < no_contact_prob
    no_pulse = (not clean and not nc and rng.random()<.12)
    raw, peaks,mask,bpm,artifact = generate_stream(rng,5.3,FS,clean=clean,no_contact=nc,
                                                   no_pulse=no_pulse)
    start=int(rng.integers(0,raw.shape[1]-WINDOW+1))
    end=start+WINDOW
    p = peaks[(peaks >=start)&(peaks<end)]-start
    y=np.zeros(WINDOW,dtype=np.float32)
    x=np.arange(WINDOW)
    for peak in p:
        y=np.maximum(y,np.exp(-.5*((x-int(peak))/2.5)**2).astype(np.float32))
    validity=mask[start:end].copy()
    validity[:RECEPTIVE_RADIUS]=0
    validity[-RECEPTIVE_RADIUS:]=0
    # The label is not available for peaks masked by invalid motion intervals.
    p=p[validity[p] > 0]
    raw_win=raw[:,start:end].copy()
    return SyntheticWindow(raw_win,raw_to_int8(raw_win),y,validity,p,bpm,artifact)


def make_dataset(count, seed, clean=False):
    rng=np.random.default_rng(seed)
    q=np.empty((count,2,WINDOW),np.int8)
    y=np.empty((count,1,WINDOW),np.float32)
    mask=np.empty((count,1,WINDOW),np.float32)
    peaks=[]
    for i in range(count):
        window=make_window(rng,clean=clean)
        q[i]=window.q
        y[i,0]=window.y
        mask[i,0]=window.mask
        peaks.append(window.peaks)
    return q,y,mask,peaks


MAX_PEAKS = 12


def build_split(n, seed, split_name):
    rng = np.random.default_rng(int(seed))
    raw = np.empty((n, 2, WINDOW), dtype=np.uint16)
    q = np.empty((n, 2, WINDOW), dtype=np.int8)
    y = np.empty((n, 1, WINDOW), dtype=np.float32)
    mask = np.empty((n, 1, WINDOW), dtype=np.float32)
    peaks = np.full((n, MAX_PEAKS), -1, dtype=np.int16)
    bpm = np.empty((n,), dtype=np.float32)
    artifact = np.empty((n,), dtype=np.uint8)
    for i in range(n):
        w = make_window(rng)
        if len(w.peaks) > MAX_PEAKS:
            raise RuntimeError(f"more than {MAX_PEAKS} peaks in one window")
        raw[i] = w.raw
        q[i] = w.q
        y[i, 0] = w.y
        mask[i, 0] = w.mask
        peaks[i, :len(w.peaks)] = w.peaks
        bpm[i] = w.bpm
        artifact[i] = int(w.artifact)
    print(f"{split_name}: {n} windows, {int((peaks >= 0).sum())} labeled peaks, "
          f"{int(artifact.sum())} artifact windows", flush=True)
    return {f"{split_name}_{key}":value for key,value in
            dict(raw=raw, q=q, y=y, mask=mask, peaks=peaks,
                 bpm=bpm, artifact=artifact).items()}


def plot_preview(ds, path, split='train', index=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw=ds[f'{split}_raw'][index].astype(np.float32)
    q=ds[f'{split}_q'][index].astype(np.float32)
    y=ds[f'{split}_y'][index,0]
    good=ds[f'{split}_mask'][index,0]>0
    peaks=ds[f'{split}_peaks'][index]
    peaks=peaks[peaks>=0]
    t=np.arange(WINDOW)/FS
    fig,ax=plt.subplots(3,1,figsize=(11,7),sharex=True)
    ax[0].plot(t,raw[0],label='IR raw uint16');ax[0].plot(t,raw[1],label='Red raw uint16')
    ax[0].legend(loc='upper right');ax[0].set_ylabel('counts')
    ax[1].plot(t,q[0],label='IR int8');ax[1].plot(t,q[1],label='Red int8')
    ax[1].legend(loc='upper right');ax[1].set_ylabel('quantized')
    ax[2].plot(t,y,label='peak target');ax[2].plot(t,good.astype(float)*0.2,label='validity x0.2')
    for p in peaks:ax[2].axvline(float(p)/FS,alpha=0.45,color='tab:green')
    ax[2].legend(loc='upper right');ax[2].set_ylabel('target');ax[2].set_xlabel('time (s)')
    fig.suptitle(f'MAX30100-like PPG | split={split} index={index} | CHW IR, RED')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=int,default=6000)
    p.add_argument('--val',type=int,default=1000)
    p.add_argument('--test',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261108)
    p.add_argument('--output',default='ppg_synthetic_100hz_256.npz')
    p.add_argument('--preview',action='store_true',help='Save PNG of a generated window (matplotlib optional)')
    a=p.parse_args()
    if min(a.train,a.val,a.test)<=0: p.error('All split sizes must be > 0')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    ds={}
    for split,n,offset in [('train',a.train,101),('val',a.val,200001),('test',a.test,400001)]:
        ds.update(build_split(n,a.seed+offset,split))
    meta=dict(description='Synthetic MAX30100-like PPG for engineering experiments (not clinical data)',
       fs_hz=FS,window=WINDOW,seconds=WINDOW/FS,channels=['ir','red'],order='N,C,T (CHW per window)',
       acquisition_dtype='uint16',model_dtype='int8',input_scale=INPUT_SCALE,
       preprocessing='per-channel integer mean floor, integer MAD floor 16, symmetric round-half-away-from-zero, times32',
       labels='synthetic clean IR systolic local maxima; heatmap sigma 2.5 samples; invalid motion and edges masked',
       seed=a.seed,train=a.train,val=a.val,test=a.test,
       no_patient_data=True,spO2_labels=False,synthetic_only=True)
    ds['metadata_json']=np.array(json.dumps(meta,sort_keys=True))
    np.savez_compressed(output,**ds)
    print('Dataset:',output.resolve(),'bytes:',output.stat().st_size,flush=True)
    if a.preview:
        png=output.with_name(output.stem+'_preview.png')
        plot_preview(ds,png)
        print('Preview:',png.resolve(),flush=True)


if __name__=='__main__':
    main()
