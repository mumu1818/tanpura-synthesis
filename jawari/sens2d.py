"""Thread-placement sensitivity analysis of the rendered sweep (reads runs/*.npz written by render2d.py).

    python sens2d.py [variant ...]        # default: all variants that have renders
Writes sensitivity_results.txt / .json and figures png/fig_buzz_tracks*.png, png/fig_sensitivity.png.

Per case, from the metrics.py tracks (frame hop 23 ms), smoothed with a 0.5 s moving average:
  mean_buzz      mean buzz_db over 0.5-6 s
  t_on           first time the smoothed hf_frac reaches 0.1 (jvari 'bloom' / formant-sweep onset)
  t_bmax, bmax   time and value of the maximum smoothed buzz
  jv_len         'jvari length': time from t_bmax until the smoothed buzz is 10 dB below bmax (level-independent)
  t_end15/20     last time the smoothed buzz_db is >= -15 / -20 dB (jvari persistence)
  t_cmax, cmax   time and value of the maximum smoothed spectral centroid (top of the formant sweep)
Sensitivity of a variant to thread placement (positions 3, 4, 5, 6.5, 8 mm):
  S_track  = time-average over 0.5-6 s of the across-position std of the smoothed buzz track  [dB]
  S_scalar = across-position std of each scalar above
Noise floor of the same quantities: across 4 'twins' at x_c = 4.999, 5.000, 5.001, 5.002 mm.  The contact dynamics
is chaotic (validate2d.py D), so even a 1 um change gives a different fine trajectory; a position effect is only
meaningful where it exceeds this floor.  Reported: ratio = S_positions / S_twins.
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

from common2d import HERE
from render2d import POSITIONS, TWINS, VARIANTS

RUNS = f"{HERE}/runs"
PNGS = f"{HERE}/png"
T0, T1 = 0.5, 6.0
SMOOTH_S = 0.5
POS_ORDER = ["xc3", "xc4", "xc5", "xc6p5", "xc8"]
TWIN_ORDER = ["xc5", "xc4p999", "xc5p001", "xc5p002"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def load(name, which="my"):
    path = f"{RUNS}/{name}.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path)
    t = d[f"{which}_t"]
    hop = t[1] - t[0]
    k = max(1, int(round(SMOOTH_S / hop)))
    sm = {key: uniform_filter1d(d[f"{which}_{key}"], k, mode="nearest")
          for key in ("buzz_db", "hf_frac", "centroid_hz", "frame_db")}
    sel = (t >= T0) & (t <= T1)
    out = dict(t=t, sm=sm, sel=sel)
    out["mean_buzz"] = float(np.mean(sm["buzz_db"][sel]))
    out["mean_hf"] = float(np.mean(sm["hf_frac"][sel]))
    hf = sm["hf_frac"]
    on = np.where((hf >= 0.1) & (t > 0.1))[0]
    out["t_on"] = float(t[on[0]]) if len(on) else np.nan
    for thr in (15, 20):
        idx = np.where((sm["buzz_db"] >= -thr) & (t > 0.1) & (t < t[-1] - 0.3))[0]
        out[f"t_end{thr}"] = float(t[idx[-1]]) if len(idx) else 0.0
    sb = (t > 0.1) & (t < t[-1] - 0.3)
    ib = int(np.argmax(np.where(sb, sm["buzz_db"], -1e9)))
    out["t_bmax"] = float(t[ib]); out["bmax"] = float(sm["buzz_db"][ib])
    after = np.where((t > t[ib]) & (sm["buzz_db"] < out["bmax"] - 10.0))[0]
    out["jv_len"] = float(t[after[0]] - t[ib]) if len(after) else float(t[-1] - t[ib])
    selc = (t > 0.1) & (t < 6.0)
    ic = np.argmax(np.where(selc, sm["centroid_hz"], -1))
    out["t_cmax"] = float(t[ic]); out["cmax"] = float(sm["centroid_hz"][ic])
    lev = sm["frame_db"]
    out["lev6"] = float(lev[np.argmin(np.abs(t - 6.0))])
    if "Ey" in d and d["pol2"]:
        tr = d["t_rec"]; Ey = d["Ey"]; Ez = d["Ez"]
        s1 = (tr > 0.5) & (tr < 6.0)
        out["Ez_frac"] = float(np.mean(Ez[s1] / (Ey[s1] + Ez[s1] + 1e-300)))
    return out


SCALARS = ["mean_buzz", "mean_hf", "t_on", "t_end15", "t_end20", "t_cmax", "cmax", "t_bmax", "bmax", "jv_len"]


def variant_sens(v, which="my"):
    base, suf = (v.split("@", 1)[0], "@" + v.split("@", 1)[1]) if "@" in v else (v, "")
    pos = {p: load(f"{base}_{p}{suf}", which) for p in POS_ORDER}
    tw = {p: load(f"{base}_{p}{suf}", which) for p in TWIN_ORDER}
    pos = {k: x for k, x in pos.items() if x is not None}
    tw = {k: x for k, x in tw.items() if x is not None}
    if len(pos) < 4:
        return None

    def track_std(group, t0=T0, t1=T1):
        B = np.array([g["sm"]["buzz_db"] for g in group.values()])
        t = next(iter(group.values()))["t"]
        sel = (t >= t0) & (t <= t1)
        return float(np.mean(np.std(B[:, sel], axis=0, ddof=1))), B

    Sp, Bp = track_std(pos)
    St = track_std(tw)[0] if len(tw) >= 2 else np.nan
    Sp_late = track_std(pos, 3.0, 8.0)[0]
    St_late = track_std(tw, 3.0, 8.0)[0] if len(tw) >= 2 else np.nan
    res = dict(variant=v, n_pos=len(pos), n_twin=len(tw), S_track_pos=Sp, S_track_twin=St,
               ratio_track=Sp / St if St and np.isfinite(St) else np.nan, per_pos={}, per_twin={},
               S_late_pos=Sp_late, S_late_twin=St_late)
    for key in SCALARS:
        a = np.array([pos[p][key] for p in pos]); b = np.array([tw[p][key] for p in tw]) if tw else np.array([])
        sp = float(np.nanstd(a, ddof=1)); stw = float(np.nanstd(b, ddof=1)) if len(b) >= 2 else np.nan
        res[f"S_{key}_pos"] = sp; res[f"S_{key}_twin"] = stw
        res[f"range_{key}_pos"] = float(np.nanmax(a) - np.nanmin(a))
        res[f"ratio_{key}"] = sp / stw if stw and np.isfinite(stw) and stw > 0 else np.inf
    for p in pos:
        res["per_pos"][p] = {k: pos[p][k] for k in SCALARS + ["lev6"] + (["Ez_frac"] if "Ez_frac" in pos[p] else [])}
    for p in tw:
        res["per_twin"][p] = {k: tw[p][k] for k in SCALARS}
    # pairwise distinguishability: mean |b_i - b_j| over 0.5-6 s, positions vs twins
    def pair_dist(group):
        B = np.array([g["sm"]["buzz_db"] for g in group.values()])
        sel = next(iter(group.values()))["sel"]
        ds = [np.mean(np.abs(B[i, sel] - B[j, sel])) for i in range(len(B)) for j in range(i + 1, len(B))]
        return float(np.mean(ds))
    res["pair_dist_pos"] = pair_dist(pos)
    res["pair_dist_twin"] = pair_dist(tw) if len(tw) >= 2 else np.nan
    res["_pos"] = pos; res["_tw"] = tw
    return res


def fig_tracks(results, path, which_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axs = plt.subplots(n, 1, figsize=(10, 2.3 * n + 0.6), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, r in zip(axs, results):
        for p in TWIN_ORDER[1:]:
            if p in r["_tw"]:
                g = r["_tw"][p]
                ax.plot(g["t"], g["sm"]["buzz_db"], color="#9a9893", lw=1.0, ls="--", zorder=1)
        for i, p in enumerate(POS_ORDER):
            if p in r["_pos"]:
                g = r["_pos"][p]
                ax.plot(g["t"], g["sm"]["buzz_db"], color=COLORS[i], lw=2.0, zorder=2,
                        label=f"x_c = {POSITIONS[p] * 1e3:g} mm")
        ax.set_title(f"{r['variant']}: {VARIANTS[r['variant'].split('@')[0]][1]}", fontsize=8.5, loc="left", color="#0b0b0b")
        ax.text(0.995, 0.95, f"S_track = {r['S_track_pos']:.1f} dB (positions) vs {r['S_track_twin']:.1f} dB "
                f"(1 um twins, grey dashed)", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#52514e")
        ax.axvspan(T0, T1, color="#f0efec", zorder=0)
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_ylabel("buzz dB", fontsize=8)
        ax.set_ylim(-45, 0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(loc="lower left", fontsize=7.5, ncol=5, frameon=False)
    axs[-1].set_xlabel("time (s)")
    fig.suptitle(f"Thread-placement dependence of the jvari: buzz (2-8 kHz vs <1 kHz), 0.5 s smoothed, {which_label}",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_summary(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("S_track", "buzz-track spread (dB)"), ("S_mean_buzz", "std of mean buzz 0.5-6 s (dB)"),
            ("S_t_on", "std of onset time t_on (s)"), ("S_t_end15", "std of persistence t_end(-15 dB) (s)")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    names = [r["variant"] for r in results]
    x = np.arange(len(names))
    for ax, (k, lab) in zip(axs, keys):
        pos = [r[f"{k}_pos"] if k != "S_track" else r["S_track_pos"] for r in results]
        tw = [r[f"{k}_twin"] if k != "S_track" else r["S_track_twin"] for r in results]
        ax.bar(x - 0.2, pos, 0.38, color="#2a78d6", label="across thread positions")
        ax.bar(x + 0.2, tw, 0.38, color="#9a9893", label="across 1 um twins (noise floor)")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, fontsize=8)
        ax.set_title(lab, fontsize=9, loc="left")
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_by_position(path, variants=("1d", "1dtm", "2d", "2dnf", "2dcf"), which="my"):
    """small multiples per thread position: smoothed buzz of each variant, plus the 2-D horizontal energy share."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(POS_ORDER), 2, figsize=(13, 2.2 * len(POS_ORDER) + 0.6), sharex=True,
                            gridspec_kw=dict(width_ratios=[2.2, 1]))
    for row, p in enumerate(POS_ORDER):
        ax = axs[row, 0]
        for i, v in enumerate(variants):
            g = load(f"{v}_{p}", which)
            if g is not None:
                ax.plot(g["t"], g["sm"]["buzz_db"], color=COLORS[i], lw=1.8, label=v)
        ax.set_ylim(-45, 0); ax.grid(alpha=0.3, lw=0.5); ax.set_ylabel("buzz dB", fontsize=8)
        ax.set_title(f"x_c = {POSITIONS[p] * 1e3:g} mm (thread lift {3 * (0.010 - POSITIONS[p]) ** 2 * 1e6:.0f} um)",
                     fontsize=9, loc="left")
        ax2 = axs[row, 1]
        for i, v in enumerate(variants):
            path_ = f"{RUNS}/{v}_{p}.npz"
            if v.startswith("2d") and os.path.exists(path_):
                d = np.load(path_)
                fr = d["Ez"] / (d["Ey"] + d["Ez"] + 1e-300)
                ax2.plot(d["t_rec"], uniform_filter1d(fr, 200), color=COLORS[i], lw=1.5, label=v)
        ax2.set_ylim(0, 1); ax2.grid(alpha=0.3, lw=0.5); ax2.set_ylabel("E_z / (E_y+E_z)", fontsize=8)
        for a in (ax, ax2):
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    axs[0, 0].legend(fontsize=7.5, ncol=5, frameon=False, loc="lower left")
    axs[0, 1].legend(fontsize=7.5, ncol=3, frameon=False, loc="lower right")
    axs[-1, 0].set_xlabel("time (s)"); axs[-1, 1].set_xlabel("time (s)")
    fig.suptitle("1-D vs 2-D at each thread position: smoothed buzz of the vertical bridge force (left) and share of "
                 "string energy in the horizontal polarisation (right)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt_variant(r, label):
    L = [f"\n### {r['variant']} [{label}] -- {VARIANTS[r['variant'].split('@')[0]][1]}  ({r['n_pos']} positions, {r['n_twin']} twins)"]
    L.append("  pos      mean_buzz  mean_hf   t_on   t_end15  t_end20  t_cmax   cmax   t_bmax   bmax  jv_len   lev6   Ez_frac")
    for p, d in list(r["per_pos"].items()) + [(f"{k}*", v) for k, v in r["per_twin"].items() if k != "xc5"]:
        L.append(f"  {p:8s} {d['mean_buzz']:8.1f}  {d['mean_hf']:7.3f}  {d['t_on']:6.2f}  {d['t_end15']:6.2f}  "
                 f"{d['t_end20']:6.2f}  {d['t_cmax']:6.2f}  {d['cmax']:6.0f}  {d['t_bmax']:6.2f} {d['bmax']:6.1f} {d['jv_len']:6.2f}  {d.get('lev6', np.nan):6.1f}  "
                 f"{d.get('Ez_frac', np.nan):6.3f}")
    L.append(f"  S_track (std of smoothed buzz across positions, mean 0.5-6 s): {r['S_track_pos']:.2f} dB; twins "
             f"{r['S_track_twin']:.2f} dB; ratio {r['ratio_track']:.2f}")
    L.append(f"  S_late (same, 3-8 s): positions {r['S_late_pos']:.2f} dB; twins {r['S_late_twin']:.2f} dB")
    L.append(f"  mean pairwise |buzz_i - buzz_j| (0.5-6 s): positions {r['pair_dist_pos']:.2f} dB, twins {r['pair_dist_twin']:.2f} dB")
    for key in SCALARS:
        L.append(f"  {key:9s}: std positions {r[f'S_{key}_pos']:.3f} (range {r[f'range_{key}_pos']:.3f}), "
                 f"std twins {r[f'S_{key}_twin']:.3f}, ratio {r[f'ratio_{key}']:.2f}")
    return "\n".join(L)


if __name__ == "__main__":
    vs = sys.argv[1:] or (list(VARIANTS) + ["1d@2x", "2d@2x", "1d@m40", "2d@m40", "2dd5@2x"])
    out_txt = []
    js = {}
    for which, label in (("my", "vertical bridge force"), ("mm", "mono mix F_y + 0.3 F_z")):
        res = [r for r in (variant_sens(v, which) for v in vs) if r is not None]
        if not res:
            continue
        out_txt.append(f"\n================ {label} ================")
        for r in res:
            out_txt.append(fmt_variant(r, label))
        out_txt.append("\nSUMMARY (" + label + "): variant: S_track pos/twin (ratio) | std mean_buzz pos/twin | "
                       "std t_on pos/twin | std t_end15 pos/twin | std t_end20 pos/twin | std jv_len pos/twin | S_late(3-8 s) pos/twin")
        for r in res:
            out_txt.append(f"  {r['variant']:6s}: {r['S_track_pos']:5.2f}/{r['S_track_twin']:5.2f} ({r['ratio_track']:4.1f}) | "
                           f"{r['S_mean_buzz_pos']:5.2f}/{r['S_mean_buzz_twin']:5.2f} | {r['S_t_on_pos']:5.2f}/{r['S_t_on_twin']:5.2f} | "
                           f"{r['S_t_end15_pos']:5.2f}/{r['S_t_end15_twin']:5.2f} | {r['S_t_end20_pos']:5.2f}/{r['S_t_end20_twin']:5.2f} | "
                           f"{r['S_jv_len_pos']:5.2f}/{r['S_jv_len_twin']:5.2f} | {r['S_late_pos']:5.2f}/{r['S_late_twin']:5.2f}")
        main = [r for r in res if r["variant"] in ("1d", "1dtm", "2d", "2dnf", "2dcf")]
        extra = [r for r in res if r["variant"] not in ("1d", "1dtm", "2d", "2dnf", "2dcf") and "@" not in r["variant"]]
        conv = [r for r in res if "@" in r["variant"]]
        res = [r for r in res if "@" not in r["variant"]] + [r for r in res if "@" in r["variant"]]
        tag = "" if which == "my" else "_mono"
        if main:
            fig_tracks(main, f"{PNGS}/fig_buzz_tracks{tag}.png", label)
            fig_summary(res, f"{PNGS}/fig_sensitivity{tag}.png")
        if extra:
            fig_tracks(extra, f"{PNGS}/fig_buzz_tracks_extra{tag}.png", label)
        if conv:
            fig_tracks(conv, f"{PNGS}/fig_buzz_tracks_convergence{tag}.png", label)
        js[which] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]
    fig_by_position(f"{PNGS}/fig_1d_vs_2d_by_position.png")
    txt = "\n".join(out_txt)
    print(txt)
    with open(f"{HERE}/sensitivity_results.txt", "w") as fh:
        fh.write(__doc__ + "\n" + txt + "\n")
    with open(f"{HERE}/sensitivity_results.json", "w") as fh:
        json.dump(js, fh, indent=1, default=float)
