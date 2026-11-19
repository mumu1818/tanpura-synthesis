"""Diagnostic: long-term band levels (dB re total bridge-force energy) of each force component for
fs = 44.1 / 88.2 / 176.4 kHz with the same 133 modes.  Shows that the energy-conserving contact
discretisation puts the unresolved (near-rigid) contact oscillation at fs/4: in the 10-15 kHz band at
fs = 44.1 kHz, at ~22 kHz for 88.2 kHz, and out of the audio band (44 kHz) for 176.4 kHz.
Run: python diag_bands.py"""
import numpy as np
import scipy.signal as ss
from common import to_audio
from tanpura1d import simulate

BANDS = [(0, 2e3), (2e3, 5e3), (5e3, 10e3), (10e3, 15e3), (15e3, 20e3), (20e3, 22.05e3)]


def bandlev(y, ref):
    f, P = ss.welch(y, 44100, nperseg=8192)
    _, Pr = ss.welch(ref, 44100, nperseg=8192)
    return [10 * np.log10(P[(f >= a) & (f < b)].sum() / Pr.sum() + 1e-30) for a, b in BANDS]


if __name__ == "__main__":
    print("bands (kHz):       0-2    2-5   5-10  10-15  15-20  20-22")
    for fs in (44100.0, 88200.0, 176400.0):
        p, s, r = simulate(dict(dur=3.0, fs=fs))
        ref = to_audio(r["Fbridge"], fs)
        for sig, lab in (("Fbridge", "bridge total"), ("Fnut", "nut"), ("Fb", "contact F_b"), ("Fc", "thread F_c")):
            y = to_audio(r[sig], fs)
            print(f"{fs / 1e3:5.1f}k {lab:12s}" + " ".join(f"{v:6.1f}" for v in bandlev(y, ref if sig != "Fnut" else y)))
