from __future__ import annotations
import numpy as np
import librosa


def compute_loss(ref_y: np.ndarray, synth_y: np.ndarray, sr: int) -> float:
    ref = librosa.util.normalize(ref_y)
    synth = librosa.util.normalize(synth_y)
    min_len = min(len(ref), len(synth))
    ref = ref[:min_len]
    synth = synth[:min_len]

    n_fft = 2048
    hop = 512

    S_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop))
    S_synth = np.abs(librosa.stft(synth, n_fft=n_fft, hop_length=hop))

    # --- SPECTRAL LOSSES ---

    # Log Spectral Distance
    log_ref = librosa.amplitude_to_db(S_ref, ref=np.max)
    log_synth = librosa.amplitude_to_db(S_synth, ref=np.max)
    loss_lsd = np.mean(np.abs(log_ref - log_synth))

    # Spectral centroid trajectory
    cent_ref = librosa.feature.spectral_centroid(S=S_ref)[0]
    cent_synth = librosa.feature.spectral_centroid(S=S_synth)[0]
    loss_centroid = (
        abs(np.mean(cent_ref) - np.mean(cent_synth))
        + abs(np.std(cent_ref) - np.std(cent_synth))
    )

    # Spectral flux
    flux_ref = np.diff(S_ref, axis=1)
    flux_synth = np.diff(S_synth, axis=1)
    loss_flux = (
        abs(np.mean(flux_ref) - np.mean(flux_synth)) * 10.0
        + abs(np.std(flux_ref) - np.std(flux_synth)) * 10.0
    )

    # --- PERCEPTUAL LOSSES ---

    # MFCC distance
    mfcc_ref = librosa.feature.mfcc(S=librosa.power_to_db(S_ref ** 2), n_mfcc=13)
    mfcc_synth = librosa.feature.mfcc(S=librosa.power_to_db(S_synth ** 2), n_mfcc=13)
    min_frames = min(mfcc_ref.shape[1], mfcc_synth.shape[1])
    loss_mfcc = np.mean(np.abs(mfcc_ref[:, :min_frames] - mfcc_synth[:, :min_frames]))

    # Temporal envelope correlation
    env_ref = np.abs(librosa.onset.onset_strength(S=librosa.power_to_db(S_ref ** 2)))
    env_synth = np.abs(librosa.onset.onset_strength(S=librosa.power_to_db(S_synth ** 2)))
    min_env = min(len(env_ref), len(env_synth))
    if np.std(env_ref[:min_env]) > 1e-9 and np.std(env_synth[:min_env]) > 1e-9:
        corr = np.corrcoef(env_ref[:min_env], env_synth[:min_env])[0, 1]
        loss_envelope = 1.0 - max(corr, 0.0)
    else:
        loss_envelope = 1.0

    # --- WEIGHTED COMBINATION ---
    return (
        loss_lsd * 0.3
        + (loss_centroid / 1000.0) * 0.15
        + loss_flux * 0.15
        + loss_mfcc * 0.25
        + loss_envelope * 0.15
    )
