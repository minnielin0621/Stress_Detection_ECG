
import warnings
import numpy as np
import pandas as pd

from neurokit2.signal import signal_psd
from neurokit2.signal.signal_power import _signal_power_continuous
from neurokit2.hrv.hrv_frequency import _hrv_format_input
from neurokit2.hrv.intervals_process import intervals_process
from antropy.utils import _xlogx
from scipy.integrate import simpson


def frequency_domain(
    ulf=(0, 0.0033),
    vlf=(0.0033, 0.04),
    lf=(0.04, 0.15),
    hf=(0.15, 0.4),
    vhf=(0.4, 0.5),
    uhf=(0.5, 1),
    sampling_rate=1000,
    output_all=False,
):
    """Extract frequency features.

    output_all=False: original UHF total power only.
    output_all=True: all 24 HF/VHF/UHF features.
    """

    def inner(rpeaks):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = hrv_frequency(
                rpeaks,
                sampling_rate=sampling_rate,
                hf=hf,
                vhf=vhf,
                uhf=uhf,
            )

        if output_all:
            return dict(features)

        return {
            "total_band_power_uhf_band": features.get(
                "total_band_power_uhf_band", np.nan
            )
        }

    return inner


def hrv_frequency(
    peaks,
    sampling_rate=1000,
    hf=(0.15, 0.4),
    vhf=(0.4, 0.5),
    uhf=(0.5, 1),
    psd_method="welch",
    normalize=True,
    order_criteria=None,
    interpolation_rate=100,
    **kwargs,
):
    rri, rri_time, _ = _hrv_format_input(
        peaks, sampling_rate=sampling_rate
    )

    rri, rri_time, sampling_rate = intervals_process(
        rri,
        intervals_time=rri_time,
        interpolate=True,
        interpolation_rate=interpolation_rate,
        **kwargs,
    )

    t = rri_time if interpolation_rate is None else None
    frequency_band = [hf, vhf, uhf]
    max_frequency = np.max([np.max(b) for b in frequency_band])

    return signal_power(
        rri,
        frequency_band=frequency_band,
        sampling_rate=sampling_rate,
        method=psd_method,
        max_frequency=max_frequency,
        normalize=normalize,
        order_criteria=order_criteria,
        t=t,
    )


def signal_power(
    signal,
    frequency_band,
    sampling_rate=1000,
    continuous=False,
    normalize=True,
    **kwargs,
):
    if continuous:
        return _signal_power_continuous(
            signal, frequency_band, sampling_rate=sampling_rate
        )

    return _signal_power_instant(
        signal,
        frequency_band,
        sampling_rate=sampling_rate,
        normalize=normalize,
        **kwargs,
    )


def _signal_power_instant(
    signal,
    frequency_band,
    sampling_rate=1000,
    normalize=True,
    order_criteria="KIC",
    **kwargs,
):
    if isinstance(frequency_band[0], (int, float)):
        frequency_band = [frequency_band]

    min_freq = min(b[0] for b in frequency_band)
    max_freq = max(b[1] for b in frequency_band)

    psd = signal_psd(
        signal,
        sampling_rate=sampling_rate,
        show=False,
        normalize=normalize,
        order_criteria=order_criteria,
        **kwargs,
    )

    psd = psd[
        (psd["Frequency"] >= min_freq)
        & (psd["Frequency"] <= max_freq)
    ]

    total_power = sum(
        _signal_power_instant_compute(psd, band)
        for band in frequency_band
    )

    result = {}

    for band in frequency_band:
        name = _return_band_classification(band)

        result[f"min_power_{name}_band"] = _signal_min_instant_compute(psd, band)
        result[f"max_power_{name}_band"] = _signal_max_instant_compute(psd, band)
        result[f"median_power_{name}_band"] = _signal_median_instant_compute(psd, band)
        result[f"mean_power_{name}_band"] = _signal_mean_instant_compute(psd, band)
        result[f"std_power_{name}_band"] = _signal_std_instant_compute(psd, band)
        result[f"entropy_power_{name}_band"] = _signal_entropy_instant_compute(psd, band)

        band_power = _signal_power_instant_compute(psd, band)
        result[f"total_band_power_{name}_band"] = band_power
        result[f"relative_band_power_{name}_band"] = (
            band_power / total_power * 100
            if np.isfinite(total_power) and total_power != 0
            else np.nan
        )

    return result


def _return_band_classification(band):
    mapping = {
        0.0: "ulf",
        0.0033: "vlf",
        0.04: "lf",
        0.15: "hf",
        0.4: "vhf",
        0.5: "uhf",
    }
    return mapping.get(band[0], "unknown")


def _band_values(psd, band):
    if band[1] < 1.0:
        mask = (
            (psd["Frequency"] >= band[0])
            & (psd["Frequency"] < band[1])
        )
    else:
        mask = (
            (psd["Frequency"] >= band[0])
            & (psd["Frequency"] <= band[1])
        )

    return psd.loc[mask, "Power"].to_numpy()


def _signal_min_instant_compute(psd, band):
    values = _band_values(psd, band)
    return np.min(values) if len(values) else np.nan


def _signal_max_instant_compute(psd, band):
    values = _band_values(psd, band)
    return np.max(values) if len(values) else np.nan


def _signal_mean_instant_compute(psd, band):
    values = _band_values(psd, band)
    return np.mean(values) if len(values) else np.nan


def _signal_median_instant_compute(psd, band):
    values = _band_values(psd, band)
    return np.median(values) if len(values) else np.nan


def _signal_std_instant_compute(psd, band):
    values = _band_values(psd, band)
    return np.std(values) if len(values) else np.nan


def _signal_power_instant_compute(psd, band):
    if band[1] < 1.0:
        mask = (
            (psd["Frequency"] >= band[0])
            & (psd["Frequency"] < band[1])
        )
    else:
        mask = (
            (psd["Frequency"] >= band[0])
            & (psd["Frequency"] <= band[1])
        )

    power = psd.loc[mask, "Power"].to_numpy()
    frequency = psd.loc[mask, "Frequency"].to_numpy()

    if len(power) < 2:
        return np.nan

    return simpson(y=power, x=frequency)


def _signal_entropy_instant_compute(psd, band):
    values = _band_values(psd, band)

    if not len(values) or values.sum() <= 0:
        return np.nan

    normalized = values / values.sum()
    return -_xlogx(normalized).sum()
