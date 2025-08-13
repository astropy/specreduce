#!/usr/bin/env python3
import os
import numpy as np
import astropy.units as u

from dataclasses import dataclass
from astropy.io import fits
from astropy.nddata import NDData, VarianceUncertainty
from astropy.modeling import models

from specreduce.extract import HorneExtract
from specreduce.tracing import FlatTrace


@dataclass
class CaseSpec:
    name: str
    ny: int
    nx: int
    amp: float
    mean: float
    std: float
    bkg: float
    sigma_per_pix: float
    vary_row_sigma: bool = False
    mask_fraction: float = 0.0
    kill_rows: tuple = ()


def make_image(spec: CaseSpec, seed: int = 1234):
    rng = np.random.default_rng(seed)
    y = np.arange(spec.ny, dtype=float)

    profile = spec.amp * np.exp(-0.5 * ((y - spec.mean) / spec.std) ** 2) + spec.bkg
    ideal = np.tile(profile[:, None], (1, spec.nx))

    if spec.vary_row_sigma:
        sigma_row = spec.sigma_per_pix * (0.5 + 0.5 * (y - y.min()) / (y.max() - y.min()))
        var2d = (sigma_row[:, None] ** 2) * np.ones((1, spec.nx))
    else:
        var2d = np.full((spec.ny, spec.nx), spec.sigma_per_pix**2, dtype=float)

    noise = rng.normal(0.0, np.sqrt(var2d))
    data = ideal + noise

    mask = np.zeros_like(data, dtype=bool)
    if spec.mask_fraction > 0.0:
        k = int(spec.mask_fraction * data.size)
        idx = rng.choice(data.size, size=k, replace=False)
        mask.flat[idx] = True
    if spec.kill_rows:
        mask[np.array(spec.kill_rows, dtype=int), :] = True

    return data, var2d, mask


def write_fits(path: str, data: np.ndarray, var2d: np.ndarray, mask: np.ndarray, unit: str = "DN"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    phdu = fits.PrimaryHDU(data.astype(np.float64))
    phdu.header["BUNIT"] = unit
    vhdu = fits.ImageHDU(var2d.astype(np.float64), name="VARIANCE")
    mhdu = fits.ImageHDU(mask.astype(np.uint8), name="MASK")
    fits.HDUList([phdu, vhdu, mhdu]).writeto(path, overwrite=True)


def read_fits_as_nddata(path: str) -> NDData:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        unit = hdul[0].header.get("BUNIT", "DN")
        var2d = hdul["VARIANCE"].data.astype(np.float64)
        mask = hdul["MASK"].data.astype(bool)
    return NDData(
        data=data * u.Unit(unit),
        uncertainty=VarianceUncertainty(var2d * (u.Unit(unit) ** 2)),
        mask=mask,
        unit=u.Unit(unit),
    )


def summarize_sigma_row(title: str, computed: np.ndarray, expected: np.ndarray):
    diff = computed - expected
    finite = np.isfinite(expected) & np.isfinite(computed)
    mad = np.nanmedian(np.abs(diff[finite])) if np.any(finite) else np.nan
    maxad = np.nanmax(np.abs(diff[finite])) if np.any(finite) else np.nan
    nfin = int(np.sum(finite))
    print(f"[{title}] sigma_row vs expected")
    print(f"  finite rows   : {nfin}")
    print(f"  median|diff|  : {mad:.6g}")
    print(f"  max|diff|     : {maxad:.6g}\n")


def print_fit_summary(model, label="fit"):
    # Gaussian component is index 0, background (if present) is index 1
    stderr = model.meta.get("param_stderr", {}) or {}
    cov = model.meta.get("param_cov", None)

    def serr(pname, comp_index):
        key = f"{pname}_{comp_index}"
        return float(stderr.get(key, np.nan))

    # Gaussian params from component 0
    g = model[0] if hasattr(model, "__len__") else model
    print(f"[{label}] fitted parameters")
    print(f"  amplitude = {g.amplitude.value:.6g} ± {serr('amplitude', 0):.3g}")
    print(f"  mean      = {g.mean.value:.6g} ± {serr('mean', 0):.3g}")
    print(f"  stddev    = {g.stddev.value:.6g} ± {serr('stddev', 0):.3g}")

    # Background amplitude, if present, is component 1
    if hasattr(model, "__len__") and len(model) > 1:
        b = model[1]
        print(f"  background = {b.amplitude.value:.6g} ± {serr('amplitude', 1):.3g}")

    if cov is not None:
        print(f"  covariance shape: {np.shape(cov)}")
    print("")


def run_case(spec: CaseSpec, outdir="synthetic_cases", with_background=True):
    print("=" * 70)
    print(f"Case: {spec.name}")

    data, var2d, mask = make_image(spec)
    fpath = os.path.join(outdir, f"{spec.name}.fits")
    write_fits(fpath, data, var2d, mask, unit="DN")
    nd = read_fits_as_nddata(fpath)

    valid = ~mask
    Ni = valid.sum(axis=1).astype(float)
    sumvar = (valid * var2d).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_mean = np.where(Ni > 0, sumvar / (Ni**2), np.nan)
    expected_sigma = np.sqrt(var_mean)

    trace = FlatTrace(image=nd, trace_pos=spec.mean)
    ex = HorneExtract(nd, trace_object=trace)

    bkgrd = models.Const1D() if with_background else None
    model = ex._fit_gaussian_spatial_profile(
        img=nd.data,          # <-- only change: use nd.data, not nd.data.value
        disp_axis=1,
        crossdisp_axis=0,
        or_mask=mask,
        bkgrd_prof=bkgrd,
    )

    summarize_sigma_row(spec.name, ex._last_profile_sigma, expected_sigma)
    print_fit_summary(model, label=spec.name)


if __name__ == "__main__":
    cases = [
        CaseSpec("uniform_no_mask", 50, 300, 150.0, 23.0, 4.0, 3.0, 6.0),
        CaseSpec("random_mask_empty_rows", 40, 200, 120.0, 15.0, 3.0, 2.0, 4.0, mask_fraction=0.35, kill_rows=(3, 7)),
        CaseSpec("varying_row_sigma", 60, 250, 200.0, 29.0, 5.0, 1.0, 5.0, vary_row_sigma=True),
    ]
    for spec in cases:
        run_case(spec, with_background=True)
