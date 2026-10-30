"""Float-K source scenes, component-domain CLEAN, and blind peak assessment."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, linear_sum_assignment
from scipy.ndimage import maximum_filter, map_coordinates, label, find_objects
from scipy.ndimage import convolve1d


def dirty_fwhm(op, degrees):
    angle = np.deg2rad(degrees)
    q = op.u*np.cos(angle)+op.v*np.sin(angle)
    def half(s):
        return float(np.cos(2*np.pi*q*s).mean()-.5)
    samples = np.linspace(0, op.theta/2, 8193)
    previous = 0.
    for s in samples[1:]:
        if half(s) <= 0:
            return 2*brentq(half, previous, s, xtol=1e-14)
        previous = s
    raise ValueError("no first half-height crossing inside the declared field")


def source_scene(op, center=(0., 0.), degrees=0., ratio=1., distance=0.,
                 sources=2, bounds=(150., 320.)):
    if sources not in (0, 1, 2) or ratio < 1 or distance < 0:
        raise ValueError("invalid source configuration")
    lo, hi = bounds
    xx, yy = np.meshgrid(op.axis, op.axis, indexing="ij")
    width = op.theta/op.n
    gaussian = lambda p: np.exp(-((xx-p[0])**2+(yy-p[1])**2)/(2*width**2))
    reference_flux = float(np.sum(op.w*gaussian((0., 0.)))*.9*(hi-lo))
    truth = np.full_like(xx, lo)
    if sources == 0:
        return truth, np.empty((0, 2)), reference_flux
    angle = np.deg2rad(degrees)
    shift = .5*distance*np.array([np.cos(angle), np.sin(angle)])
    locations = np.array([center]) if sources == 1 else np.array([np.asarray(center)-shift, np.asarray(center)+shift])
    profile = gaussian(locations[0])
    if sources == 2:
        profile = ratio*profile+gaussian(locations[1])
    contrast = profile*(reference_flux/np.sum(op.w*profile))
    truth += contrast
    if truth.max() > hi+1e-9 or truth.min() < lo:
        raise ValueError("source configuration exceeds temperature bounds")
    # Peak matching uses array-axis order, identical to forward xi,eta.
    pixels = (locations/op.theta)*op.n+(op.n-1)/2
    return truth, pixels, reference_flux


def experiment_noise(clean, sigma, family, config, repeat, seed=20261023):
    if min(family, config, repeat) < 0 or sigma < 0:
        raise ValueError("invalid independent RNG family")
    rng = np.random.default_rng(np.random.SeedSequence([seed, family, config, repeat]))
    return clean+sigma/np.sqrt(2)*(rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape))


def dirty_components(op, residual):
    # Divide out image quadrature weights. A unit component is a normalized
    # weighted integral c=w*DeltaT, so dirty beam central amplitude is one.
    return op.adjoint(residual)/op.w/len(op.u)


@dataclass
class CleanResult:
    components: np.ndarray
    residual: np.ndarray
    iterations: int
    status: str


def clean(op, y, background=150., stop_threshold=None, gain=.1, max_iterations=10000):
    if stop_threshold is None or stop_threshold < 0 or not 0 < gain <= 1:
        raise ValueError("CLEAN requires an independently calibrated residual threshold")
    residual = np.asarray(y, complex)-op(np.full((op.n, op.n), background))
    components = np.zeros((op.n, op.n))
    for iteration in range(max_iterations):
        dirty = dirty_components(op, residual)
        pos = np.unravel_index(np.argmax(dirty), dirty.shape)
        amplitude = float(dirty[pos])
        if amplitude <= stop_threshold:
            return CleanResult(components, residual, iteration, "threshold_reached")
        c = gain*amplitude
        components[pos] += c
        residual -= c*np.exp(-2j*np.pi*(op.u*op.axis[pos[0]]+op.v*op.axis[pos[1]]))
    return CleanResult(components, residual, max_iterations, "iteration_limit")


def matched_filter(op, y, sigma, background=150., block=8):
    """Weighted Gaussian templates; standardized scores at every pixel.

    Known width, unknown location/count/amplitude. Template tails beyond eight
    pixel standard deviations are below 1.3e-14 of peak. The same truncated
    template is used for numerator and normalization. No true locations/flux.
    """
    if sigma <= 0:
        raise ValueError("positive noise scale required")
    residual = np.asarray(y)-op(np.full((op.n, op.n), background))
    kernel = np.exp(-.5*np.arange(-8, 9, dtype=float)**2)
    def smooth(a):
        return convolve1d(convolve1d(a, kernel, axis=-1, mode="constant"), kernel, axis=-2, mode="constant")
    numerator = smooth(op.adjoint(residual))
    norm2 = np.zeros((op.n, op.n))
    for start in range(0, len(op.u), block):
        phase = np.exp(-2j*np.pi*(op.u[start:start+block, None, None]*op.axis[None, :, None]
                                  +op.v[start:start+block, None, None]*op.axis[None, None, :]))
        template_vis = smooth(phase*op.w)
        norm2 += np.sum(np.abs(template_vis)**2, axis=0)
    return numerator/(sigma/np.sqrt(2)*np.sqrt(norm2))


def blind_peaks(image, threshold, min_distance=1.):
    image = np.asarray(image, float)
    if image.ndim != 2 or not np.isfinite(image).all() or threshold < 0:
        raise ValueError("finite contrast/component/score map and nonnegative threshold required")
    maxima = (image > threshold) & (image == maximum_filter(image, size=3, mode="constant", cval=-np.inf))
    components, _ = label(maxima, structure=np.ones((3, 3), dtype=int))
    candidates = []
    # One deterministic pixel nearest the centroid per 8-connected plateau.
    # Keep pixel coordinates so component/temperature threshold units are unchanged.
    for index, region in enumerate(find_objects(components), 1):
        pixels = np.argwhere(components[region] == index)
        center = pixels.mean(axis=0)
        representative = pixels[np.argmin(np.sum((pixels-center)**2, axis=1))]
        representative += [s.start for s in region]
        candidates.append(representative.tolist())
    candidates.sort(key=lambda p: (-image[tuple(p)], p[0], p[1]))
    selected = []
    for p in candidates:
        if all(np.linalg.norm(np.asarray(p)-q) > min_distance for q in selected):
            selected.append(np.asarray(p))
    return np.asarray(selected, float).reshape(-1, 2)


def assess_peaks(image, peaks, truth_positions, background=0., require_valley=True):
    truth_positions, peaks = np.asarray(truth_positions), np.asarray(peaks)
    out = {"detected_count": len(peaks), "matched": False, "separated": False, "valley_ratio": None}
    if len(peaks) != len(truth_positions) or not len(peaks):
        return out
    distance = np.linalg.norm(truth_positions[0]-truth_positions[1]) if len(truth_positions) == 2 else 0.
    tolerance = max(1., .2*distance)
    distances = np.linalg.norm(truth_positions[:, None]-peaks[None], axis=2)
    rows, columns = linear_sum_assignment(distances)
    out["matched"] = bool(np.all(distances[rows, columns] <= tolerance))
    if not out["matched"] or len(peaks) != 2:
        return out
    if not require_valley:
        return out  # MF reports matching/detection, never a temperature separation.
    steps = max(3, int(np.ceil(np.linalg.norm(peaks[0]-peaks[1])/.25))+1)
    line = np.linspace(peaks[0], peaks[1], steps).T
    values = map_coordinates(np.asarray(image, float), line, order=1, mode="nearest")
    denominator = min(values[0], values[-1])-background
    if denominator > 0:
        ratio = (float(values[1:-1].min())-background)/denominator
        out["valley_ratio"] = ratio
        out["separated"] = bool(ratio <= .8)
    return out


def calibrate_threshold(blank_maps, single_maps, single_positions, min_distance=1.):
    """Smallest tested threshold satisfying both empirical control targets.

    Input maps must already be in ONE method's declared detection units.
    No threshold may be transferred between CLEAN, temperature, and MF.
    """
    blank_maps, single_maps = list(blank_maps), list(single_maps)
    if not blank_maps or len(single_maps) != len(single_positions):
        raise ValueError("independent blank and located single-source calibration sets required")
    def peak_height(image, index):
        peaks = blind_peaks(image, 0., min_distance)
        return float(image[tuple(peaks[index].astype(int))]) if len(peaks) > index else 0.
    maxima = np.array([peak_height(x, 0) for x in blank_maps])
    second = np.array([peak_height(x, 1) for x in single_maps])
    for threshold in sorted(set([0., *maxima, *second])):
        false_blank = np.mean(maxima > threshold)
        false_split = np.mean(second > threshold)
        if false_blank <= .01 and false_split <= .01:
            return {"threshold": threshold, "empirical_false_blank": float(false_blank),
                    "empirical_false_split": float(false_split), "calibration_only": True}
    raise RuntimeError("threshold calibration failed")
