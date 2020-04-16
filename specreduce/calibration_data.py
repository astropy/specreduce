"""
Utilities for defining, loading, and handling spectroscopic calibration data
"""

import os
import pkg_resources
import warnings


"""
Make specreduce_data optional. If it's available, great and we can access its data via
pkg_resources. If not, we'll fall back to downloading and caching it using astropy.utils.data.
"""
LOCAL_DATA = True
try:
    import specreduce_data  # noqa
except ModuleNotFoundError:
    warnings.warn("Can't import specreduce_data package. Falling back to downloading data...")
    LOCAL_DATA = False
    from astropy.utils.data import download_file


def get_reference_file_path(path=None, cache=False, show_progress=False):
    """
    Basic function to take a path to a file and load it via pkg_resources if the specreduce_data
    package is available and load it via ghithub raw user content if not.
    """
    if path is None:
        return None

    if LOCAL_DATA:
        file_path = pkg_resources.resource_filename(
            "specreduce_data",
            os.path.join("reference_data", path)
        )
    else:
        repo_url = "https://raw.githubusercontent.com/astropy/specreduce-data"
        remote_url = f"{repo_url}/master/specreduce_data/reference_data/{path}"
        try:
            file_path = download_file(
                remote_url,
                cache=cache,
                show_progress=show_progress,
                pkgname='specreduce'
            )
        except Exception as e:
            msg = f"Downloading of {path} failed: {e}"
            warnings.warn(msg)
            return None

    # final sanity check to make sure file_path is actually a file.
    if os.path.isfile(file_path):
        return file_path
    else:
        warnings.warn(f"Able to construct {file_path}, but it is not a file.")
        return None
