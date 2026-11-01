"""
COMPASS AI ModelController.
Manages model discovery across the user's autonomous AppData directory
(`%APPDATA%\\COMPASS_AI\\models` on Windows, `~/Library/Application Support/COMPASS_AI/models` on macOS,
`~/.config/COMPASS_AI/models` on Linux) and bundled `./data/models`, hardware backend selection,
wrapper instantiation, and generation execution.
Includes automatic bypass of Windows Store Python MSIX AppData VFS redirection so that
`%APPDATA%\\COMPASS_AI\\models` always exists on the real physical host filesystem.
"""

from collections import OrderedDict
import copy
import glob
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import zipfile

from models.mortm.mortm45 import MORTM45Rapper
from models.mortm.mortm45d import MORTM45DRapper
from mortm.models.modules.backend import HardwareBackendFactory, IHardwareBackend
from mortm.models.modules.progress import (
    GLOBAL_GENERATION_PROGRESS,
    GenerationCancelledError,
)
from rapper import AbstractModelRapper, GenerateMeta, ModelRapperFactory


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _next_available_backup_path(base_path: str, suffix: str) -> str:
    candidate = f"{base_path}{suffix}"
    if not os.path.exists(candidate):
        return candidate
    idx = 1
    while True:
        numbered = f"{base_path}{suffix}.{idx}"
        if not os.path.exists(numbered):
            return numbered
        idx += 1


def _is_msix_python() -> bool:
    if not sys.platform.startswith("win"):
        return False
    probes = (
        sys.executable or "",
        getattr(sys, "_base_executable", "") or "",
        sys.base_prefix or "",
    )
    return any("windowsapps" in p.lower() for p in probes)


def _strip_win_device_prefix(path: str) -> str:
    """
    Strips Win32 local device / extended-length prefixes (`\\\\?\\C:\\...`, `\\\\.\\C:\\...`,
    `\\\\?\\Volume{GUID}\\...`, `\\\\.\\Volume{GUID}\\...`, `\\\\?\\UNC\\server\\share\\...`,
    `\\\\.\\UNC\\server\\share\\...`) so local volume paths are never misclassified as network
    SMB UNC shares (`[R6C-MED-01]`, `[R4-LOW-01]`).
    """
    if not sys.platform.startswith("win"):
        return path
    norm = path.replace("/", "\\")
    low = norm.lower()
    if low.startswith(("\\\\?\\unc\\", "\\\\.\\unc\\")):
        return "\\\\" + norm[8:]
    if low.startswith(("\\\\?\\volume{", "\\\\.\\volume{")):
        close_idx = norm.find("}", 11)
        if close_idx != -1 and (
            close_idx + 1 == len(norm) or norm[close_idx + 1] == "\\"
        ):
            vol_root = norm[: close_idx + 1] + "\\"
            tail_part = norm[close_idx + 1 :].lstrip("\\")
            try:
                res_root = os.path.realpath(vol_root).replace("/", "\\")
                res_low = res_root.lower()
                if (
                    res_low.startswith(("\\\\?\\", "\\\\.\\"))
                    and len(res_root) >= 6
                    and res_root[4].isalpha()
                    and res_root[5] == ":"
                ):
                    res_root = res_root[4:]
                if len(res_root) >= 2 and res_root[0].isalpha() and res_root[1] == ":":
                    drive_root = f"{res_root[0]}:\\"
                    return os.path.join(drive_root, tail_part) if tail_part else drive_root
            except OSError:
                pass
    if (
        low.startswith(("\\\\?\\", "\\\\.\\"))
        and len(norm) >= 6
        and norm[4].isalpha()
        and norm[5] == ":"
        and (len(norm) == 6 or norm[6] == "\\")
    ):
        return (norm[4:] + "\\") if len(norm) == 6 else norm[4:]
    return path


def _is_smb_unc_path(path: str) -> bool:
    """
    True strictly for genuine network / loopback SMB UNC paths (`\\\\server\\share\\...`,
    including `\\\\?\\UNC\\server\\share\\...`), and False for local drive or Win32 device paths
    (`C:\\...`, `\\\\?\\C:\\...`, `\\\\.\\C:\\...`, `\\\\?\\Volume{...}`) (`[R6C-MED-01]`).
    """
    if not sys.platform.startswith("win"):
        return False
    norm = _strip_win_device_prefix(path).replace("/", "\\")
    return norm.startswith("\\\\") and not norm.lower().startswith(("\\\\?\\", "\\\\.\\"))


def _unwrap_loopback_drive_unc(path: str) -> str:
    """
    Normalizes Win32 device prefixes and loopback admin drive shares (`\\\\127.0.0.1\\<Drive>$\\...`,
    `\\\\localhost\\<Drive>$\\...`) into standard `<Drive>:\\...` form (`[R6C-MED-01]`).
    """
    if not sys.platform.startswith("win"):
        return path
    norm = _strip_win_device_prefix(path).replace("/", "\\")
    low = norm.lower()
    for host_prefix in ("\\\\127.0.0.1\\", "\\\\localhost\\"):
        plen = len(host_prefix)
        if (
            low.startswith(host_prefix)
            and len(norm) >= plen + 2
            and norm[plen].isalpha()
            and norm[plen + 1] == "$"
            and (len(norm) == plen + 2 or norm[plen + 2] == "\\")
        ):
            rest = norm[plen + 2 :].lstrip("\\")
            return f"{norm[plen]}:\\{rest}"
    return norm


def _get_default_platform_models_dir() -> str:
    """
    Returns the default OS-native user models directory (ignoring `COMPASS_MODELS_DIR` overrides).
    """
    if sys.platform.startswith("win"):
        appdata = _strip_win_device_prefix(
            os.environ.get("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
        )
        return os.path.abspath(os.path.join(appdata, "COMPASS_AI", "models"))
    elif sys.platform == "darwin":
        return os.path.abspath(os.path.expanduser("~/Library/Application Support/COMPASS_AI/models"))
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
        return os.path.abspath(os.path.join(xdg, "COMPASS_AI", "models"))


def get_user_models_dir() -> str:
    """
    Returns the OS-appropriate user AppData models directory where users autonomously place
    model folders (`model.pth`, `config.json`, `data.json`).
    - Windows: `%APPDATA%\\COMPASS_AI\\models` (e.g. `C:\\Users\\<User>\\AppData\\Roaming\\COMPASS_AI\\models`)
    - macOS: `~/Library/Application Support/COMPASS_AI/models`
    - Linux: `~/.config/COMPASS_AI/models`
    Can be overridden via `COMPASS_MODELS_DIR` environment variable.
    """
    override = os.environ.get("COMPASS_MODELS_DIR")
    if override and override.strip():
        return os.path.abspath(_strip_win_device_prefix(override.strip()))
    return _get_default_platform_models_dir()


def _is_default_windows_models_dir(target_dir: str) -> bool:
    """
    True strictly when running on Windows with NO `COMPASS_MODELS_DIR` override and `target_dir`
    resolves to the canonical `%APPDATA%\\COMPASS_AI\\models` directory.
    Prevents unit tests / E2E temp sandboxes from ever touching a user's real MSIX `LocalCache`.
    """
    if not sys.platform.startswith("win"):
        return False
    if os.environ.get("COMPASS_MODELS_DIR", "").strip():
        return False
    try:
        return _canonical_local_realpath(target_dir) == _canonical_local_realpath(
            _get_default_platform_models_dir()
        )
    except OSError:
        return False


def resolve_host_write_path(path: str) -> str:
    """
    On Windows Store Python (MSIX `WindowsApps`), direct writes to `C:\\Users\\<User>\\AppData\\Roaming`
    are intercepted by `bindflt.sys` and redirected to `LocalCache\\Roaming`.
    Using the loopback UNC path `\\\\127.0.0.1\\<Drive>$\\...` when available writes directly to the
    physical host filesystem without triggering MSIX VFS redirection (`[R4-LOW-01]`).
    """
    abs_path = os.path.abspath(_strip_win_device_prefix(path))
    if not sys.platform.startswith("win") or _is_smb_unc_path(abs_path) or not _is_msix_python():
        return abs_path

    drive, tail = os.path.splitdrive(abs_path)
    if len(drive) == 2 and drive[1] == ":":
        unc_root = f"\\\\127.0.0.1\\{drive[0]}$\\"
        try:
            if os.path.isdir(unc_root):
                return f"\\\\127.0.0.1\\{drive[0]}${tail}"
        except OSError:
            pass
    return abs_path


def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


def resolve_physical_models_dir_for_shell(user_models_dir: str) -> str:
    """
    Returns the true physical host path that external Win32 shells (`explorer.exe` / Electron UI)
    should display or open (`[R2-MED-02]`, `[R6C-MED-01]`, `[R4-LOW-01]`).
    - Standard Python or MSIX Python with loopback UNC available: `%APPDATA%\\COMPASS_AI\\models`.
    - MSIX Store Python WITHOUT loopback UNC (e.g. non-admin Windows Home): Python writes are
      redirected by `bindflt.sys` into `%LOCALAPPDATA%\\Packages\\PythonSoftwareFoundation.Python.<major>.<minor>_*\\LocalCache\\Roaming\\COMPASS_AI\\models`,
      so returning that exact version's physical `LocalCache` path ensures external shells open the folder Python reads.
    """
    abs_dir = os.path.abspath(_strip_win_device_prefix(user_models_dir))
    if not _is_default_windows_models_dir(abs_dir):
        return abs_dir
    if not _is_msix_python():
        return abs_dir
    if _is_smb_unc_path(resolve_host_write_path(abs_dir)):
        return abs_dir

    localappdata = _strip_win_device_prefix(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    )
    pattern = os.path.join(
        localappdata,
        "Packages",
        "PythonSoftwareFoundation.Python*",
        "LocalCache",
        "Roaming",
        "COMPASS_AI",
        "models",
    )
    candidates = [os.path.abspath(p) for p in glob.glob(pattern) if os.path.isdir(p)]
    if candidates:
        ver_prefix = (
            f"PythonSoftwareFoundation.Python.{sys.version_info[0]}.{sys.version_info[1]}_"
        )
        matching_ver = [
            p for p in candidates if ver_prefix.lower() in os.path.normcase(p)
        ]
        if matching_ver:
            matching_ver.sort(key=_safe_mtime, reverse=True)
            return matching_ver[0]
    return abs_dir


def _is_reparse_point(path: str) -> bool:
    """
    Detects POSIX/Win32 symlinks, NTFS directory junctions (`IO_REPARSE_TAG_MOUNT_POINT`),
    and Win32 reparse points even on Python < 3.12 where `os.path.islink()` returns `False`
    for NTFS junctions (`[R4-HIGH-01]`). Fails closed (`True`) on `OSError`.
    """
    try:
        if os.path.islink(path):
            return True
        st = os.lstat(path)
        if getattr(st, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        ):
            return True
        if getattr(st, "st_reparse_tag", 0) != 0:
            return True
        return False
    except OSError:
        return True


def _canonical_local_realpath(path: str) -> str:
    """
    Resolves Win32 extended-length/device prefixes (`\\\\?\\`, `\\\\.\\`, `\\\\?\\UNC\\`),
    Windows loopback UNC shares (`\\\\127.0.0.1\\C$\\...`, `\\\\localhost\\C$\\...`), NTFS junctions,
    symlinks, `subst` drives, and 8.3 short names (`TAKAFU~1`) into a canonical
    `normcase(realpath(...))` (`[R6B-MED-04]`, `[R6C-MED-01]`).
    """
    raw = os.path.abspath(_unwrap_loopback_drive_unc(path))
    try:
        resolved = _unwrap_loopback_drive_unc(os.path.realpath(raw))
        return os.path.normcase(os.path.normpath(resolved))
    except OSError:
        return os.path.normcase(os.path.normpath(raw))


def _paths_alias_or_overlap(path_a: str, path_b: str, check_containment: bool = False) -> bool:
    """
    Returns `True` if `path_a` and `path_b` refer to the same physical file/directory via `os.path.samefile`
    or canonical `realpath` equality (and optionally parent/child containment for directories) (`[R6B-MED-04]`).
    """
    try:
        if os.path.exists(path_a) and os.path.exists(path_b) and os.path.samefile(path_a, path_b):
            return True
    except OSError:
        pass
    canon_a = _canonical_local_realpath(path_a)
    canon_b = _canonical_local_realpath(path_b)
    if canon_a == canon_b:
        return True
    if check_containment:
        try:
            common = os.path.commonpath([canon_a, canon_b])
            if common in (canon_a, canon_b):
                return True
        except ValueError:
            pass
    return False


def _clean_msix_localcache_shadow(
    real_user_models_dir: str,
    bundled_models_dir: Optional[str] = None,
) -> None:
    """
    Safely migrates files trapped inside Microsoft Store Python's MSIX VFS shadow
    (`%LOCALAPPDATA%\\Packages\\PythonSoftwareFoundation.Python*\\LocalCache\\Roaming\\COMPASS_AI`)
    to the physical host `%APPDATA%\\COMPASS_AI` directory.

    Strict Data-Safety Invariants (`[CRIT-01]`, `[R2-MED-01]`, `[R3-MED-04]`, `[R6B-MED-04]`):
    1. Runs ONLY when `real_user_models_dir` is the real default `%APPDATA%\\COMPASS_AI\\models`
       (never when `COMPASS_MODELS_DIR` or a custom/temp `user_models_dir` is active).
    2. When running inside MSIX Store Python (`_is_msix_python() == True`) without loopback UNC share
       (`\\\\127.0.0.1\\<Drive>$\\`), returns immediately WITHOUT deleting `shadow_compass` so user
       `model.pth` weights inside `LocalCache` remain 100% intact. When running standard Win32 Python
       (`_is_msix_python() == False`), `real_user_models_dir` writes directly to the physical host disk
       without `bindflt.sys` redirection even when `UNC=False`, safely migrating prior Store weights.
    3. Verifies `os.path.samefile` and canonical `os.path.realpath` non-aliasing/non-overlap across
       NTFS junctions, symlinks, `subst` drives, and Windows 8.3 short paths (`TAKAFU~1`) before any delete.
    4. Propagates `os.walk(..., onerror=...)` errors so unreadable subdirectories abort deletion.
    5. Compares SHA-256 content hashes whenever destination files already exist, preserving both
       same-size and different-size user edits (migrating or backing up divergent files) and verifying
       exact byte size + SHA-256 hash before removing `shadow_compass`.
    """
    if not _is_default_windows_models_dir(real_user_models_dir):
        return

    host_write_models_dir = resolve_host_write_path(real_user_models_dir)
    if _is_msix_python() and not _is_smb_unc_path(host_write_models_dir):
        return

    real_compass_dir = os.path.dirname(
        os.path.abspath(_strip_win_device_prefix(real_user_models_dir))
    )
    host_write_compass_dir = os.path.dirname(host_write_models_dir)
    localappdata = _strip_win_device_prefix(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    )
    pattern = os.path.join(
        localappdata,
        "Packages",
        "PythonSoftwareFoundation.Python*",
        "LocalCache",
        "Roaming",
        "COMPASS_AI",
    )

    for shadow_compass in glob.glob(pattern):
        try:
            if not os.path.isdir(shadow_compass) or _is_reparse_point(shadow_compass):
                continue

            # Reject if any MSIX container ancestor (`Roaming`, `LocalCache`, `PythonSoftwareFoundation.*`)
            # is itself a symlink or NTFS junction (`[R4-HIGH-01]`).
            anc_roaming = os.path.dirname(shadow_compass)
            anc_localcache = os.path.dirname(anc_roaming)
            anc_pkg = os.path.dirname(anc_localcache)
            if any(_is_reparse_point(anc) for anc in (anc_roaming, anc_localcache, anc_pkg)):
                continue

            protected_roots: List[str] = [
                real_compass_dir,
                host_write_compass_dir,
                real_user_models_dir,
                host_write_models_dir,
            ]
            if bundled_models_dir:
                protected_roots.append(bundled_models_dir)

            if any(
                _paths_alias_or_overlap(shadow_compass, pr, check_containment=True)
                for pr in protected_roots
            ):
                continue

            shadow_canon = _canonical_local_realpath(shadow_compass)

            # Phase 1: Pure read-only tree discovery & reparse/containment audit (`[R4-HIGH-01]`).
            # Zero `os.makedirs` or `shutil.copy2` calls execute during `os.walk`, eliminating
            # self-referential directory growth and sibling `%APPDATA%` junction exfiltration.
            planned_dirs: List[str] = [host_write_compass_dir]
            planned_files: List[Tuple[str, str]] = []
            copy_failed = False
            walk_failed = False

            def _on_walk_error(_err: OSError) -> None:
                nonlocal walk_failed
                walk_failed = True

            for root, dirs, files in os.walk(
                shadow_compass, topdown=True, onerror=_on_walk_error, followlinks=False
            ):
                if walk_failed or _is_reparse_point(root):
                    copy_failed = True
                    dirs[:] = []
                    break

                rel_dir = os.path.relpath(root, shadow_compass)
                rel_parts = [] if rel_dir in (".", "") else rel_dir.split(os.sep)
                if len(rel_parts) > 32 or ".." in rel_parts:
                    copy_failed = True
                    dirs[:] = []
                    break

                root_canon = _canonical_local_realpath(root)
                try:
                    if os.path.commonpath([shadow_canon, root_canon]) != shadow_canon:
                        copy_failed = True
                        dirs[:] = []
                        break
                except ValueError:
                    copy_failed = True
                    dirs[:] = []
                    break

                dst_dir = (
                    host_write_compass_dir
                    if rel_dir in (".", "")
                    else os.path.join(host_write_compass_dir, rel_dir)
                )
                if _paths_alias_or_overlap(root, dst_dir, check_containment=True) or any(
                    _paths_alias_or_overlap(root, pr, check_containment=True)
                    for pr in protected_roots
                ):
                    copy_failed = True
                    dirs[:] = []
                    break

                planned_dirs.append(dst_dir)

                for dname in list(dirs):
                    sub_src = os.path.join(root, dname)
                    sub_dst = os.path.join(dst_dir, dname)
                    if _is_reparse_point(sub_src):
                        copy_failed = True
                        dirs[:] = []
                        break
                    sub_canon = _canonical_local_realpath(sub_src)
                    try:
                        if os.path.commonpath([shadow_canon, sub_canon]) != shadow_canon:
                            copy_failed = True
                            dirs[:] = []
                            break
                    except ValueError:
                        copy_failed = True
                        dirs[:] = []
                        break
                    if _paths_alias_or_overlap(sub_src, sub_dst, check_containment=True) or any(
                        _paths_alias_or_overlap(sub_src, pr, check_containment=True)
                        for pr in protected_roots
                    ):
                        copy_failed = True
                        dirs[:] = []
                        break
                    planned_dirs.append(sub_dst)

                if copy_failed:
                    dirs[:] = []
                    break

                for fname in files:
                    sf = os.path.join(root, fname)
                    df = os.path.join(dst_dir, fname)
                    if _is_reparse_point(sf):
                        copy_failed = True
                        dirs[:] = []
                        break
                    sf_canon = _canonical_local_realpath(sf)
                    try:
                        if os.path.commonpath([shadow_canon, sf_canon]) != shadow_canon:
                            copy_failed = True
                            dirs[:] = []
                            break
                    except ValueError:
                        copy_failed = True
                        dirs[:] = []
                        break
                    if _paths_alias_or_overlap(sf, df, check_containment=False):
                        copy_failed = True
                        dirs[:] = []
                        break
                    planned_files.append((sf, df))

                if copy_failed:
                    dirs[:] = []
                    break

            if copy_failed or walk_failed:
                continue

            # Phase 2: Materialize destination directories and migrate verified regular files.
            for dpath in planned_dirs:
                try:
                    if os.path.exists(dpath) and _is_reparse_point(dpath):
                        copy_failed = True
                        break
                    os.makedirs(dpath, exist_ok=True)
                    if _is_reparse_point(dpath):
                        copy_failed = True
                        break
                except OSError:
                    copy_failed = True
                    break

            if copy_failed or _paths_alias_or_overlap(
                shadow_compass, host_write_compass_dir, check_containment=True
            ):
                continue

            files_to_verify: List[Tuple[str, int, Optional[str]]] = []
            for sf, df in planned_files:
                try:
                    if _is_reparse_point(sf) or (os.path.exists(df) and _is_reparse_point(df)):
                        copy_failed = True
                        break
                    src_stat = os.lstat(sf)
                    if not stat.S_ISREG(src_stat.st_mode):
                        copy_failed = True
                        break
                    src_size = src_stat.st_size
                    if _paths_alias_or_overlap(sf, df, check_containment=False):
                        copy_failed = True
                        break
                    dst_exists = os.path.exists(df)
                    dst_size = os.path.getsize(df) if dst_exists else 0

                    if not dst_exists or (dst_size == 0 and src_size > 0):
                        src_hash = _sha256_file(sf)
                        shutil.copy2(sf, df)
                        files_to_verify.append((df, src_size, src_hash))
                    elif src_size == 0:
                        files_to_verify.append((df, dst_size, None))
                    else:
                        src_hash = _sha256_file(sf)
                        dst_hash = _sha256_file(df)
                        if src_hash == dst_hash:
                            files_to_verify.append((df, dst_size, dst_hash))
                        else:
                            # Check if either side matches an unedited bundled template file
                            bundled_hash: Optional[str] = None
                            if bundled_models_dir and os.path.isdir(bundled_models_dir):
                                rel_to_shadow = os.path.relpath(sf, shadow_compass)
                                parts = rel_to_shadow.split(os.sep)
                                if len(parts) >= 2 and parts[0].lower() == "models":
                                    bf = os.path.join(bundled_models_dir, *parts[1:])
                                    if os.path.isfile(bf) and not _is_reparse_point(bf):
                                        try:
                                            bundled_hash = _sha256_file(bf)
                                        except OSError:
                                            bundled_hash = None

                            if bundled_hash is not None and dst_hash == bundled_hash:
                                # Host has untouched bundled default; LocalCache has user's edit -> promote LocalCache
                                shutil.copy2(sf, df)
                                files_to_verify.append((df, src_size, src_hash))
                            elif bundled_hash is not None and src_hash == bundled_hash:
                                # LocalCache has untouched bundled default; Host has user's edit -> keep Host
                                files_to_verify.append((df, dst_size, dst_hash))
                            else:
                                dst_stat = os.stat(df)
                                if src_stat.st_mtime >= dst_stat.st_mtime:
                                    bak_path = _next_available_backup_path(df, ".host_bak")
                                    shutil.copy2(df, bak_path)
                                    files_to_verify.append((bak_path, dst_size, dst_hash))
                                    shutil.copy2(sf, df)
                                    files_to_verify.append((df, src_size, src_hash))
                                else:
                                    mig_path = _next_available_backup_path(
                                        df, ".localcache_migrated"
                                    )
                                    shutil.copy2(sf, mig_path)
                                    files_to_verify.append((mig_path, src_size, src_hash))
                                    files_to_verify.append((df, dst_size, dst_hash))
                except OSError:
                    copy_failed = True
                    break

            if copy_failed or walk_failed:
                continue

            # Phase 3: Byte-size + SHA-256 verification before shadow cleanup.
            all_verified = True
            for target_file, expected_size, expected_hash in files_to_verify:
                if (
                    not os.path.isfile(target_file)
                    or _is_reparse_point(target_file)
                    or os.path.getsize(target_file) != expected_size
                ):
                    all_verified = False
                    break
                if expected_hash is not None and _sha256_file(target_file) != expected_hash:
                    all_verified = False
                    break
            if not all_verified or _is_reparse_point(shadow_compass):
                continue

            shutil.rmtree(shadow_compass, ignore_errors=True)
            if os.path.exists(shadow_compass) and not _is_reparse_point(shadow_compass):
                subprocess.run(
                    ["cmd", "/c", "rmdir", "/S", "/Q", shadow_compass],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except (OSError, RecursionError):
            pass


def seed_user_models_dir(user_models_dir: str, bundled_models_dir: Optional[str] = None) -> None:
    """
    Initializes the user's `AppData/Roaming/COMPASS_AI/models` directory on the physical host
    filesystem and seeds default model folder templates (`config.json`, `data.json`) without
    overwriting any existing user files.
    """
    _clean_msix_localcache_shadow(user_models_dir, bundled_models_dir)
    write_root = resolve_host_write_path(user_models_dir)
    os.makedirs(write_root, exist_ok=True)

    readme_path = os.path.join(write_root, "README_MODELS_JA.txt")
    if not os.path.exists(readme_path):
        readme_text = (
            "======================================================================\n"
            " COMPASS AI - ユーザー自律管理モデルディレクトリ (AppData/Roaming)\n"
            "======================================================================\n\n"
            "このフォルダ (`models`) に各モデルのサブフォルダを配置すると、\n"
            "COMPASS AI が自動的に認識してローカル推論エンジンで読み込みます。\n\n"
            "【各モデルフォルダの構成】\n"
            "  models/\n"
            "    └── <任意のフォルダ名>/\n"
            "          ├── model.pth    (学習済み重みファイル)\n"
            "          ├── config.json  (モデル構造設定ファイル)\n"
            "          └── data.json    (メタデータ・対応タスク定義ファイル)\n\n"
            "※ 既存のテンプレートフォルダ (MORTM45D_80M_sft_gen 等) の中に\n"
            "   お手元の `model.pth` を配置するだけでそのままご使用いただけます。\n"
            "※ 新しくSFTした独自モデルを追加する場合も、同様に3ファイルを\n"
            "   新しいサブフォルダへ入れるだけで自動的に認識されます。\n"
        )
        try:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_text)
        except OSError:
            pass

    if bundled_models_dir and os.path.isdir(bundled_models_dir):
        for entry in os.listdir(bundled_models_dir):
            src_folder = os.path.join(bundled_models_dir, entry)
            if not os.path.isdir(src_folder):
                continue
            dst_folder = os.path.join(write_root, entry)
            os.makedirs(dst_folder, exist_ok=True)
            for fname in ("config.json", "data.json"):
                src_file = os.path.join(src_folder, fname)
                dst_file = os.path.join(dst_folder, fname)
                if os.path.isfile(src_file) and not os.path.exists(dst_file):
                    try:
                        shutil.copy2(src_file, dst_file)
                    except OSError:
                        pass

    # Clean up any MSIX LocalCache shadow so standard paths resolve to the physical host directory
    _clean_msix_localcache_shadow(user_models_dir, bundled_models_dir)


class ModelController:
    """
    Thread-safe Model Controller (MVC Model Layer Facade).
    Discovers models from `%APPDATA%\\COMPASS_AI\\models` (and fallback `./data/models`),
    manages hardware backend acceleration (CUDA / Apple Silicon MPS / CPU),
    caches active `AbstractModelRapper` instances, and runs the generation pipeline.
    """

    _PRE_CANCEL_TTL_SEC = 30.0
    _MAX_PRE_CANCEL_ENTRIES = 256

    def __init__(
        self,
        user_models_dir: Optional[str] = None,
        bundled_models_dir: Optional[str] = None,
        backend: Optional[IHardwareBackend] = None,
        max_cached_models: int = 4,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.bundled_models_dir = os.path.abspath(
            bundled_models_dir or os.path.join(base_dir, "data", "models")
        )
        self.user_models_dir = os.path.abspath(user_models_dir or get_user_models_dir())
        self.backend: IHardwareBackend = backend or HardwareBackendFactory.create(
            os.environ.get("COMPASS_DEVICE", "").strip() or "auto"
        )
        cache_limit_default = _env_int("COMPASS_MAX_LOADED_MODELS", max_cached_models)
        self.max_cached_models = max(1, _env_int("COMPASS_MODEL_CACHE_SIZE", cache_limit_default))
        self._catalog_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._cancel_lock = threading.Lock()
        self._pending_cancel_events: "OrderedDict[str, threading.Event]" = OrderedDict()
        self._anon_cancel_events: set[threading.Event] = set()
        self._pre_cancelled_jobs: "OrderedDict[str, float]" = OrderedDict()
        self._rapper_cache: "OrderedDict[str, AbstractModelRapper]" = OrderedDict()

        # Seed AppData/Roaming/COMPASS_AI/models with default templates if needed
        seed_user_models_dir(self.user_models_dir, self.bundled_models_dir)

        self.factory = ModelRapperFactory()
        self._register_default_rappers()

        self.model_info_dict: Dict[str, dict] = {}
        self.rescan_models()

    def _register_default_rappers(self) -> None:
        # Register by explicit model_name
        self.factory.register_rapper("MORTM4.5-Flash", MORTM45Rapper)
        self.factory.register_rapper("MORTM4.5-Pro", MORTM45Rapper)
        self.factory.register_rapper("MORTM4.5-Flash-Preview", MORTM45Rapper)
        self.factory.register_rapper("MORTM4.5-Pro-Preview", MORTM45Rapper)
        self.factory.register_rapper("MORTM4.5D-80M", MORTM45DRapper)
        self.factory.register_rapper("MORTM4.5D-80M-SFT-Gen", MORTM45DRapper)
        self.factory.register_rapper("MORTM4.5D-160M-SFT-Gen", MORTM45DRapper)
        self.factory.register_rapper("MORTM4.5E-A80M-E64", MORTM45DRapper)
        self.factory.register_rapper("MORTM4.5E-A80M-E64-SFT-Gen", MORTM45DRapper)

        # Register by tag.model so any custom user model folder dropped into AppData works automatically
        self.factory.register_tag_rapper("foundation", MORTM45DRapper)
        self.factory.register_tag_rapper("sft_gen", MORTM45DRapper)
        self.factory.register_tag_rapper("pretrained", MORTM45Rapper)
        self.factory.register_tag_rapper("generation", MORTM45Rapper)

    @staticmethod
    def _is_within_directory(child_path: str, parent_dir: str) -> bool:
        try:
            real_child = os.path.normcase(os.path.realpath(child_path))
            real_parent = os.path.normcase(os.path.realpath(parent_dir))
            return os.path.commonpath([real_child, real_parent]) == real_parent
        except ValueError:
            return False

    def rescan_models(self) -> Dict[str, dict]:
        """
        Scans both `self.user_models_dir` (`AppData/Roaming/COMPASS_AI/models`) and
        `self.bundled_models_dir` (`./data/models`).
        User AppData models take priority; if a model folder in AppData does not yet have
        `model.pth` but the bundled folder does, falls back to the folder containing `model.pth`.
        """
        with self._catalog_lock:
            _clean_msix_localcache_shadow(self.user_models_dir, self.bundled_models_dir)
            discovered: Dict[str, dict] = {}

            search_roots = []
            if os.path.isdir(self.bundled_models_dir):
                search_roots.append(("bundled", self.bundled_models_dir))
            if os.path.isdir(self.user_models_dir) and os.path.normcase(self.user_models_dir) != os.path.normcase(
                self.bundled_models_dir
            ):
                search_roots.append(("user_appdata", self.user_models_dir))

            for source_kind, root_dir in search_roots:
                try:
                    entries = sorted(os.listdir(root_dir))
                except OSError:
                    continue

                for folder_name in entries:
                    folder_path = os.path.join(root_dir, folder_name)
                    if not os.path.isdir(folder_path):
                        continue
                    if not self._is_within_directory(folder_path, root_dir):
                        continue

                    data_json_path = os.path.join(folder_path, "data.json")
                    config_json_path = os.path.join(folder_path, "config.json")
                    if not (os.path.isfile(data_json_path) and os.path.isfile(config_json_path)):
                        continue

                    try:
                        with open(data_json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        continue

                    model_name = data.get("model_name") or folder_name
                    has_weights = os.path.isfile(os.path.join(folder_path, "model.pth"))

                    prev = discovered.get(model_name)
                    if prev is not None and prev.get("has_weights") and not has_weights:
                        continue

                    info = dict(data)
                    info["model_name"] = model_name
                    info["model_folder_path"] = folder_path
                    info["source"] = source_kind
                    info["has_weights"] = has_weights
                    discovered[model_name] = info

            self.model_info_dict = discovered
            return copy.deepcopy(self.model_info_dict)

    def get_model_info(self, refresh: bool = False) -> List[dict]:
        """
        Returns public model metadata list for `POST /model_info`.
        Uses `_catalog_lock` (never blocked by GPU/CPU `_inference_lock`) and serves cached
        metadata in O(M) unless `refresh=True` (`POST /rescan_models`).
        """
        with self._catalog_lock:
            if refresh or not self.model_info_dict:
                self.rescan_models()
            clean_info_list: List[dict] = []
            for info in self.model_info_dict.values():
                clean_data = {
                    k: copy.deepcopy(v)
                    for k, v in info.items()
                    if k not in ("model_folder_path",)
                }
                clean_info_list.append(clean_data)
            return clean_info_list

    def get_hardware_status(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Returns active hardware acceleration status and user models directory path.
        Uses `_catalog_lock` without blocking on `_inference_lock` or redundant disk scans.
        """
        with self._catalog_lock:
            if refresh or not self.model_info_dict:
                self.rescan_models()
            return {
                "device": str(self.backend.device),
                "dtype": str(self.backend.compute_dtype),
                "backend_class": self.backend.__class__.__name__,
                "user_models_dir": resolve_physical_models_dir_for_shell(self.user_models_dir),
                "bundled_models_dir": self.bundled_models_dir,
                "available_models_count": len(self.model_info_dict),
                "models_with_weights": [
                    name for name, info in self.model_info_dict.items() if info.get("has_weights")
                ],
            }

    @staticmethod
    def get_generation_progress(job_id: Optional[str] = None) -> Dict[str, Any]:
        return GLOBAL_GENERATION_PROGRESS.snapshot(job_id=job_id)

    def _prune_pre_cancelled_locked(self, now: float) -> None:
        cutoff = now - self._PRE_CANCEL_TTL_SEC
        while self._pre_cancelled_jobs:
            oldest_job, ts = next(iter(self._pre_cancelled_jobs.items()))
            if ts <= cutoff or len(self._pre_cancelled_jobs) > self._MAX_PRE_CANCEL_ENTRIES:
                self._pre_cancelled_jobs.pop(oldest_job, None)
            else:
                break

    def register_cancel_event(
        self,
        ev: Optional[Union[threading.Event, str]] = None,
        job_id: Optional[str] = None,
    ) -> threading.Event:
        """
        Registers a request cancellation event BEFORE upload parsing / `_inference_lock` wait
        so `cancel_active_generation(job_id)` cancels only the targeted request (`[R2-HIGH-07]`)
        or all active/queued requests when called without `job_id`.
        """
        if isinstance(ev, str) and job_id is None:
            job_id = ev
            ev = None
        target = ev if isinstance(ev, threading.Event) else threading.Event()
        clean_id = (job_id or "").strip()
        now = time.monotonic()
        with self._cancel_lock:
            self._prune_pre_cancelled_locked(now)
            if clean_id:
                if clean_id in self._pre_cancelled_jobs:
                    self._pre_cancelled_jobs.pop(clean_id, None)
                    target.set()
                self._pending_cancel_events[clean_id] = target
            else:
                self._anon_cancel_events.add(target)
        return target

    def unregister_cancel_event(
        self,
        ev: Optional[threading.Event],
        job_id: Optional[str] = None,
    ) -> None:
        if ev is None and not job_id:
            return
        clean_id = (job_id or "").strip()
        with self._cancel_lock:
            if ev is not None:
                self._anon_cancel_events.discard(ev)
            if clean_id and self._pending_cancel_events.get(clean_id) in (ev, None if ev is None else ev):
                if ev is None or self._pending_cancel_events.get(clean_id) is ev:
                    self._pending_cancel_events.pop(clean_id, None)
            if ev is not None:
                stale_ids = [jid for jid, registered in self._pending_cancel_events.items() if registered is ev]
                for jid in stale_ids:
                    self._pending_cancel_events.pop(jid, None)

    def cancel_active_generation(self, job_id: Optional[str] = None) -> bool:
        """
        Signals cooperative cancellation.
        - When `job_id` is provided: cancels ONLY that `job_id` (and records a short-TTL pre-cancel
          tombstone if `/cancel_generate` arrives slightly before `/generate` registers `job_id`),
          never cancelling sibling or subsequent jobs (`[R2-HIGH-07]`).
        - When `job_id` is omitted: cancels all currently active and queued requests.
        """
        clean_id = (job_id or "").strip()
        now = time.monotonic()
        with self._cancel_lock:
            self._prune_pre_cancelled_locked(now)
            if clean_id:
                ev = self._pending_cancel_events.get(clean_id)
                if ev is not None:
                    ev.set()
                    return True
                self._pre_cancelled_jobs[clean_id] = now
                self._prune_pre_cancelled_locked(now)
                return False

            events = list(self._pending_cancel_events.values()) + list(self._anon_cancel_events)
        if not events:
            return False
        for ev in events:
            ev.set()
        return True

    def _get_or_create_rapper(self, model_type: str) -> AbstractModelRapper:
        with self._inference_lock:
            with self._catalog_lock:
                if model_type not in self.model_info_dict:
                    self.rescan_models()
                if model_type not in self.model_info_dict:
                    available = ", ".join(sorted(self.model_info_dict.keys()))
                    raise ValueError(
                        f"指定されたモデル '{model_type}' は存在しません。"
                        f"AppDataモデルフォルダ を確認してください。利用可能: [{available}]"
                    )
                info = copy.deepcopy(self.model_info_dict[model_type])

            folder_path = info["model_folder_path"]
            pth_path = os.path.join(folder_path, "model.pth")
            pth_mtime = os.path.getmtime(pth_path) if os.path.exists(pth_path) else 0.0
            cache_key = f"{model_type}::{folder_path}::{pth_mtime}"

            if cache_key in self._rapper_cache:
                self._rapper_cache.move_to_end(cache_key)
                return self._rapper_cache[cache_key]

            rapper = self.factory.create_rapper(info, backend=self.backend)
            self._rapper_cache[cache_key] = rapper
            while len(self._rapper_cache) > self.max_cached_models:
                _, evicted = self._rapper_cache.popitem(last=False)
                del evicted
                self.backend.empty_cache()

            return rapper

    def generate(
        self,
        past_midi_path: Optional[str],
        const_midi_path: Optional[str],
        future_midi_path: Optional[str],
        meta: GenerateMeta,
        cancel_event: Optional[threading.Event] = None,
        job_id: Optional[str] = None,
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        """
        Executes `_get_or_create_rapper` -> `preprocessing` -> `generate` -> `postprocessing`
        within `_inference_lock` (decoupled from `_catalog_lock` so `/health`, `/hardware_info`,
        and `/model_info` remain < 2ms during generation), supports cooperative `cancel_event`
        abort within 1 token step, and scopes `GLOBAL_GENERATION_PROGRESS` by `run_id`.
        """
        eff_cancel = self.register_cancel_event(cancel_event, job_id=job_id)
        try:
            if eff_cancel.is_set():
                raise GenerationCancelledError("Generation cancelled before acquiring inference lock.")

            # Validate unified meter across `meta` and context MIDIs BEFORE acquiring
            # `_inference_lock` or loading/evicting multi-hundred-MB `.pth` weights (`[R4-MED-02]`).
            AbstractModelRapper.resolve_unified_time_signature(
                meta, (past_midi_path, const_midi_path, future_midi_path)
            )

            with self._inference_lock:
                if eff_cancel.is_set():
                    raise GenerationCancelledError("Generation cancelled while queued.")

                run_id = uuid.uuid4().hex
                GLOBAL_GENERATION_PROGRESS.start_run(
                    run_id,
                    f"モデル {meta.model_type} を準備しています...",
                    job_id=job_id,
                )
                try:
                    rapper = self._get_or_create_rapper(meta.model_type)
                    if eff_cancel.is_set():
                        raise GenerationCancelledError("Generation cancelled after model load.")

                    GLOBAL_GENERATION_PROGRESS.update(
                        active=True,
                        stage="preprocessing",
                        message="コンテキストMIDIのトークン変換とプロンプト構築を実行中...",
                        run_id=run_id,
                    )
                    kwargs = rapper.preprocessing(past_midi_path, const_midi_path, future_midi_path, meta)
                    if eff_cancel.is_set():
                        raise GenerationCancelledError("Generation cancelled during preprocessing.")
                    kwargs["cancel_event"] = eff_cancel
                    kwargs["run_id"] = run_id

                    GLOBAL_GENERATION_PROGRESS.update(
                        active=True,
                        stage="decoding",
                        message=(
                            f"{meta.model_type} で CoT構造推論・トークン生成を開始しました..."
                            if getattr(meta, "thinking", False)
                            else f"{meta.model_type} で自己回帰トークン生成を開始しました..."
                        ),
                        run_id=run_id,
                    )
                    generated_data_kwargs = rapper.generate(**kwargs)

                    if eff_cancel.is_set():
                        raise GenerationCancelledError("Generation cancelled before postprocessing.")

                    GLOBAL_GENERATION_PROGRESS.update(
                        active=True,
                        stage="postprocessing",
                        message="生成されたトークン列を Standard MIDI ファイルへ変換中...",
                        run_id=run_id,
                    )
                    save_dir = tempfile.mkdtemp(prefix="compass_gen_")
                    try:
                        paths = rapper.postprocessing(save_dir, **generated_data_kwargs)
                        if eff_cancel.is_set():
                            raise GenerationCancelledError("Generation cancelled during postprocessing.")
                        reason = getattr(rapper, "last_reason", None)

                        if paths and isinstance(paths[0], (dict, list)):
                            json_path = os.path.join(save_dir, "generated_data.json")
                            with open(json_path, "w", encoding="utf-8") as f:
                                json.dump(paths, f, ensure_ascii=False, indent=2)
                            if eff_cancel.is_set():
                                raise GenerationCancelledError("Generation cancelled before returning metadata.")
                            GLOBAL_GENERATION_PROGRESS.update(
                                active=False,
                                stage="completed",
                                message="メタデータ推論が完了しました。",
                                run_id=run_id,
                            )
                            return json_path, "application/json", reason

                        zip_path = os.path.join(save_dir, "generated_midis.zip")
                        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                            for file_path in paths:
                                if os.path.exists(file_path):
                                    zipf.write(file_path, os.path.basename(file_path))

                        if eff_cancel.is_set():
                            raise GenerationCancelledError("Generation cancelled before returning MIDI archive.")

                        GLOBAL_GENERATION_PROGRESS.update(
                            active=False,
                            stage="completed",
                            message="MIDIアーカイブの生成が完了しました。",
                            run_id=run_id,
                        )
                        return zip_path, "application/zip", reason
                    except BaseException:
                        shutil.rmtree(save_dir, ignore_errors=True)
                        raise
                except GenerationCancelledError:
                    GLOBAL_GENERATION_PROGRESS.update(
                        active=False,
                        stage="cancelled",
                        message="生成処理を中断しました。",
                        run_id=run_id,
                    )
                    raise
                except BaseException:
                    GLOBAL_GENERATION_PROGRESS.update(
                        active=False,
                        stage="error",
                        message="生成処理が中断されました。",
                        run_id=run_id,
                    )
                    raise
        finally:
            self.unregister_cancel_event(eff_cancel, job_id=job_id)
