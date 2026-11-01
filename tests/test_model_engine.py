"""
Comprehensive Physical Runtime Verification & Benchmark Suite for COMPASS AI Model Layer (`tests/test_model_engine.py`).
Tests:
1. Zero external research dependency (`flash_attn`, `loralib`, `music21`, `pylab` not required).
2. Autonomous `AppData/Roaming/COMPASS_AI/models` host filesystem persistence (including MSIX VFS bypass) and custom drop-in.
3. Mathematical equivalence of in-place LoRA folding (`LoRALinear` + 64-expert `MoE` LoRA + removal of `self.experts` + pre-folded loading).
4. 1-measure & multi-measure context MIDI extraction (verifying Measure 0 notes and 1-bar MIDIs are never dropped).
5. `omega_converter` vocabulary bounds (`< 650`) and ALiBi (`use_rope=False`) + `key_padding_mask` compatibility.
6. All 5 SFT-Gen tasks (`meta`, `meta_past`, `meta_future`, `infill`, `inst_comp`) + Foundation (`unconditional`) across CPU and CUDA.
7. Deterministic CoT (`<thinking>` + `<MGEN> <SYSTEM>`), 2-Stage Temperature, spurious `<MGEN>` resilience, and `X-Generation-Reason`.
8. FastAPI HTTP endpoints (`/health`, `/hardware_info`, `/model_info`, `/rescan_models`, `/generate`) & security guardrails (`CWE-502`, DoS bounds).
9. Physical tokens/sec and key estimation latency benchmarks.
"""

import atexit
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch
import zipfile

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pretty_midi
from pydantic import ValidationError
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Sandbox `COMPASS_MODELS_DIR` BEFORE importing `app` (`app.py:196` instantiates `ModelController()` at import time)
# so importing or running tests on a clean host NEVER seeds or touches real `%APPDATA%\COMPASS_AI\models` ([R3-HIGH-02]).
_IMPORT_SANDBOX_ROOT = tempfile.mkdtemp(prefix="compass_test_import_")
atexit.register(shutil.rmtree, _IMPORT_SANDBOX_ROOT, True)
_IMPORT_SANDBOX_DIR = os.path.join(_IMPORT_SANDBOX_ROOT, "COMPASS_AI", "models")
os.environ.setdefault("COMPASS_MODELS_DIR", _IMPORT_SANDBOX_DIR)

from app import SlidingWindowRateLimiter, app, model_controller
from model import (
    ModelController,
    get_user_models_dir,
    resolve_host_write_path,
    seed_user_models_dir,
)
from models.mortm.mortm45 import MORTM45Rapper
from mortm.models.modules.backend import (
    CpuHardwareBackend,
    CudaHardwareBackend,
    HardwareBackendFactory,
)
from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.progress import (
    GenerationCancelledError,
    GenerationProgressTracker,
)
from mortm.models.mortm import MORTM
from mortm.train.tokenizer import (
    TO_TOKEN,
    Tokenizer,
    get_token_converter_pro,
    omega_converter,
)
from mortm.utils.key import get_key_Polling
from rapper import GenerateMeta


def create_sample_midi_bytes(inst_program: int = 0, num_measures: int = 2, tempo: float = 120.0) -> bytes:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=inst_program)
    sec_per_beat = 60.0 / tempo
    total_beats = num_measures * 4
    pitches = [60, 64, 67, 72]
    for b in range(total_beats):
        start = b * sec_per_beat
        end = start + sec_per_beat * 0.8
        pitch = pitches[b % len(pitches)]
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def test_1_appdata_models_dir_and_custom_dropin():
    print("\n--- [TEST 1] AppData/Roaming/COMPASS_AI/models Isolated Host Sandbox & Custom Drop-In ---")
    from unittest.mock import patch
    from model import resolve_physical_models_dir_for_shell

    sandbox_root = tempfile.mkdtemp(prefix="compass_appdata_sandbox_")
    try:
        fake_roaming = os.path.join(sandbox_root, "Roaming")
        fake_local = os.path.join(sandbox_root, "Local")
        os.makedirs(fake_roaming, exist_ok=True)
        os.makedirs(fake_local, exist_ok=True)

        env_overrides = {
            "APPDATA": fake_roaming,
            "LOCALAPPDATA": fake_local,
            "XDG_CONFIG_HOME": fake_roaming,
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            os.environ.pop("COMPASS_MODELS_DIR", None)
            user_dir = get_user_models_dir()
            bundled_dir = os.path.join(ROOT_DIR, "data", "models")
            seed_user_models_dir(user_dir, bundled_dir)

            host_write_dir = resolve_host_write_path(user_dir)
            shell_dir = resolve_physical_models_dir_for_shell(user_dir)
            assert os.path.isdir(host_write_dir), f"Host write dir does not exist: {host_write_dir}"
            assert os.path.isdir(shell_dir), f"Shell physical dir does not exist: {shell_dir}"
            assert "COMPASS_AI" in user_dir and user_dir.endswith("models")
            assert os.path.isfile(os.path.join(host_write_dir, "README_MODELS_JA.txt"))
            assert os.path.isfile(os.path.join(host_write_dir, "MORTM45D_80M_sft_gen", "config.json"))
            assert os.path.isfile(os.path.join(host_write_dir, "MORTM45E_A80M_E64_sft_gen", "config.json"))

            # Verify no MSIX LocalCache shadow directory exists in sandbox
            shadows = glob.glob(
                os.path.join(
                    fake_local,
                    "Packages",
                    "PythonSoftwareFoundation.Python*",
                    "LocalCache",
                    "Roaming",
                    "COMPASS_AI",
                )
            )
            assert len(shadows) == 0, f"Stale MSIX LocalCache shadow directory detected: {shadows}"

            # Simulate a user autonomously dropping a custom SFT model folder into AppData/Roaming/COMPASS_AI/models
            custom_folder = os.path.join(host_write_dir, "UserCustom_MORTM45D_SFT")
            os.makedirs(custom_folder, exist_ok=True)
            tiny_cfg = {
                "vocab_size": 700,
                "d_model": 128,
                "dim_feedforward": 256,
                "num_heads": 4,
                "d_layer": 2,
                "dropout": 0.0,
                "position_length": 2048,
                "min_length": 16,
                "use_moe_decoder": False,
                "debug_attention": False,
                "norm_type": "rmsnorm",
                "use_silu": True,
                "use_bias": False,
                "use_rope": True,
                "use_attn_lora": True,
                "use_ffn_lora": True,
                "lora_r": 4,
                "lora_alpha": 8,
            }
            tiny_data = {
                "model_name": "UserCustom-MORTM4.5D-SFT",
                "description": "ユーザーがAppData/Roaming/COMPASS_AI/modelsに配置したカスタムSFTモデル",
                "tag": {
                    "instruments": ["PIANO", "SAX"],
                    "model": "sft_gen",
                    "task": ["meta", "meta_past", "meta_future", "infill", "inst_comp"],
                    "type": "mortm",
                    "version": "4.5",
                },
                "rule": {"input_midi": True},
            }
            with open(os.path.join(custom_folder, "config.json"), "w", encoding="utf-8") as f:
                json.dump(tiny_cfg, f)
            with open(os.path.join(custom_folder, "data.json"), "w", encoding="utf-8") as f:
                json.dump(tiny_data, f)

            cpu_backend = CpuHardwareBackend()
            args = MORTMArgs(os.path.join(custom_folder, "config.json"))
            temp_model = MORTM(args, backend=cpu_backend)
            torch.save(temp_model.state_dict(), os.path.join(custom_folder, "model.pth"))

            controller = ModelController(user_models_dir=host_write_dir, bundled_models_dir=bundled_dir)
            discovered = controller.rescan_models()
            assert "UserCustom-MORTM4.5D-SFT" in discovered
            assert discovered["UserCustom-MORTM4.5D-SFT"]["has_weights"] is True
            assert discovered["UserCustom-MORTM4.5D-SFT"]["source"] == "user_appdata"
            hw_status = controller.get_hardware_status()
            assert os.path.isdir(hw_status["user_models_dir"])
            print(f"  [PASS] Isolated Sandbox AppData Dir: {user_dir}")
            print("  [PASS] Custom model dropped into AppData discovered with has_weights=True and 0 live-AppData side effects")
    finally:
        shutil.rmtree(sandbox_root, ignore_errors=True)


def test_2_lora_folding_mathematical_equivalence():
    print("\n--- [TEST 2] Mathematical Equivalence of In-Place LoRA Folding (Dense & 64-Expert MoE) ---")
    cpu_backend = CpuHardwareBackend()

    for label, use_moe in [("Dense (MORTM4.5D)", False), ("64-Expert MoE (MORTM4.5E)", True)]:
        cfg_dict = {
            "vocab_size": 700,
            "d_model": 128,
            "dim_feedforward": 64 if use_moe else 256,
            "num_heads": 4,
            "d_layer": 2,
            "dropout": 0.0,
            "position_length": 1024,
            "num_experts": 16 if use_moe else 8,
            "topk_experts": 4 if use_moe else 2,
            "num_groups": 1,
            "topk_groups": 1,
            "use_moe_decoder": use_moe,
            "debug_attention": False,
            "norm_type": "rmsnorm",
            "use_silu": True,
            "use_bias": False,
            "use_rope": True,
            "use_attn_lora": True,
            "use_ffn_lora": True,
            "lora_r": 4,
            "lora_alpha": 8,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(cfg_dict, tf)
            cfg_path = tf.name

        try:
            args = MORTMArgs(cfg_path)
            model = MORTM(args, backend=cpu_backend).eval()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "lora_B" in name:
                        param.normal_(0.0, 0.05)

            x = torch.randint(1, 600, (2, 16), dtype=torch.long)
            with torch.no_grad():
                out_unmerged = model.forward(x, is_save_cache=False)
                merged_count = model.merge_lora()
                out_merged = model.forward(x, is_save_cache=False)

            if use_moe:
                for layer in model.decoder.layers:
                    assert layer.ffn.stacked is True
                    assert not hasattr(layer.ffn, "experts"), "MoE.experts should be deleted after merge_lora_inplace"

            max_diff = (out_unmerged - out_merged).abs().max().item()
            print(f"  [{label}] Merged LoRA modules: {merged_count} | Max logit diff: {max_diff:.2e}")
            assert merged_count > 0
            assert max_diff < 1e-4, f"LoRA merge mismatch for {label}: {max_diff}"
        finally:
            os.remove(cfg_path)


def test_2b_1measure_midi_and_omega_vocab_and_alibi_mask():
    print("\n--- [TEST 2B] 1-Measure MIDI Context, omega_converter Bounds, & ALiBi Mask ---")
    # 1. Verify omega_converter token IDs stay strictly < 650
    tok_omega = Tokenizer(omega_converter(TO_TOKEN))
    max_omega_id = max(tok_omega.tokens.values())
    assert max_omega_id < 650, f"omega_converter max token ID {max_omega_id} exceeds vocab_size=650"
    assert tok_omega.get("<GEN_MEASURE_COUNT_8>") < 650

    # 2. Verify ALiBi (use_rope=False) with key_padding_mask containing False
    cfg_alibi = {
        "vocab_size": 650,
        "d_model": 128,
        "dim_feedforward": 256,
        "num_heads": 4,
        "d_layer": 2,
        "dropout": 0.0,
        "position_length": 1024,
        "use_moe_decoder": False,
        "debug_attention": False,
        "norm_type": "rmsnorm",
        "use_silu": True,
        "use_bias": False,
        "use_rope": False,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(cfg_alibi, tf)
        cfg_path = tf.name
    try:
        m_alibi = MORTM(MORTMArgs(cfg_path), backend=CpuHardwareBackend()).eval()
        src = torch.tensor([[1, 20, 30, 0], [1, 20, 0, 0]], dtype=torch.long)
        pad_mask = src != 0
        with torch.no_grad():
            logits = m_alibi.forward(src, padding_mask=pad_mask, is_causal=True, is_save_cache=False)
        assert not torch.isnan(logits).any()
    finally:
        os.remove(cfg_path)
    print("  [PASS] omega_converter (<650) and ALiBi padding mask verified")


def test_3_all_5_sft_tasks_and_foundation_across_backends():
    print("\n--- [TEST 3] All 5 SFT Tasks (Including 1-Measure MIDI) + Foundation + CoT across Backends ---")
    torch.manual_seed(42)
    temp_models_root = tempfile.mkdtemp(prefix="compass_test_models_")
    temp_midi_dir = tempfile.mkdtemp(prefix="compass_test_midis_")

    try:
        # Use a 1-measure MIDI for past_path to explicitly verify [HIGH-1] (1-measure MIDI not dropped!)
        past_path = os.path.join(temp_midi_dir, "past_1bar.mid")
        future_path = os.path.join(temp_midi_dir, "future_2bar.mid")
        const_path = os.path.join(temp_midi_dir, "const_2bar.mid")
        with open(past_path, "wb") as f:
            f.write(create_sample_midi_bytes(inst_program=0, num_measures=1))
        with open(future_path, "wb") as f:
            f.write(create_sample_midi_bytes(inst_program=0, num_measures=2))
        with open(const_path, "wb") as f:
            f.write(create_sample_midi_bytes(inst_program=65, num_measures=2))

        models_to_create = [
            ("Test_Foundation", "foundation", False),
            ("Test_SFT_Dense", "sft_gen", False),
            ("Test_SFT_MoE", "sft_gen", True),
        ]
        for m_name, tag_model, use_moe in models_to_create:
            m_dir = os.path.join(temp_models_root, m_name)
            os.makedirs(m_dir, exist_ok=True)
            cfg = {
                "vocab_size": 700,
                "d_model": 128,
                "dim_feedforward": 64 if use_moe else 256,
                "num_heads": 4,
                "d_layer": 2,
                "dropout": 0.0,
                "position_length": 1024,
                "min_length": 16,
                "num_experts": 16 if use_moe else 8,
                "topk_experts": 4 if use_moe else 2,
                "use_moe_decoder": use_moe,
                "debug_attention": False,
                "norm_type": "rmsnorm",
                "use_silu": True,
                "use_bias": False,
                "use_rope": True,
                "use_attn_lora": tag_model == "sft_gen",
                "use_ffn_lora": tag_model == "sft_gen",
                "lora_r": 4,
                "lora_alpha": 8,
            }
            data = {
                "model_name": m_name,
                "description": f"Test {m_name}",
                "tag": {
                    "instruments": ["PIANO", "SAX"],
                    "model": tag_model,
                    "task": ["meta", "meta_past", "meta_future", "infill", "inst_comp"]
                    if tag_model == "sft_gen"
                    else ["Generate"],
                    "type": "mortm",
                    "version": "4.5",
                },
                "rule": {"input_midi": tag_model == "sft_gen"},
            }
            with open(os.path.join(m_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            with open(os.path.join(m_dir, "data.json"), "w", encoding="utf-8") as f:
                json.dump(data, f)

        backends = [CpuHardwareBackend()]
        if torch.cuda.is_available():
            backends.append(CudaHardwareBackend())

        for backend in backends:
            print(f"  Testing on backend: {backend.device} (dtype={backend.compute_dtype})")
            controller = ModelController(
                user_models_dir=temp_models_root,
                bundled_models_dir=temp_models_root,
                backend=backend,
            )

            for m_name in ("Test_Foundation", "Test_SFT_Dense", "Test_SFT_MoE"):
                rapper = controller._get_or_create_rapper(m_name)
                orig_fn = rapper.model.top_sampling_measure_kv_cache

                def _fast_sample(*args, _orig=orig_fn, **kwargs):
                    kwargs["max_tokens"] = 24
                    return _orig(*args, **kwargs)

                rapper.model.top_sampling_measure_kv_cache = _fast_sample

            # Verify 1-measure past_midi produces <PAST_M> token in preprocessing!
            dense_rapper = controller._get_or_create_rapper("Test_SFT_Dense")
            prep = dense_rapper.preprocessing(
                past_path,
                None,
                None,
                GenerateMeta(model_type="Test_SFT_Dense", task="meta_past", genfield_measure=None),
            )
            past_m_id = prep["tokenizer"].get("<PAST_M>")
            assert past_m_id in prep["sequence"][0], "1-measure past_midi failed to emit <PAST_M> in prompt!"

            # 1) Foundation unconditional
            zip_p, media_t, reason = controller.generate(
                None, None, None, GenerateMeta(model_type="Test_Foundation", num_gems=1)
            )
            assert os.path.isfile(zip_p) and media_t == "application/zip"
            assert reason and reason[0]["source"] == "foundation" and reason[0]["task"] == "unconditional"
            shutil.rmtree(os.path.dirname(zip_p), ignore_errors=True)

            # 2) All 5 SFT tasks on Dense and MoE
            sft_cases = [
                ("meta", None, None, None, False, "request"),
                ("meta_past", past_path, None, None, True, "model_cot"),
                ("meta_future", None, None, future_path, True, "model_cot"),
                ("infill", past_path, None, future_path, True, "model_cot"),
                ("inst_comp", None, const_path, None, True, "model_cot"),
            ]
            for model_id in ("Test_SFT_Dense", "Test_SFT_MoE"):
                for task_name, p_mid, c_mid, f_mid, thinking_flag, expected_src in sft_cases:
                    meta_req = GenerateMeta(
                        model_type=model_id,
                        task=task_name,
                        program=["PIANO", "SAX"],
                        genfield_measure=2 if task_name == "meta" else None,
                        gen_note_dense={"PIANO": 5, "SAX": 6},
                        genre=["jazz", "soundtrack"],
                        key="CM",
                        thinking=thinking_flag,
                        cot_temperature=0.1,
                        temperature=1.0,
                        num_gems=1,
                    )
                    zip_p, media_t, reason = controller.generate(p_mid, c_mid, f_mid, meta_req)
                    assert os.path.isfile(zip_p), f"Missing zip output for {model_id} / {task_name}"
                    assert reason is not None and len(reason) == 1
                    assert reason[0]["task"] == task_name
                    assert reason[0]["source"] == expected_src
                    with zipfile.ZipFile(zip_p, "r") as zf:
                        assert "output_0.mid" in zf.namelist()
                    shutil.rmtree(os.path.dirname(zip_p), ignore_errors=True)
                    print(f"    [OK] {model_id} | task={task_name:<11} | reason.source={reason[0]['source']}")
    finally:
        shutil.rmtree(temp_models_root, ignore_errors=True)
        shutil.rmtree(temp_midi_dir, ignore_errors=True)


def test_4_fastapi_endpoints_and_security_guardrails():
    print("\n--- [TEST 4] FastAPI Endpoints (/health, /model_info, /rescan_models, /generate) & Security Guardrails ---")
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    health_data = r_health.json()
    assert health_data["status"] == "ok"
    assert health_data["models_dir_ready"] is True
    assert "user_models_dir" not in health_data

    r_hw_init = client.get("/hardware_info")
    assert r_hw_init.status_code == 200
    assert "COMPASS_AI" in r_hw_init.json()["user_models_dir"]

    r_prog = client.get("/generation_progress")
    assert r_prog.status_code == 200
    prog_data = r_prog.json()
    for key in ("active", "stage", "generated_tokens", "max_tokens", "message"):
        assert key in prog_data, f"Missing {key} in /generation_progress"

    # Verify Legacy MORTM 4.5 task alias compatibility with SFT-Gen UI task names
    assert MORTM45Rapper._normalize_task("meta") == "Meta2MIDI"
    assert MORTM45Rapper._normalize_task("meta_past") == "Meta2MIDI"
    assert MORTM45Rapper._normalize_task("meta_future") == "Meta2MIDI"
    assert MORTM45Rapper._normalize_task("infill") == "Meta2MIDI"
    assert MORTM45Rapper._normalize_task("inst_comp") == "Chord2MIDI"

    # Verify CORS blocks `Origin: null` and allows `Origin: compass-app://studio`
    r_cors_null = client.options(
        "/generate",
        headers={
            "Origin": "null",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r_cors_null.headers.get("access-control-allow-origin") != "null"

    r_cors_app = client.options(
        "/generate",
        headers={
            "Origin": "compass-app://studio",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r_cors_app.headers.get("access-control-allow-origin") == "compass-app://studio"

    r_info = client.post("/model_info")
    assert r_info.status_code == 200
    models_list = r_info.json()
    model_names = {m["model_name"] for m in models_list}
    for expected in (
        "MORTM4.5D-80M",
        "MORTM4.5D-80M-SFT-Gen",
        "MORTM4.5D-160M-SFT-Gen",
        "MORTM4.5E-A80M-E64",
        "MORTM4.5E-A80M-E64-SFT-Gen",
    ):
        assert expected in model_names, f"Missing expected model in /model_info: {expected}"

    # Security guardrail: reject non-MIDI file masquerading as .mid
    bad_files = {"past_midi": ("malicious.mid", b"NOT_A_MIDI_HEADER", "audio/midi")}
    r_bad = client.post(
        "/generate",
        data={"meta": json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta_past"})},
        files=bad_files,
    )
    assert r_bad.status_code == 400
    assert "MThd" in r_bad.json()["detail"]

    # Security guardrail: Layer-3 SMF binary check rejects truncated MThd without MTrk chunk
    truncated_mthd = {
        "past_midi": ("truncated.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0", "audio/midi")
    }
    r_trunc = client.post(
        "/generate",
        data={"meta": json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta_past"})},
        files=truncated_mthd,
    )
    assert r_trunc.status_code == 400
    assert "MTrk" in r_trunc.json()["detail"]

    # Security guardrail: SlidingWindowRateLimiter enforces HTTP 429 + Retry-After header
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0)
    limiter.check_or_raise("127.0.0.1", "rate limit exceeded")
    limiter.check_or_raise("127.0.0.1", "rate limit exceeded")
    try:
        limiter.check_or_raise("127.0.0.1", "rate limit exceeded")
        raise AssertionError("Expected HTTPException(429) on 3rd request")
    except HTTPException as exc:
        assert exc.status_code == 429
        assert exc.headers is not None and "Retry-After" in exc.headers

    # Security guardrail: reject invalid extension
    bad_ext = {"past_midi": ("script.exe", b"MThd0000", "application/octet-stream")}
    r_ext = client.post(
        "/generate",
        data={"meta": json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta_past"})},
        files=bad_ext,
    )
    assert r_ext.status_code == 400

    # Security guardrail: DoS & negative slice bounds in GenerateMeta (both Python & HTTP /generate)
    for invalid_payload in [
        {"model_type": "MORTM4.5D-80M-SFT-Gen", "num_gems": 1000},
        {"model_type": "MORTM4.5D-80M-SFT-Gen", "measure_count": -5},
        {"model_type": "MORTM4.5D-80M-SFT-Gen", "tempo": 99999},
    ]:
        try:
            GenerateMeta(**invalid_payload)
            raise AssertionError(f"Expected ValidationError for {invalid_payload}")
        except ValidationError:
            pass
        r_inv = client.post("/generate", data={"meta": json.dumps(invalid_payload)})
        assert r_inv.status_code in (400, 422), f"Expected 400/422 over HTTP, got {r_inv.status_code}"

    # R2-1 Regression Guard: Windows .bat files MUST be CRLF without BOM; .ps1 files MUST be UTF-8 BOM + CRLF
    for bat_name in ("install_compass_ai.bat", "start_compass_ai.bat"):
        with open(os.path.join(ROOT_DIR, bat_name), "rb") as fh_bat:
            raw_bat = fh_bat.read()
        assert not raw_bat.startswith(b"\xef\xbb\xbf"), f"{bat_name} must not have UTF-8 BOM"
        assert raw_bat.count(b"\r\n") > 0 and (raw_bat.count(b"\n") - raw_bat.count(b"\r\n")) == 0, (
            f"{bat_name} must use pure CRLF line endings"
        )
    for ps1_rel in (os.path.join("installer", "install_and_run.ps1"), os.path.join("scripts", "verify_build_integrity.ps1")):
        with open(os.path.join(ROOT_DIR, ps1_rel), "rb") as fh_ps1:
            raw_ps1 = fh_ps1.read()
        assert raw_ps1.startswith(b"\xef\xbb\xbf"), f"{ps1_rel} must start with UTF-8 BOM for PowerShell 5.1"
        assert raw_ps1.count(b"\r\n") > 0 and (raw_ps1.count(b"\n") - raw_ps1.count(b"\r\n")) == 0, (
            f"{ps1_rel} must use pure CRLF line endings"
        )
    with open(os.path.join(ROOT_DIR, "installer", "install_and_run.ps1"), "r", encoding="utf-8-sig") as fh_inst:
        inst_ps1_text = fh_inst.read()
    assert r"& cmd.exe /d /c .\start_compass_ai.bat" in inst_ps1_text, (
        "install_and_run.ps1 must invoke .\\start_compass_ai.bat without embedded quotes for spaces/& safety"
    )
    if os.name == "nt":
        import subprocess
        adv_parent = tempfile.mkdtemp(prefix="compass_ps1_space_")
        try:
            adv_dir = os.path.join(adv_parent, "Dir With Spaces & (Parens)!")
            os.makedirs(adv_dir, exist_ok=True)
            with open(os.path.join(adv_dir, "start_compass_ai.bat"), "wb") as bf:
                bf.write(b"@echo off\r\necho SPACE_AMP_BAT_OK\r\nexit /b 0\r\n")
            ps_probe = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f"Set-Location -LiteralPath '{adv_dir}'; & cmd.exe /d /c .\\start_compass_ai.bat; exit $LASTEXITCODE",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert ps_probe.returncode == 0 and "SPACE_AMP_BAT_OK" in ps_probe.stdout, (
                f"PowerShell 5.1 space/& batch launch failed: rc={ps_probe.returncode} stderr={ps_probe.stderr}"
            )
        finally:
            shutil.rmtree(adv_parent, ignore_errors=True)

    # R2-2 Regression Guard: _catalog_lock decoupled from _inference_lock + run_id progress isolation + cancel
    lock_ready = threading.Event()
    release_lock = threading.Event()

    def _hold_inference_lock():
        with model_controller._inference_lock:
            lock_ready.set()
            release_lock.wait(timeout=3.0)

    holder = threading.Thread(target=_hold_inference_lock, daemon=True)
    holder.start()
    assert lock_ready.wait(timeout=1.0)
    try:
        t_probe = time.perf_counter()
        r_hw_mid = client.get("/hardware_info")
        r_mi_mid = client.post("/model_info")
        probe_ms = (time.perf_counter() - t_probe) * 1000.0
        assert r_hw_mid.status_code == 200 and r_mi_mid.status_code == 200
        assert probe_ms < 200.0, f"/hardware_info & /model_info blocked during inference lock ({probe_ms:.1f} ms)"
    finally:
        release_lock.set()
        holder.join(timeout=1.0)

    tracker = GenerationProgressTracker()
    tracker.start_run("run_1", "Starting run 1")
    tracker.start_run("run_2", "Starting run 2")
    tracker.update(generated_tokens=180, max_tokens=512, message="stale", run_id="run_1")
    snap = tracker.snapshot()
    assert snap["run_id"] == "run_2" and snap["generated_tokens"] == 0, "Stale run_1 overwrote active run_2 progress!"

    pre_cancelled = threading.Event()
    pre_cancelled.set()
    try:
        model_controller.generate(
            None,
            None,
            None,
            GenerateMeta(model_type="MORTM4.5D-80M", num_gems=1),
            cancel_event=pre_cancelled,
        )
        raise AssertionError("Expected GenerationCancelledError when cancel_event is pre-set")
    except GenerationCancelledError:
        pass

    r_cancel = client.post("/cancel_generate", json={"cancel_all": True})
    assert r_cancel.status_code == 200 and r_cancel.json()["cancelled"] is False

    print("  [PASS] FastAPI endpoints, CRLF/BOM launchers, lock decoupling & security guardrails verified")


def test_5_speed_benchmarks():
    print("\n--- [TEST 5] Speed Benchmarks (Key Estimator + Contiguous SDPA KV Cache) ---")
    midi_bytes = create_sample_midi_bytes(inst_program=0, num_measures=8)
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))

    t0 = time.perf_counter()
    for _ in range(20):
        detected_key = get_key_Polling(midi_data=pm)
    key_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0
    print(f"  Fast NumPy Ensemble Key Estimator : {key_ms:.3f} ms/file (Detected: {detected_key}) [music21 was ~1500 ms]")
    assert key_ms < 50.0

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    backend = HardwareBackendFactory.create(device_type)
    cfg_path = os.path.join(ROOT_DIR, "data", "models", "MORTM45D_80M_sft_gen", "config.json")
    args = MORTMArgs(cfg_path)
    model = MORTM(args, backend=backend)
    model.merge_lora()
    model.to(device=backend.device, dtype=backend.compute_dtype)
    model.eval()

    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
    prompt = np.array(
        [[tokenizer.get("<EOS>"), tokenizer.get("<SYSTEM>"), tokenizer.get("<TAG_END>"), tokenizer.get("<MGEN>")]],
        dtype=np.int64,
    )

    model.top_sampling_measure_kv_cache(tokenizer, prompt, max_tokens=16, end_tokens=[999999])
    if backend.device.type == "cuda":
        torch.cuda.synchronize()

    num_bench_tokens = 128
    t_start = time.perf_counter()
    model.top_sampling_measure_kv_cache(tokenizer, prompt, max_tokens=num_bench_tokens, end_tokens=[999999])
    if backend.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    tok_per_sec = num_bench_tokens / elapsed
    print(
        f"  MORTM4.5D-80M-SFT-Gen (11L, 768d, LoRA folded) on {backend.device} ({backend.compute_dtype}): "
        f"{tok_per_sec:.1f} tokens/sec ({elapsed*1000:.1f} ms for {num_bench_tokens} tokens)"
    )


def test_6_adversarial_round3_regressions():
    print("\n--- [TEST 6] Round 3 Adversarial Regressions (MSIX Guard, MIDI Bombs, Float Snap, 480 PPQN, Left-Pad) ---")
    from app import _client_key_from_request
    from model import _clean_msix_localcache_shadow
    from mortm.utils.convert import MIDI2Seq
    from mortm.utils.de_convert import ct_token_to_midi
    from mortm.utils.key import get_key_dict

    # 1. [CRIT-01] Non-default temp dir never touches LocalCache, and shadow migration byte-verifies user weights
    import model as model_mod
    tmp_root = tempfile.mkdtemp(prefix="compass_msix_test_")
    try:
        fake_localappdata = os.path.join(tmp_root, "LocalAppData")
        fake_appdata = os.path.join(tmp_root, "RoamingHost")
        fake_default_models = os.path.join(fake_appdata, "COMPASS_AI", "models")
        pkg_shadow = os.path.join(
            fake_localappdata, "Packages", "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0",
            "LocalCache", "Roaming", "COMPASS_AI", "models", "MORTM45D_80M_sft_gen",
        )
        os.makedirs(pkg_shadow, exist_ok=True)
        src_dir = os.path.join(ROOT_DIR, "data", "models")
        shutil.copy2(
            os.path.join(src_dir, "MORTM45D_80M_sft_gen", "config.json"),
            os.path.join(pkg_shadow, "config.json"),
        )
        user_weight_path = os.path.join(pkg_shadow, "my_custom_weights.safetensors")
        weight_payload = b"USER_PRECIOUS_CUSTOM_WEIGHTS_DO_NOT_DELETE"
        with open(user_weight_path, "wb") as f:
            f.write(weight_payload)

        old_local = os.environ.get("LOCALAPPDATA")
        old_appdata = os.environ.get("APPDATA")
        old_override = os.environ.pop("COMPASS_MODELS_DIR", None)
        os.environ["LOCALAPPDATA"] = fake_localappdata
        os.environ["APPDATA"] = fake_appdata
        orig_resolve = model_mod.resolve_host_write_path
        try:
            # (a) Seeding a custom non-default dir MUST NOT touch LocalCache at all
            custom_target = os.path.join(tmp_root, "custom_models")
            seed_user_models_dir(user_models_dir=custom_target, bundled_models_dir=src_dir)
            assert os.path.isfile(user_weight_path), (
                "Custom user_models_dir seed must not touch MSIX LocalCache!"
            )

            # (b) When running MSIX Store Python and UNC share is unavailable (non-admin Windows), shadow dir MUST NOT be deleted
            model_mod.resolve_host_write_path = lambda p: os.path.abspath(p)
            with patch.object(model_mod, "_is_msix_python", return_value=True):
                _clean_msix_localcache_shadow(fake_default_models)
            assert os.path.isfile(user_weight_path), (
                "Without UNC loopback share, LocalCache weights must remain untouched!"
            )

            # (c) When UNC path is available on MSIX Store Python, every file (including user weights) is migrated & byte-verified
            with patch.object(model_mod, "_is_msix_python", return_value=True):
                unc_Candidate = orig_resolve(fake_default_models)
                if model_mod._is_smb_unc_path(unc_Candidate):
                    model_mod.resolve_host_write_path = orig_resolve
                    _clean_msix_localcache_shadow(fake_default_models)
                    migrated_weight = os.path.join(
                        fake_default_models, "MORTM45D_80M_sft_gen", "my_custom_weights.safetensors"
                    )
                    assert os.path.isfile(migrated_weight) and os.path.getsize(migrated_weight) == len(weight_payload), (
                        "User custom weights were not byte-verified at physical destination!"
                    )
        finally:
            model_mod.resolve_host_write_path = orig_resolve
            if old_local is None:
                os.environ.pop("LOCALAPPDATA", None)
            else:
                os.environ["LOCALAPPDATA"] = old_local
            if old_appdata is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = old_appdata
            if old_override is not None:
                os.environ["COMPASS_MODELS_DIR"] = old_override
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # 2. [CRIT-03] Measure-0 5000-note chord bomb & extreme duration MIDI bounded cleanly
    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
    bomb_pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    bomb_inst = pretty_midi.Instrument(program=0)
    for idx in range(5000):
        bomb_inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=36 + (idx % 60), start=0.0, end=0.25)
        )
    bomb_inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=7200.0, end=7201.0))
    bomb_pm.instruments.append(bomb_inst)

    t_bomb = time.perf_counter()
    kd = get_key_dict(None, window_measures=8, step_measures=8, midi_data=bomb_pm)
    conv_bomb = MIDI2Seq(tokenizer, None, None, ["PIANO"], midi_data=bomb_pm)
    conv_bomb.convert()
    bomb_ms = (time.perf_counter() - t_bomb) * 1000.0
    assert bomb_ms < 1500.0, f"Chord/duration bomb took too long: {bomb_ms:.1f} ms"
    assert kd["metadata"]["total_measures"] == 512 and len(kd["segments"]) <= 65
    assert "PIANO" in conv_bomb.aya_node and len(conv_bomb.aya_node["PIANO"]) <= 8192

    # 3. [CRIT-04] RateLimiter non-eviction under key flood & X-Forwarded-For spoof immunity
    flood_limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0, max_tracked_keys=512)
    flood_limiter.check_or_raise("loopback", "limit")
    flood_limiter.check_or_raise("loopback", "limit")
    for k_i in range(600):
        try:
            flood_limiter.check_or_raise(f"10.0.{k_i // 256}.{k_i % 256}", "limit")
        except HTTPException:
            pass
    try:
        flood_limiter.check_or_raise("loopback", "limit")
        raise AssertionError("Active loopback bucket must not be evicted by key flood!")
    except HTTPException as exc:
        assert exc.status_code == 429

    class _DummyClient:
        host = "::ffff:127.0.0.1"

    class _DummyReq:
        client = _DummyClient()
        headers = {"X-Forwarded-For": "203.0.113.99"}

    assert _client_key_from_request(_DummyReq()) == "loopback"

    # 4. [HIGH-09] Exact Beat-1 grid snapping across prime BPMs (67, 73, 83) + 3/4 Time Signature support
    for bpm_val in (67.0, 73.0, 83.0):
        pm_prime = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=bpm_val)
        inst_p = pretty_midi.Instrument(program=0)
        sec_per_bar = (60.0 / bpm_val) * 4.0
        for bar_i in range(4):
            t_on = bar_i * sec_per_bar
            inst_p.notes.append(pretty_midi.Note(velocity=100, pitch=60 + bar_i, start=t_on, end=t_on + 0.3))
        pm_prime.instruments.append(inst_p)
        c_prime = MIDI2Seq(tokenizer, None, None, ["PIANO"], key="CM", midi_data=pm_prime)
        c_prime.convert()
        seq_tokens = [tokenizer.rev_get(int(t)) for t in c_prime.aya_node["PIANO"]]
        assert "s_95" not in seq_tokens, f"Float underflow s_95 at BPM {bpm_val}: {seq_tokens}"
        assert seq_tokens.count("s_0") == 4, f"Expected 4 exact Beat-1 s_0 tokens at BPM {bpm_val}, got {seq_tokens}"

    pm_waltz = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
    pm_waltz.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 0.0))
    inst_w = pretty_midi.Instrument(program=0)
    inst_w.notes.append(pretty_midi.Note(velocity=95, pitch=64, start=0.0, end=0.5))
    pm_waltz.instruments.append(inst_w)
    c_waltz = MIDI2Seq(tokenizer, None, None, ["PIANO"], midi_data=pm_waltz)
    c_waltz.convert()
    assert not c_waltz.is_error and "PIANO" in c_waltz.aya_node

    # 5. [HIGH-10.2] Truncated 2nd MTrk chunk and zero-tempo FF 51 03 00 00 00 -> HTTP 400 (never 500)
    client = TestClient(app)
    valid_mtrk = b"MTrk\x00\x00\x00\x04\x00\xff\x2f\x00"
    trunc_second_mtrk = (
        b"MThd\x00\x00\x00\x06\x00\x01\x00\x02\x01\xe0"
        + valid_mtrk
        + b"MTrk\x00\x00\x00\x20\x00\x90"
    )
    r_t2 = client.post(
        "/generate",
        data={"meta": json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta_past"})},
        files={"past_midi": ("trunc2.mid", trunc_second_mtrk, "audio/midi")},
    )
    assert r_t2.status_code == 400, f"Expected 400 for truncated 2nd MTrk, got {r_t2.status_code}"

    zero_tempo_mtrk = b"MTrk\x00\x00\x00\x0b\x00\xff\x51\x03\x00\x00\x00\x00\xff\x2f\x00"
    zero_tempo_smf = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0" + zero_tempo_mtrk
    r_zt = client.post(
        "/generate",
        data={"meta": json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta_past"})},
        files={"past_midi": ("zerotempo.mid", zero_tempo_smf, "audio/midi")},
    )
    assert r_zt.status_code == 400, f"Expected 400 for zero-tempo SMF, got {r_zt.status_code}"

    # 6. [MED-04] & [LOW-03] 480 PPQN zero-jitter output + heterogeneous left-padded prompts free of NaN
    tmp_out = tempfile.mkdtemp(prefix="compass_deconv_")
    try:
        mid_480_path = os.path.join(tmp_out, "test_480.mid")
        tok_seq = [
            tokenizer.get("<INST_PIANO>"),
            tokenizer.get("s_1"),
            tokenizer.get("p_60"),
            tokenizer.get("d_3"),
            tokenizer.get("<SME>"),
            tokenizer.get("s_1"),
            tokenizer.get("p_64"),
            tokenizer.get("d_3"),
            tokenizer.get("<ESEQ>"),
        ]
        pm_out = ct_token_to_midi(tokenizer, tok_seq, mid_480_path, tempo=120)
        assert pm_out.resolution == 480
        reloaded = pretty_midi.PrettyMIDI(mid_480_path)
        assert reloaded.resolution == 480
        tick_bar1 = round(reloaded.instruments[0].notes[0].start * 2.0 * 480)
        tick_bar2 = round(reloaded.instruments[0].notes[1].start * 2.0 * 480)
        assert tick_bar1 == 20 and tick_bar2 == 1940, (
            f"Expected exact 20 and 1940 ticks at 480 PPQN, got {tick_bar1}, {tick_bar2}"
        )
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)

    backend = HardwareBackendFactory.create("cpu")
    cfg_path = os.path.join(ROOT_DIR, "data", "models", "MORTM45D_80M_sft_gen", "config.json")
    model = MORTM(MORTMArgs(cfg_path), backend=backend).eval()
    pad_id = tokenizer.get("<PAD>")
    hetero_prompts = [
        np.array([tokenizer.get("<EOS>"), tokenizer.get("<MGEN>")], dtype=np.int64),
        np.array(
            [
                tokenizer.get("<EOS>"),
                tokenizer.get("<SYSTEM>"),
                tokenizer.get("<TAG_END>"),
                tokenizer.get("<MGEN>"),
                tokenizer.get("<INST_PIANO>"),
            ],
            dtype=np.int64,
        ),
    ]
    all_toks, (prompts_out, gen_only) = model.top_sampling_measure_kv_cache(
        tokenizer, hetero_prompts, max_tokens=6, end_tokens=[999999]
    )
    assert len(gen_only) == 2 and len(prompts_out) == 2 and len(all_toks) == 2
    assert len(prompts_out[0]) == 2 and len(prompts_out[1]) == 5
    assert len(all_toks[0]) == 8 and len(all_toks[1]) == 11
    assert len(gen_only[0]) == 7 and len(gen_only[1]) == 8
    for seq in all_toks:
        assert pad_id not in seq

    print("  [PASS] All Round 3 backend security, numerical & timing regressions verified")


def test_7_round2_critic_remediations():
    print("\n--- [TEST 7] Phase 2 Critic Round 2 Remediations (HMAC Auth, 7/8 & 3/4 Meter, 4.5 Budget, Scoped Cancel, VLQ Bomb) ---")
    import hashlib
    import hmac
    from unittest.mock import patch

    import app as app_mod
    import main as main_mod
    import model as model_mod
    from mortm.utils.convert import MIDI2Seq
    from mortm.utils.de_convert import ct_token_to_midi
    from rapper import AbstractModelRapper

    # 1. [R2-HIGH-03] Cryptographic /health HMAC challenge-response + X-Compass-Handshake HTTP 403 enforcement
    client = TestClient(app)
    secret = "compass_round2_secret_7f8a9b"
    with patch.dict(os.environ, {"COMPASS_HANDSHAKE_TOKEN": secret}, clear=False):
        r_h_plain = client.get("/health")
        assert r_h_plain.status_code == 200
        h_plain = r_h_plain.json()
        assert h_plain["auth_required"] is True
        assert "handshake" not in h_plain
        assert h_plain.get("handshake_hmac") is None

        nonce = "challenge_nonce_2026_10_26"
        r_h_chal = client.get("/health", headers={"X-Compass-Challenge": nonce})
        expected_mac = hmac.new(secret.encode("utf-8"), nonce.encode("utf-8"), hashlib.sha256).hexdigest()
        assert r_h_chal.status_code == 200
        assert r_h_chal.json()["handshake_hmac"] == expected_mac

        # Unauthenticated requests to protected endpoints MUST fail with HTTP 403
        assert client.get("/hardware_info").status_code == 403
        assert client.get("/generation_progress").status_code == 403
        assert client.post("/model_info").status_code == 403
        assert client.post("/rescan_models").status_code == 403
        assert client.post("/cancel_generate").status_code == 403
        assert client.post(
            "/generate",
            data={"meta": json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta"})},
        ).status_code == 403

        # Wrong token also fails with HTTP 403
        assert client.get("/hardware_info", headers={"X-Compass-Handshake": "wrong_secret"}).status_code == 403

        # Valid X-Compass-Handshake succeeds with HTTP 200
        auth_headers = {"X-Compass-Handshake": secret}
        assert client.get("/hardware_info", headers=auth_headers).status_code == 200
        assert client.post("/model_info", headers=auth_headers).status_code == 200
        assert client.post("/cancel_generate", headers=auth_headers, json={"cancel_all": True}).status_code == 200

    # 2. [R2-HIGH-05] Odd & Triple Meter (3/4, 7/8) Round-Trip + Unsupported 5/4 HTTP 400 Rejection
    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
    tmp_meter = tempfile.mkdtemp(prefix="compass_meter_test_")
    try:
        # 7/8 at 120 BPM -> 1 bar = 3.5 quarters = 1.75s = 1680 ticks (480 PPQN)
        mid_78_path = os.path.join(tmp_meter, "odd_7_8.mid")
        seq_78 = [
            tokenizer.get("<INST_PIANO>"),
            tokenizer.get("s_0"),
            tokenizer.get("p_60"),
            tokenizer.get("d_6"),
            tokenizer.get("<SME>"),
            tokenizer.get("s_0"),
            tokenizer.get("p_67"),
            tokenizer.get("d_6"),
            tokenizer.get("<ESEQ>"),
        ]
        ct_token_to_midi(tokenizer, seq_78, mid_78_path, tempo=120, time_signature=(7, 8))
        reloaded_78 = pretty_midi.PrettyMIDI(mid_78_path)
        assert len(reloaded_78.time_signature_changes) >= 1
        assert reloaded_78.time_signature_changes[0].numerator == 7
        assert reloaded_78.time_signature_changes[0].denominator == 8
        bar1_tick = round(reloaded_78.instruments[0].notes[0].start * 2.0 * 480)
        bar2_tick = round(reloaded_78.instruments[0].notes[1].start * 2.0 * 480)
        assert bar1_tick == 0 and bar2_tick == 1680, (
            f"7/8 Bar 2 must start at 1680 ticks (not 1920), got {bar1_tick}, {bar2_tick}"
        )

        # Re-tokenize 7/8 MIDI via MIDI2Seq and confirm 2 exact Beat-1 s_0 measures
        conv_78 = MIDI2Seq(tokenizer, None, None, ["PIANO"], key="CM", midi_data=reloaded_78)
        conv_78.convert()
        toks_78_names = [tokenizer.rev_get(int(t)) for t in conv_78.aya_node["PIANO"]]
        assert toks_78_names.count("<SME>") >= 1
        assert toks_78_names.count("s_0") == 2, f"Expected 2 exact s_0 in 7/8 roundtrip, got {toks_78_names}"

        # Unsupported 5/4 (120 steps > 96 max vocabulary) must return clean HTTP 400
        r_54 = client.post(
            "/generate",
            data={
                "meta": json.dumps(
                    {
                        "model_type": "MORTM4.5D-80M-SFT-Gen",
                        "task": "meta",
                        "time_signature_numerator": 5,
                        "time_signature_denominator": 4,
                    }
                )
            },
        )
        assert r_54.status_code == 400 and "拍子" in r_54.json()["detail"]
    finally:
        shutil.rmtree(tmp_meter, ignore_errors=True)

    # 3. [R2-HIGH-06] & [R2-MED-07] Event-Boundary Slicing, MORTM45 Combined Prompt Budget (512), & Bar-0 <SME> Rest
    tok_omega = Tokenizer(omega_converter(TO_TOKEN))
    # Construct 30 notes (90 event tokens) + <INST_PIANO> + <SME>, slice to max_tokens=10 -> must never start with p_* or d_*
    long_music = [tok_omega.get("<INST_PIANO>")]
    for step_i in range(20):
        long_music.extend([tok_omega.get(f"s_{step_i * 4}"), tok_omega.get("p_60"), tok_omega.get("d_4")])
    trimmed_tail = AbstractModelRapper.trim_music_tokens_on_event_boundary(
        tok_omega, long_music, 11, keep_tail=True
    )
    trimmed_head = AbstractModelRapper.trim_music_tokens_on_event_boundary(
        tok_omega, long_music, 11, keep_tail=False
    )
    for trimmed in (trimmed_tail, trimmed_head):
        assert len(trimmed) <= 11
        trimmed_names = [tok_omega.rev_get(int(x)) for x in trimmed]
        first_event_after_meta = next(
            (nm for nm in trimmed_names if nm not in ("<INST_PIANO>", "<SME>")),
            None,
        )
        assert first_event_after_meta is None or first_event_after_meta.startswith("s_"), (
            f"Trimmed sequence started mid-triplet with orphan {first_event_after_meta}: {trimmed_names}"
        )

    tmp_legacy = tempfile.mkdtemp(prefix="compass_legacy45_")
    try:
        cfg_45 = {
            "vocab_size": 650,
            "d_model": 64,
            "dim_feedforward": 128,
            "num_heads": 4,
            "d_layer": 2,
            "dropout": 0.0,
            "position_length": 512,
            "min_length": 8,
            "use_moe_decoder": False,
            "norm_type": "rmsnorm",
            "use_silu": True,
            "use_bias": False,
            "use_rope": False,
        }
        cfg_45_path = os.path.join(tmp_legacy, "config.json")
        with open(cfg_45_path, "w", encoding="utf-8") as f:
            json.dump(cfg_45, f)
        dense_midi = os.path.join(tmp_legacy, "dense_ctx.mid")
        pm_dense = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
        inst_d = pretty_midi.Instrument(program=0)
        for i in range(180):
            t0 = i * 0.125
            inst_d.notes.append(pretty_midi.Note(velocity=95, pitch=48 + (i % 24), start=t0, end=t0 + 0.1))
        pm_dense.instruments.append(inst_d)
        pm_dense.write(dense_midi)

        legacy_rapper = MORTM45Rapper(
            model_info={
                "model_folder_path": tmp_legacy,
                "model_name": "MORTM4.5-Flash",
                "tag": {"instruments": ["PIANO"], "model": "pretrained", "task": ["Meta2MIDI"]},
            },
            backend=CpuHardwareBackend(),
        )
        prep_45 = legacy_rapper.preprocessing(
            dense_midi,
            dense_midi,
            dense_midi,
            GenerateMeta(
                model_type="MORTM4.5-Flash",
                task="infill",
                ai_continue_mode=True,
                program=["PIANO"],
            ),
        )
        prompt_len = prep_45["sequence"].shape[1]
        assert prompt_len <= int(512 * 0.75) + 4 and prompt_len < 512, (
            f"Combined MORTM45 prompt overflowed position_length=512 budget: {prompt_len}"
        )

        # Verify Bar-0 <SME> (1-bar rest) is preserved by MORTM45Rapper.postprocessing (note stays at 2.0s / Bar 2)
        gen_with_bar0_rest = np.array(
            [
                tok_omega.get("<INST_PIANO>"),
                tok_omega.get("<SME>"),
                tok_omega.get("s_0"),
                tok_omega.get("p_64"),
                tok_omega.get("d_12"),
                tok_omega.get("<ESEQ>"),
            ],
            dtype=np.int64,
        )
        out_paths = legacy_rapper.postprocessing(
            save_directory=tmp_legacy,
            meta=GenerateMeta(model_type="MORTM4.5-Flash", tempo=120),
            tokenizer=Tokenizer(omega_converter(TO_TOKEN)),
            task="Meta2MIDI",
            full_sequences=[gen_with_bar0_rest],
            generated_parts=[gen_with_bar0_rest],
        )
        pm_rest = pretty_midi.PrettyMIDI(out_paths[0])
        assert len(pm_rest.instruments) == 1 and len(pm_rest.instruments[0].notes) == 1
        rest_start_tick = round(pm_rest.instruments[0].notes[0].start * 2.0 * 480)
        assert rest_start_tick == 1920, (
            f"Leading <SME> 1-bar rest must keep Bar 2 note at 1920 ticks (2.0s), got {rest_start_tick}"
        )
    finally:
        shutil.rmtree(tmp_legacy, ignore_errors=True)

    # 4. [R2-HIGH-07] Scoped job_id cancellation leaves sibling queued jobs untouched + pre-cancel race cache
    ev_alpha = model_controller.register_cancel_event(job_id="job_alpha_r2")
    ev_beta = model_controller.register_cancel_event(job_id="job_beta_r2")
    try:
        assert model_controller.cancel_active_generation("job_alpha_r2") is True
        assert ev_alpha.is_set() is True
        assert ev_beta.is_set() is False, "Cancelling job_alpha must NOT cancel queued sibling job_beta!"

        # Pre-cancel race: /cancel_generate arrives BEFORE /generate registers job_gamma_r2
        assert model_controller.cancel_active_generation("job_gamma_r2") is False
        ev_gamma = model_controller.register_cancel_event(job_id="job_gamma_r2")
        try:
            assert ev_gamma.is_set() is True, "Pre-cancelled job_id must immediately set registered Event!"
        finally:
            model_controller.unregister_cancel_event(ev_gamma, job_id="job_gamma_r2")
    finally:
        model_controller.unregister_cancel_event(ev_alpha, job_id="job_alpha_r2")
        model_controller.unregister_cancel_event(ev_beta, job_id="job_beta_r2")

    # 5. [R2-MED-01] MSIX _clean_msix_localcache_shadow preserves same-size user-edited config.json
    tmp_msix_edit = tempfile.mkdtemp(prefix="compass_msix_samesize_")
    try:
        fake_local = os.path.join(tmp_msix_edit, "Local")
        fake_roaming = os.path.join(tmp_msix_edit, "Roaming")
        fake_host_models = os.path.join(fake_roaming, "COMPASS_AI", "models")
        fake_host_sub = os.path.join(fake_host_models, "MORTM45D_80M_sft_gen")
        shadow_sub = os.path.join(
            fake_local,
            "Packages",
            "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0",
            "LocalCache",
            "Roaming",
            "COMPASS_AI",
            "models",
            "MORTM45D_80M_sft_gen",
        )
        os.makedirs(fake_host_sub, exist_ok=True)
        os.makedirs(shadow_sub, exist_ok=True)

        bundled_cfg_path = os.path.join(ROOT_DIR, "data", "models", "MORTM45D_80M_sft_gen", "config.json")
        with open(bundled_cfg_path, "rb") as fh_bundled:
            raw_bundled = fh_bundled.read()
        # Host still has unmodified bundled template
        with open(os.path.join(fake_host_sub, "config.json"), "wb") as f:
            f.write(raw_bundled)
        # User edited 1 digit inside shadow config.json -> EXACT SAME BYTE LENGTH, DIFFERENT SHA-256!
        edited_bytes = raw_bundled.replace(b'"lora_r": 4', b'"lora_r": 8', 1)
        assert len(edited_bytes) == len(raw_bundled) and edited_bytes != raw_bundled
        with open(os.path.join(shadow_sub, "config.json"), "wb") as f:
            f.write(edited_bytes)

        with patch.dict(os.environ, {"LOCALAPPDATA": fake_local, "APPDATA": fake_roaming}, clear=False):
            os.environ.pop("COMPASS_MODELS_DIR", None)
            if os.name == "nt":
                with patch.object(model_mod, "_is_msix_python", return_value=False), patch.object(
                    model_mod, "resolve_host_write_path", side_effect=lambda p: os.path.abspath(p)
                ):
                    model_mod._clean_msix_localcache_shadow(
                        fake_host_models, os.path.join(ROOT_DIR, "data", "models")
                    )
                with open(os.path.join(fake_host_sub, "config.json"), "rb") as fh_host_cfg:
                    host_cfg_bytes = fh_host_cfg.read()
                assert host_cfg_bytes == edited_bytes, (
                    "Same-size user edit in shadow config.json was lost instead of migrated!"
                )
    finally:
        shutil.rmtree(tmp_msix_edit, ignore_errors=True)

    # 6. [R2-MED-06] 38-byte 9,999,000-tick VLQ MIDI bomb rejected in <10ms + pre-reserved "loopback" rate slot
    # 9,999,000 = 0x989298 -> VLQ: (4 << 21) | (0x62 << 14) | (0x25 << 7) | 0x18 -> 0x84 0xE2 0xA5 0x18
    vlq_9999000 = bytes([0x84, 0xE2, 0xA5, 0x18])
    mtrk_body = vlq_9999000 + b"\x90\x3c\x64\x00\x80\x3c\x00\x00\xff\x2f\x00"
    vlq_bomb_smf = (
        b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0"
        + b"MTrk"
        + len(mtrk_body).to_bytes(4, "big")
        + mtrk_body
    )
    t_vlq = time.perf_counter()
    r_vlq = client.post(
        "/generate",
        data={"meta": json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta_past"})},
        files={"past_midi": ("vlq_bomb.mid", vlq_bomb_smf, "audio/midi")},
    )
    vlq_ms = (time.perf_counter() - t_vlq) * 1000.0
    assert r_vlq.status_code == 400, f"Expected HTTP 400 for 9.999M-tick VLQ bomb, got {r_vlq.status_code}"
    assert vlq_ms < 100.0

    # Pre-reserved "loopback" slot cannot be starved even if remote keys flood BEFORE loopback's 1st request
    pre_limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0, max_tracked_keys=4)
    for r_i in range(20):
        try:
            pre_limiter.check_or_raise(f"198.51.100.{r_i}", "limit")
        except HTTPException:
            pass
    pre_limiter.check_or_raise("loopback", "limit")
    pre_limiter.check_or_raise("loopback", "limit")
    try:
        pre_limiter.check_or_raise("loopback", "limit")
        raise AssertionError("Pre-reserved loopback slot bypassed rate limiter after remote flood!")
    except HTTPException as exc:
        assert exc.status_code == 429

    # 7. [R2-LOW-01] Empty/malformed COMPASS_*="" env vars fall back cleanly without ValueError
    with patch.dict(
        os.environ,
        {
            "COMPASS_PORT": "",
            "COMPASS_GEN_RPM": "",
            "COMPASS_MAX_CACHED_MODELS": "   ",
            "COMPASS_MAX_MIDI_BYTES": "invalid_num",
        },
        clear=False,
    ):
        assert main_mod._env_int("COMPASS_PORT", 8000) == 8000
        assert app_mod._env_int("COMPASS_GEN_RPM", 30) == 30
        assert model_mod._env_int("COMPASS_MAX_CACHED_MODELS", 2) == 2
        assert app_mod._env_int("COMPASS_MAX_MIDI_BYTES", 10485760) == 10485760

    print("  [PASS] All 7 Round-2 Critic backend/security/meter/budget regressions verified")


def test_8_round3_fact_checker_remediations():
    print("\n--- [TEST 8] Round-3 Fact-Checker Remediations (Non-4/4 Live Decoding, AppData Import Sandbox, Budget & Offline Bootstrap) ---")
    from unittest.mock import patch
    import installer.bootstrap as bootstrap_mod
    from models.mortm.mortm45d import MORTM45DRapper
    from mortm.utils.convert import MIDIConverter
    from rapper import AbstractModelRapper

    cpu_backend = CpuHardwareBackend()
    tmp_r3 = tempfile.mkdtemp(prefix="compass_r3_test_")
    try:
        # 1. [R3-HIGH-01a] Live decoding in 3/4 (72), 2/4 (48), 7/8 (84), 5/8 (60), 3/8 (36)
        # Seed a model whose output bias STRONGLY prefers out-of-bar shift tokens (`s_40`, `s_64`, `s_80`, `s_95`)
        # unless `forbidden_token_ids` masks `s_{steps_per_bar}..s_96` to `-inf`.
        sft_dir = os.path.join(tmp_r3, "sft_meter_mask")
        out_dir = os.path.join(tmp_r3, "sft_out")
        os.makedirs(sft_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        cfg_path = os.path.join(sft_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab_size": 700,
                    "d_model": 64,
                    "dim_feedforward": 128,
                    "num_heads": 4,
                    "d_layer": 2,
                    "dropout": 0.0,
                    "position_length": 128,
                    "min_length": 8,
                    "use_moe_decoder": False,
                    "norm_type": "rmsnorm",
                    "use_silu": True,
                    "use_bias": False,
                    "use_rope": True,
                    "use_attn_lora": False,
                    "use_ffn_lora": False,
                },
                f,
            )

        tok_sft = Tokenizer(music_token=get_token_converter_pro(TO_TOKEN))
        m_args = MORTMArgs(cfg_path)
        biased_model = MORTM(m_args, backend=cpu_backend)
        with torch.no_grad():
            for p in biased_model.parameters():
                p.zero_()
            for name, p in biased_model.named_parameters():
                if "norm" in name and p.ndim == 1:
                    p.fill_(1.0)
            biased_model.Wout.bias.fill_(-20.0)
            chain = [
                "<MGEN>",
                "<INST_PIANO>",
                "s_0",
                "p_60",
                "d_12",
                "<SME>",
                "s_12",
                "p_64",
                "d_12",
                "<ESEQ>",
                "<TE>",
            ]
            for dim_idx in range(len(chain) - 1):
                src_id = tok_sft.get(chain[dim_idx])
                dst_id = tok_sft.get(chain[dim_idx + 1])
                biased_model.embedding.weight[src_id, dim_idx] = 1.0
                biased_model.Wout.bias[dst_id] = 0.0
                biased_model.Wout.weight[dst_id, dim_idx] = 3.0  # post-RMSNorm (~8.0) * 3.0 = +24.0 (< +28.0 out-of-bar!)
        pth_path = os.path.join(sft_dir, "model.pth")
        torch.save(biased_model.state_dict(), pth_path)

        rapper_d = MORTM45DRapper(
            model_info={
                "model_folder_path": sft_dir,
                "model_name": "MORTM4.5D-80M-SFT-Gen",
                "tag": {"instruments": ["PIANO"], "model": "sft_gen", "task": ["meta", "meta_past"]},
            },
            backend=cpu_backend,
        )
        for num, den, expected_steps in [(3, 4, 72), (2, 4, 48), (7, 8, 84), (5, 8, 60), (3, 8, 36)]:
            # Give EVERY out-of-bar shift (`s_{expected_steps}..s_96`) +28.0 logit bias so if even one is unmasked it wins!
            with torch.no_grad():
                for step_k in range(13, 97):
                    tid_k = tok_sft.get(f"s_{step_k}")
                    if tid_k is not None:
                        rapper_d.model.Wout.bias[tid_k] = 28.0 if step_k >= expected_steps else -20.0

            forbidden = AbstractModelRapper.build_out_of_bar_shift_token_ids(tok_sft, (num, den))
            assert tok_sft.get(f"s_{expected_steps}") in forbidden
            assert tok_sft.get("s_96") in forbidden
            assert tok_sft.get(f"s_{expected_steps - 1}") not in forbidden

            meta_meter = GenerateMeta(
                model_type="MORTM4.5D-80M-SFT-Gen",
                task="meta",
                program=["PIANO"],
                num_gems=1,
                genfield_measure=2,
                thinking=False,
                temperature=0.05,
                p=0.95,
                time_signature_numerator=num,
                time_signature_denominator=den,
            )
            prep = rapper_d.preprocessing(None, None, None, meta_meter)
            gen = rapper_d.generate(**prep)
            gen_seq = np.asarray(gen["generated_parts"][0]).flatten()
            forbidden_set = set(forbidden)
            assert all(int(t) not in forbidden_set for t in gen_seq), (
                f"Decoded sequence in {num}/{den} leaked forbidden shift token!"
            )
            out_paths = rapper_d.postprocessing(out_dir, **gen)
            assert len(out_paths) == 1
            out_pm = pretty_midi.PrettyMIDI(out_paths[0])
            assert len(out_pm.time_signature_changes) >= 1
            assert out_pm.time_signature_changes[0].numerator == num
            assert out_pm.time_signature_changes[0].denominator == den
            all_notes = [n for inst in out_pm.instruments for n in inst.notes]
            assert len(all_notes) >= 1

        # 2. [R3-HIGH-01b] Bidirectional meter alignment between `meta` and `context_midi`
        # Case A: SMF with NO TimeSignature meta event (only raw notes over 12 quarter-note beats = 4 bars of 3/4)
        # Build raw SMF bytes directly without FF 58 TimeSignature meta event (`pretty_midi` writes 4/4 if empty)
        mtrk_events = bytearray()
        # Tempo = 120 BPM (500,000 us/quarter): 00 FF 51 03 07 A1 20
        mtrk_events.extend(b"\x00\xff\x51\x03\x07\xa1\x20")
        # 12 quarter notes at 480 PPQN (duration 384 = 0x83 0x00, gap 96 = 0x60)
        for beat in range(12):
            pitch = 60 + (beat % 4)
            delta_on = b"\x00" if beat == 0 else b"\x60"
            mtrk_events.extend(delta_on + bytes([0x90, pitch, 95]) + b"\x83\x00" + bytes([0x80, pitch, 0]))
        mtrk_events.extend(b"\x00\xff\x2f\x00")
        raw_no_ts_smf = (
            b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0"
            + b"MTrk"
            + len(mtrk_events).to_bytes(4, "big")
            + bytes(mtrk_events)
        )
        no_ts_path = os.path.join(tmp_r3, "no_ts_12beats.mid")
        with open(no_ts_path, "wb") as f:
            f.write(raw_no_ts_smf)
        assert len(pretty_midi.PrettyMIDI(no_ts_path).time_signature_changes) == 0

        conv_34 = MIDIConverter(
            tok_sft,
            tmp_r3,
            "no_ts_12beats.mid",
            program_list=["PIANO"],
            use_midi2seq=True,
            use_midi2seq_with_chord=False,
            default_time_signature=(3, 4),
        )
        conv_34.convert()
        assert conv_34.has_explicit_time_signature is False
        assert conv_34.time_signature == (3, 4)
        sme_id = tok_sft.get("<SME>")
        piano_seq = conv_34.midi2seq.aya_node["PIANO"]
        # 12 quarter-note beats under 3/4 = 4 bars -> 3 `<SME>` bar separators (vs 2 `<SME>` under 4/4!)
        assert int(np.sum(piano_seq == sme_id)) == 3, (
            f"Expected 3 <SME> separators (4 bars of 3/4), got {int(np.sum(piano_seq == sme_id))}"
        )

        # Case B: Context MIDI HAS explicit 3/4 TimeSignature, while `GenerateMeta` omits time_signature_* fields
        pm_34 = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        pm_34.time_signature_changes = [pretty_midi.TimeSignature(3, 4, 0.0)]
        inst_34 = pretty_midi.Instrument(program=0)
        for beat in range(6):
            inst_34.notes.append(pretty_midi.Note(velocity=95, pitch=62, start=beat * 0.5, end=beat * 0.5 + 0.4))
        pm_34.instruments.append(inst_34)
        raw_34_path = os.path.join(tmp_r3, "ctx_34.mid")
        pm_34.write(raw_34_path)

        meta_implicit_ts = GenerateMeta(
            model_type="MORTM4.5D-80M-SFT-Gen",
            task="meta_past",
            program=["PIANO"],
            num_gems=1,
            genfield_measure=1,
            thinking=False,
        )
        assert AbstractModelRapper.has_explicit_time_signature(meta_implicit_ts) is False
        prep_inh = rapper_d.preprocessing(raw_34_path, None, None, meta_implicit_ts)
        assert prep_inh["effective_time_sig"] == (3, 4)
        gen_inh = rapper_d.generate(**prep_inh)
        inh_paths = rapper_d.postprocessing(out_dir, **gen_inh)
        inh_pm = pretty_midi.PrettyMIDI(inh_paths[0])
        assert inh_pm.time_signature_changes[0].numerator == 3
        assert inh_pm.time_signature_changes[0].denominator == 4

        # 3. [R3-HIGH-02] Fresh subprocess import of `tests/test_model_engine.py` NEVER touches clean `%APPDATA%`
        clean_roaming = os.path.join(tmp_r3, "clean_host_roaming")
        clean_local = os.path.join(tmp_r3, "clean_host_local")
        os.makedirs(clean_roaming, exist_ok=True)
        os.makedirs(clean_local, exist_ok=True)
        sub_env = os.environ.copy()
        sub_env.pop("COMPASS_MODELS_DIR", None)
        sub_env["APPDATA"] = clean_roaming
        sub_env["LOCALAPPDATA"] = clean_local
        sub_env["XDG_CONFIG_HOME"] = clean_roaming
        res_import = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util; s = importlib.util.spec_from_file_location('tme', 'tests/test_model_engine.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print('IMPORT_OK')",
            ],
            cwd=ROOT_DIR,
            env=sub_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res_import.returncode == 0 and "IMPORT_OK" in res_import.stdout, res_import.stderr
        non_driver_roaming = [e for e in os.listdir(clean_roaming) if e.upper() != "NVIDIA"]
        assert non_driver_roaming == [] and not os.path.exists(os.path.join(clean_roaming, "COMPASS_AI")), (
            f"Importing tests/test_model_engine.py polluted clean APPDATA Roaming: {non_driver_roaming}"
        )
        assert not os.path.exists(os.path.join(clean_local, "COMPASS_AI"))

        # 4. [R3-MED-06] Balanced `<ESEQ>` frame trimming + `MORTM45Rapper` tiny `position_length=32` safety
        p_id = tok_sft.get("<INST_PIANO>")
        x_id = tok_sft.get("<INST_SAX>")
        eseq_id = tok_sft.get("<ESEQ>")
        s0_id = tok_sft.get("s_0")
        p60_id = tok_sft.get("p_60")
        d12_id = tok_sft.get("d_12")
        two_frames = [
            p_id, s0_id, p60_id, d12_id, eseq_id,
            x_id, s0_id, p60_id, d12_id, eseq_id,
        ]
        trimmed_5 = MORTM45DRapper._trim_framed_context_tokens(
            tok_sft, two_frames, 5, keep_tail=True
        )
        assert len(trimmed_5) <= 5
        assert trimmed_5.count(eseq_id) == sum(1 for t in trimmed_5 if t in (p_id, x_id)), (
            f"Unbalanced <INST_*>/<ESEQ> after _trim_framed_context_tokens: {trimmed_5}"
        )

        legacy_dir = os.path.join(tmp_r3, "legacy_tiny_pos32")
        os.makedirs(legacy_dir, exist_ok=True)
        legacy_cfg_path = os.path.join(legacy_dir, "config.json")
        with open(legacy_cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab_size": 700,
                    "d_model": 64,
                    "dim_feedforward": 128,
                    "num_heads": 4,
                    "d_layer": 2,
                    "dropout": 0.0,
                    "position_length": 32,
                    "min_length": 4,
                    "use_moe_decoder": False,
                    "norm_type": "layernorm",
                    "use_silu": False,
                    "use_bias": True,
                    "use_rope": True,
                },
                f,
            )
        legacy_model = MORTM(MORTMArgs(legacy_cfg_path), backend=cpu_backend)
        legacy_pth = os.path.join(legacy_dir, "model.pth")
        torch.save(legacy_model.state_dict(), legacy_pth)
        rapper_45 = MORTM45Rapper(
            model_info={
                "model_folder_path": legacy_dir,
                "model_name": "MORTM4.5-Flash",
                "tag": {"instruments": ["PIANO", "SAX"], "model": "generation", "task": ["Meta2MIDI"]},
            },
            backend=cpu_backend,
        )

        duet_pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        for prog in (0, 65):
            inst = pretty_midi.Instrument(program=prog)
            for b in range(8):
                inst.notes.append(pretty_midi.Note(velocity=90, pitch=60 + (b % 5), start=b * 0.5, end=b * 0.5 + 0.4))
            duet_pm.instruments.append(inst)
        duet_path = os.path.join(tmp_r3, "duet_ctx.mid")
        duet_pm.write(duet_path)

        meta_tiny = GenerateMeta(
            model_type="MORTM4.5-Flash",
            task="infill",
            program=["PIANO", "SAX"],
            num_gems=1,
            genfield_measure=2,
            thinking=False,
        )
        prep_tiny = rapper_45.preprocessing(duet_path, duet_path, duet_path, meta_tiny)
        prompt_tiny_len = int(len(prep_tiny["sequence"][0]))
        assert prompt_tiny_len < 32, (
            f"Expected prompt < 32 for position_length=32, got {prompt_tiny_len}"
        )

        # 5. [R3-LOW-02] Offline CUDA wheel `CalledProcessError` fallback when CPU `torch` is already installed
        real_check_call = subprocess.check_call
        saw_cuda_install_attempt = False

        def _fake_offline_cuda_pip(cmd, *args, **kwargs):
            nonlocal saw_cuda_install_attempt
            if any("download.pytorch.org/whl/cu" in str(part) for part in cmd):
                saw_cuda_install_attempt = True
                raise subprocess.CalledProcessError(1, cmd)
            return 0

        with (
            patch.object(bootstrap_mod, "has_nvidia_gpu", return_value=True),
            patch.object(bootstrap_mod, "is_torch_ready", side_effect=lambda expect_cuda: not expect_cuda),
            patch.object(subprocess, "check_call", side_effect=_fake_offline_cuda_pip),
        ):
            bootstrap_mod.ensure_dependencies(skip_install=False)
        assert saw_cuda_install_attempt is True
        assert subprocess.check_call is real_check_call
    finally:
        shutil.rmtree(tmp_r3, ignore_errors=True)

    print("  [PASS] All Round-3 Fact-Checker remediations (meter masking, bidirectional TS, AppData isolation, budget & offline CUDA fallback) verified")


def test_9_round4_fact_checker_remediations() -> None:
    """
    Physical regression suite for all Round-4 Fact-Checker backend & SMF findings:
    - `[R4-MED-02]`: Unified meter pre-resolution across multi-context (`infill` where `past` omits `0xFF 0x58`
      and `future` has `3/4`), explicit `meta` override (`meta=3/4` vs `past=4/4` `0xFF 0x58`), and conflicting
      explicit context meters (`3/4` vs `7/8` -> `ValueError` / `HTTP 400`).
    - `[R4-LOW-02]`: SMF `0xFF` Meta and `0xF1..0xF6` System Common events reset `running_status = 0` in `app.py`.
    """
    print("\n=== [TEST 9] Round-4 Fact-Checker Remediations (Unified Multi-Context Meter & SMF Running Status) ===")
    import pretty_midi
    import app as app_mod
    from models.mortm.mortm45 import MORTM45Rapper
    from models.mortm.mortm45d import MORTM45DRapper
    from rapper import AbstractModelRapper

    tmp_r4 = tempfile.mkdtemp(prefix="compass_r4_audit_")
    try:
        cpu_backend = CpuHardwareBackend()
        tok_sft = Tokenizer(get_token_converter_pro(TO_TOKEN))
        forbidden_34 = set(AbstractModelRapper.build_out_of_bar_shift_token_ids(tok_sft, (3, 4)))

        model_dir = os.path.join(tmp_r4, "sft_r4_meter")
        os.makedirs(model_dir, exist_ok=True)
        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab_size": 700,
                    "d_model": 64,
                    "dim_feedforward": 128,
                    "num_heads": 4,
                    "d_layer": 2,
                    "dropout": 0.0,
                    "position_length": 512,
                    "min_length": 8,
                    "use_moe_decoder": False,
                    "norm_type": "rmsnorm",
                    "use_silu": True,
                    "use_bias": False,
                    "use_rope": True,
                },
                f,
            )
        model = MORTM(MORTMArgs(cfg_path), backend=cpu_backend)
        pth_path = os.path.join(model_dir, "model.pth")
        torch.save(model.state_dict(), pth_path)

        rapper_d = MORTM45DRapper(
            model_info={
                "model_folder_path": model_dir,
                "model_name": "MORTM4.5D-80M-SFT-Gen",
                "tag": {"instruments": ["PIANO", "SAX"], "model": "sft_gen", "task": ["meta", "infill", "meta_past"]},
            },
            backend=cpu_backend,
        )

        # Build Past MIDI with NO 0xFF 0x58 TimeSignature event (6 quarter-note beats = 2 bars of 3/4, which previously quantized as 1.5 bars of 4/4 emitting `s_72`!)
        mtrk_no_ts = bytearray(b"\x00\xff\x51\x03\x07\xa1\x20")
        for beat in range(6):
            delta_on = b"\x00" if beat == 0 else b"\x60"
            mtrk_no_ts.extend(delta_on + bytes([0x90, 60 + (beat % 4), 95]) + b"\x83\x00" + bytes([0x80, 60 + (beat % 4), 0]))
        mtrk_no_ts.extend(b"\x00\xff\x2f\x00")
        past_no_ts_path = os.path.join(tmp_r4, "past_no_ts.mid")
        with open(past_no_ts_path, "wb") as f:
            f.write(
                b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0MTrk"
                + len(mtrk_no_ts).to_bytes(4, "big")
                + bytes(mtrk_no_ts)
            )
        assert len(pretty_midi.PrettyMIDI(past_no_ts_path).time_signature_changes) == 0

        # Build Future MIDI WITH explicit 3/4 TimeSignature
        fut_34 = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        fut_34.time_signature_changes = [pretty_midi.TimeSignature(3, 4, 0.0)]
        inst_f = pretty_midi.Instrument(program=0)
        for beat in range(6):
            inst_f.notes.append(pretty_midi.Note(velocity=95, pitch=67, start=beat * 0.5, end=beat * 0.5 + 0.4))
        fut_34.instruments.append(inst_f)
        fut_34_path = os.path.join(tmp_r4, "future_34.mid")
        fut_34.write(fut_34_path)

        # Case A (`[R4-MED-02]`): Infill with `past` lacking 0xFF 0x58 + `future` having 3/4 + `meta` omitting TS
        meta_infill_auto = GenerateMeta(
            model_type="MORTM4.5D-80M-SFT-Gen",
            task="infill",
            program=["PIANO"],
            num_gems=1,
            genfield_measure=2,
            thinking=False,
        )
        prep_a = rapper_d.preprocessing(past_no_ts_path, None, fut_34_path, meta_infill_auto)
        assert prep_a["effective_time_sig"] == (3, 4)
        prompt_a = [int(t) for t in np.asarray(prep_a["sequence"][0]).flatten()]
        assert all(t not in forbidden_34 for t in prompt_a), (
            f"Infill Case A prompt leaked >=s_72 shift token: {[tok_sft.rev_get(t) for t in prompt_a if t in forbidden_34]}"
        )

        # Also verify Legacy `MORTM45Rapper` (`pretrained` & `generation`) unified multi-context meter resolution
        rapper_45_pre = MORTM45Rapper(
            model_info={
                "model_folder_path": model_dir,
                "model_name": "MORTM4.5-Pro",
                "tag": {"instruments": ["PIANO", "SAX"], "model": "pretrained", "task": ["Meta2MIDI"]},
            },
            backend=cpu_backend,
        )
        prep_45_a = rapper_45_pre.preprocessing(past_no_ts_path, None, fut_34_path, meta_infill_auto)
        assert prep_45_a["effective_time_sig"] == (3, 4)
        prompt_45_a = [int(t) for t in np.asarray(prep_45_a["sequence"][0]).flatten()]
        assert all(t not in forbidden_34 for t in prompt_45_a)

        # Case B (`[R4-MED-02]`): Explicit `meta` 3/4 override vs explicit `4/4` 0xFF 0x58 in context MIDI
        past_44 = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        past_44.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
        inst_44 = pretty_midi.Instrument(program=0)
        for beat in range(6):
            inst_44.notes.append(pretty_midi.Note(velocity=95, pitch=64, start=beat * 0.5, end=beat * 0.5 + 0.4))
        past_44.instruments.append(inst_44)
        past_44_path = os.path.join(tmp_r4, "past_44.mid")
        past_44.write(past_44_path)

        meta_override_34 = GenerateMeta(
            model_type="MORTM4.5D-80M-SFT-Gen",
            task="meta_past",
            program=["PIANO"],
            num_gems=1,
            genfield_measure=2,
            thinking=False,
            time_signature_numerator=3,
            time_signature_denominator=4,
        )
        prep_b = rapper_d.preprocessing(past_44_path, None, None, meta_override_34)
        assert prep_b["effective_time_sig"] == (3, 4)
        prompt_b = [int(t) for t in np.asarray(prep_b["sequence"][0]).flatten()]
        assert all(t not in forbidden_34 for t in prompt_b), (
            f"Explicit override Case B prompt leaked >=s_72 shift token: {[tok_sft.rev_get(t) for t in prompt_b if t in forbidden_34]}"
        )

        # Case C (`[R4-MED-02]`): Conflicting explicit context meters (3/4 vs 7/8) when `meta` omits TS -> ValueError
        fut_78 = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        fut_78.time_signature_changes = [pretty_midi.TimeSignature(7, 8, 0.0)]
        inst_78 = pretty_midi.Instrument(program=0)
        inst_78.notes.append(pretty_midi.Note(velocity=95, pitch=60, start=0.0, end=0.4))
        fut_78.instruments.append(inst_78)
        fut_78_path = os.path.join(tmp_r4, "future_78.mid")
        fut_78.write(fut_78_path)

        saw_conflict_err = False
        try:
            rapper_d.preprocessing(fut_34_path, None, fut_78_path, meta_infill_auto)
        except ValueError as exc:
            saw_conflict_err = "3/4" in str(exc) and "7/8" in str(exc)
        assert saw_conflict_err, "Expected ValueError when past=3/4 and future=7/8 conflict without explicit meta TS"

        # Case D (`[R4-LOW-02]`): `app._validate_mtrk_events` resets running_status on 0xFF Meta event
        # Active NoteOn (00 90 3C 64) -> Meta Text (00 FF 01 01 41) -> Running-status data byte (00 3C 00) MUST return False!
        bad_mtrk = b"\x00\x90\x3c\x64\x00\xff\x01\x01\x41\x00\x3c\x00\x00\xff\x2f\x00"
        assert app_mod._validate_mtrk_events(bad_mtrk, max_track_ticks=100_000) is False
        good_mtrk = b"\x00\x90\x3c\x64\x00\xff\x01\x01\x41\x00\x80\x3c\x00\x00\xff\x2f\x00"
        assert app_mod._validate_mtrk_events(good_mtrk, max_track_ticks=100_000) is True
    finally:
        shutil.rmtree(tmp_r4, ignore_errors=True)

    print("  [PASS] All Round-4 Fact-Checker backend & SMF remediations verified")


def test_10_round5_zero_banned_ast_imports_and_zero_multipart_warnings():
    """
    Round 5 Fact-Checker Remediations (`[R5-LOW-02]`):
      1. Zero AST `import` / `from ... import` of `flash_attn`, `loralib`, `music21`, or legacy `multipart`
         across every repository `.py` source file.
      2. `import python_multipart` executes cleanly with zero `PendingDeprecationWarning`.
    """
    print("\n--- [TEST 10] Round 5 Zero Banned AST Imports & Zero Deprecation Warnings ---")
    import ast
    import warnings
    from mortm.models.modules.backend import HAS_FLASH_ATTN

    assert HAS_FLASH_ATTN is False

    skip_dirs = {".venv", "node_modules", "__pycache__", ".git", "dist"}
    banned_roots = {"flash_attn", "loralib", "music21", "multipart"}
    violations = []
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            with open(fpath, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=fpath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in banned_roots:
                            violations.append((os.path.relpath(fpath, ROOT_DIR), node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    if top in banned_roots:
                        violations.append((os.path.relpath(fpath, ROOT_DIR), node.lineno, node.module))

    assert violations == [], f"Banned AST imports detected: {violations}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import python_multipart  # noqa: F401

        dep_warnings = [w for w in caught if issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))]
        assert dep_warnings == [], f"Unexpected multipart deprecation warnings: {dep_warnings}"

    print("  [PASS] Zero banned AST imports and zero multipart deprecation warnings verified")


def test_11_round6_critic_remediations():
    """
    Round 6 Verification-Critic Remediations (`[R3-HIGH-02]`, `[R3-MED-04]`, `[R3-MED-05]`, `[R3-LOW-03]`):
      1. `>=256`-bar MIDI & `max_clip_tokens` note-boundary safety (`convert.py`, `rapper.py`, `mortm45.py`, `mortm45d.py`)
      2. MSIX `LocalCache` Store->Win32 migration when `UNC=False` + active `major.minor` `Packages` selection (`model.py`)
      3. ASGI `asyncio.CancelledError` semaphore & `%TEMP%` leak immunity in `POST /generate` (`app.py`, `model.py`)
      4. `>128`-char `job_id` normalization parity, job-scoped `/generation_progress`, & whitespace CORS fallback (`app.py`, `progress.py`)
    """
    print("\n--- [TEST 11] Round 6 Verification-Critic Remediations ---")
    import asyncio
    from unittest.mock import patch

    import app as app_mod
    import model as model_mod
    from mortm.models.modules.progress import GLOBAL_GENERATION_PROGRESS
    from mortm.utils.convert import MIDI2Seq
    from rapper import AbstractModelRapper

    tok_omega = Tokenizer(omega_converter(TO_TOKEN))

    # 1. [R3-HIGH-02] >=256-bar MIDI preserves final d_24 in MORTM45Rapper.get_context + max_clip_tokens never slices mid-note
    pm_260 = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
    inst_260 = pretty_midi.Instrument(program=0)
    # 120 BPM -> 1 bar (4 beats) = 2.0 seconds. Place notes in Bar 1 (0.0s) and Bar 256 (index 255 -> 510.0s) and Bar 258 (514.0s)
    inst_260.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst_260.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=255 * 2.0, end=255 * 2.0 + 1.0))
    inst_260.notes.append(pretty_midi.Note(velocity=100, pitch=67, start=257 * 2.0, end=257 * 2.0 + 1.0))
    pm_260.instruments.append(inst_260)

    tmp_r6 = tempfile.mkdtemp(prefix="compass_r6_")
    try:
        mid_260_path = os.path.join(tmp_r6, "bars_260.mid")
        pm_260.write(mid_260_path)

        cfg_45 = {
            "vocab_size": 650,
            "d_model": 64,
            "dim_feedforward": 128,
            "num_heads": 4,
            "d_layer": 2,
            "dropout": 0.0,
            "position_length": 4096,
            "min_length": 8,
            "use_moe_decoder": False,
            "norm_type": "rmsnorm",
            "use_silu": True,
            "use_bias": False,
            "use_rope": False,
        }
        with open(os.path.join(tmp_r6, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg_45, fh)

        legacy_rapper = MORTM45Rapper(
            model_info={
                "model_folder_path": tmp_r6,
                "model_name": "MORTM4.5-Flash",
                "tag": {"instruments": ["PIANO"], "model": "pretrained", "task": ["Meta2MIDI"]},
            },
            backend=CpuHardwareBackend(),
        )
        conv_260 = MIDI2Seq(tok_omega, None, None, ["PIANO"], midi_data=pm_260)
        conv_260.convert()
        ctx_tokens = legacy_rapper.get_context(
            tok_omega, conv_260.aya_node, "<PAST_M>", max_total_tokens=2048
        )
        ctx_names = [tok_omega.rev_get(int(t)) for t in ctx_tokens]
        # ctx_names ends with [..., 's_0', 'p_64', 'd_24', '<ESEQ>', '<TAG_END>']
        non_meta_ctx = [nm for nm in ctx_names if nm not in ("<PAST_M>", "<INST_PIANO>", "<ESEQ>", "<TAG_END>", "<SME>")]
        assert non_meta_ctx[-1].startswith("d_"), (
            f">=256-bar MORTM45Rapper.get_context dropped final duration token! Tail={ctx_names[-8:]}"
        )
        prep_260 = legacy_rapper.preprocessing(
            mid_260_path,
            None,
            mid_260_path,
            GenerateMeta(
                model_type="MORTM4.5-Flash",
                task="infill",
                ai_continue_mode=True,
                program=["PIANO"],
            ),
        )
        prep_tok = prep_260["tokenizer"]
        prep_260_names = [prep_tok.rev_get(int(t)) for t in prep_260["sequence"][0]]
        eseq_idx = prep_260_names.index("<ESEQ>")
        assert prep_260_names[eseq_idx - 1].startswith("d_"), (
            f"preprocessing(infill) dropped final d_* before <ESEQ>: {prep_260_names[max(0, eseq_idx - 5):eseq_idx + 2]}"
        )
        prep_seed = legacy_rapper.preprocessing(
            mid_260_path,
            None,
            None,
            GenerateMeta(
                model_type="MORTM4.5-Flash",
                task="meta_past",
                ai_continue_mode=True,
                program=["PIANO"],
            ),
        )
        seed_tok = prep_seed["tokenizer"]
        seed_names = [seed_tok.rev_get(int(t)) for t in prep_seed["sequence"][0]]
        assert seed_names[-1].startswith("d_"), f"preprocessing(seed) dropped final d_*: {seed_names[-6:]}"

        # Tight max_clip_tokens=15 must NEVER emit an incomplete (s_*, p_*) trailing tuple
        pm_tight = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
        inst_tight = pretty_midi.Instrument(program=0)
        for i in range(10):
            inst_tight.notes.append(pretty_midi.Note(velocity=100, pitch=60 + i, start=i * 0.25, end=i * 0.25 + 0.2))
        pm_tight.instruments.append(inst_tight)
        conv_tight = MIDI2Seq(tok_omega, None, None, ["PIANO"], midi_data=pm_tight, max_clip_tokens=15)
        conv_tight.convert()
        tight_names = [tok_omega.rev_get(int(t)) for t in conv_tight.aya_node["PIANO"]]
        non_term_names = [nm for nm in tight_names if nm != "<TE>"]
        assert len(conv_tight.aya_node["PIANO"]) <= 16
        assert non_term_names[-1].startswith("d_"), f"max_clip_tokens cut mid-note: {tight_names}"

        # Unconditional trim_music_tokens_on_event_boundary strips dangling (s_0, p_60) even when len <= max_tokens
        dangling_seq = [
            tok_omega.get("<INST_PIANO>"),
            tok_omega.get("s_0"),
            tok_omega.get("p_60"),
            tok_omega.get("d_6"),
            tok_omega.get("s_12"),
            tok_omega.get("p_64"),  # missing d_*!
        ]
        sanitized = AbstractModelRapper.trim_music_tokens_on_event_boundary(
            tok_omega, dangling_seq, 64, keep_tail=True
        )
        sanitized_names = [tok_omega.rev_get(int(t)) for t in sanitized]
        assert sanitized_names == ["<INST_PIANO>", "s_0", "p_60", "d_6"], (
            f"Dangling incomplete note tuple was not stripped when len <= budget: {sanitized_names}"
        )

        # 2. [R3-MED-04] Store -> Win32 migration under _is_msix_python()=False & non-UNC path + multi-version Packages
        fake_local = os.path.join(tmp_r6, "Local")
        fake_roaming = os.path.join(tmp_r6, "Roaming")
        fake_host_models = os.path.join(fake_roaming, "COMPASS_AI", "models")
        shadow_311 = os.path.join(
            fake_local,
            "Packages",
            "PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0",
            "LocalCache",
            "Roaming",
            "COMPASS_AI",
            "models",
        )
        shadow_312 = os.path.join(
            fake_local,
            "Packages",
            "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0",
            "LocalCache",
            "Roaming",
            "COMPASS_AI",
            "models",
        )
        os.makedirs(os.path.join(shadow_311, "MORTM45D_80M_sft_gen"), exist_ok=True)
        os.makedirs(os.path.join(shadow_312, "MORTM45D_80M_sft_gen"), exist_ok=True)
        os.makedirs(fake_host_models, exist_ok=True)
        dummy_pth_src = os.path.join(shadow_312, "MORTM45D_80M_sft_gen", "model.pth")
        with open(dummy_pth_src, "wb") as fh:
            fh.write(b"DUMMY_WEIGHTS_R6")
        # Give 3.11 a newer mtime than 3.12 so version-matching (not mtime/alphabetical luck) is proven
        os.utime(shadow_312, (1_700_000_000, 1_700_000_000))
        os.utime(shadow_311, (1_800_000_000, 1_800_000_000))

        with patch.dict(os.environ, {"LOCALAPPDATA": fake_local, "APPDATA": fake_roaming}, clear=False):
            os.environ.pop("COMPASS_MODELS_DIR", None)
            # Active 3.12 MSIX Python without loopback UNC (Windows Home) must pick 3.12 shadow (not 3.11)
            with patch.object(model_mod, "_is_msix_python", return_value=True), patch.object(
                model_mod, "resolve_host_write_path", side_effect=lambda p: os.path.abspath(p)
            ), patch.object(
                model_mod.sys, "version_info", (3, 12, 8, "final", 0)
            ):
                shell_dir = model_mod.resolve_physical_models_dir_for_shell(fake_host_models)
                assert "Python.3.12_" in shell_dir, f"Expected 3.12 shadow path, got {shell_dir}"

            # Standard Win32 Python (_is_msix_python=False) with non-UNC path MUST still migrate Store shadow weights!
            with patch.object(model_mod, "_is_msix_python", return_value=False), patch.object(
                model_mod, "resolve_host_write_path", side_effect=lambda p: os.path.abspath(p)
            ):
                model_mod._clean_msix_localcache_shadow(
                    fake_host_models, os.path.join(ROOT_DIR, "data", "models")
                )
            migrated_pth = os.path.join(fake_host_models, "MORTM45D_80M_sft_gen", "model.pth")
            assert os.path.isfile(migrated_pth), "Store->Win32 LocalCache model.pth was skipped when UNC=False!"
            assert not os.path.exists(dummy_pth_src)

        # 3. [R3-MED-05] ASGI asyncio.CancelledError does NOT leak _GEN_QUEUE_SEMAPHORE or %TEMP% dirs
        worker_entered = threading.Event()
        worker_done = threading.Event()
        created_out_dirs = []

        def _fake_slow_generate(**kwargs):
            ce = kwargs["cancel_event"]
            worker_entered.set()
            # Wait until ASGI cancellation sets cancel_event
            ce.wait(timeout=3.0)
            out_d = tempfile.mkdtemp(prefix="compass_gen_r6test_")
            created_out_dirs.append(out_d)
            worker_done.set()
            raise GenerationCancelledError("Cancelled cleanly inside worker")

        class _DummyReq:
            headers = {}
            query_params = {}
            client = type("C", (), {"host": "127.0.0.1"})()

            async def is_disconnected(self):
                return False

        sem_before = app_mod._GEN_QUEUE_SEMAPHORE._value

        async def _run_cancelled_asgi():
            with patch.object(app_mod.model_controller, "generate", side_effect=_fake_slow_generate):
                coro = app_mod.generate_music(
                    request=_DummyReq(),
                    meta=json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta"}),
                    meta_json=None,
                    job_id="job_asgi_cancel_r6",
                    past_midi=None,
                    conditions_midi=None,
                    future_midi=None,
                )
                task = asyncio.create_task(coro)
                while not worker_entered.is_set():
                    await asyncio.sleep(0.01)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.run(_run_cancelled_asgi())
        assert worker_done.wait(timeout=3.0)
        # Wait briefly for worker finally block to finish releasing semaphore & cleaning temp dirs
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and app_mod._GEN_QUEUE_SEMAPHORE._value != sem_before:
            time.sleep(0.02)
        assert app_mod._GEN_QUEUE_SEMAPHORE._value == sem_before, (
            f"Semaphore leaked or double-released: expected {sem_before}, got {app_mod._GEN_QUEUE_SEMAPHORE._value}"
        )
        for d in created_out_dirs:
            shutil.rmtree(d, ignore_errors=True)

        # 4. [R3-LOW-03] >128-char job_id parity + job-scoped /generation_progress
        long_jid = "  " + ("j" * 180) + "  "
        norm_jid = app_mod._normalize_job_id(long_jid)
        assert norm_jid == "j" * 128

        client = TestClient(app)
        r_cancel_long = client.post("/cancel_generate", json={"job_id": long_jid})
        assert r_cancel_long.status_code == 200
        assert r_cancel_long.json()["job_id"] == "j" * 128
        # Registering that same >128-char job_id via normalized id immediately picks up tombstone!
        ev_long = model_controller.register_cancel_event(job_id=norm_jid)
        try:
            assert ev_long.is_set() is True
        finally:
            model_controller.unregister_cancel_event(ev_long, job_id=norm_jid)

        GLOBAL_GENERATION_PROGRESS.start_run("run_r6_a", "ジョブA推論中...", job_id="job_scope_a")
        try:
            r_prog_other = client.get("/generation_progress?job_id=job_scope_b")
            assert r_prog_other.status_code == 200 and r_prog_other.json()["active"] is False
            r_prog_same = client.get("/generation_progress?job_id=job_scope_a")
            assert r_prog_same.status_code == 200
            assert r_prog_same.json()["active"] is True
            assert r_prog_same.json()["message"] == "ジョブA推論中..."
        finally:
            GLOBAL_GENERATION_PROGRESS.update(active=False, stage="idle", message="", run_id="run_r6_a")

        # Whitespace-only COMPASS_CORS_ORIGINS falls back cleanly to _default_origins (never [])
        assert app_mod._resolve_allowed_cors_origins("  ,  ", "8000") == app_mod._default_origins
        assert app_mod._resolve_allowed_cors_origins("http://localhost:9123, ", "8000") == ["http://localhost:9123"]

        # 5. [R6B-MED-03] Tight keep_tail=False budgets NEVER leave a dangling bare <SME> before <ESEQ>
        multi_bar_music = [
            tok_omega.get("<SME>"),
            tok_omega.get("s_0"),
            tok_omega.get("p_60"),
            tok_omega.get("d_24"),
            tok_omega.get("<SME>"),
            tok_omega.get("s_0"),
            tok_omega.get("p_64"),
            tok_omega.get("d_24"),
        ]
        for b in range(1, 10):
            trimmed_b = AbstractModelRapper.trim_music_tokens_on_event_boundary(
                tok_omega, multi_bar_music, b, keep_tail=False
            )
            trimmed_b_names = [tok_omega.rev_get(int(t)) for t in trimmed_b]
            if trimmed_b_names:
                assert trimmed_b_names[-1] != "<SME>", (
                    f"Dangling bare <SME> at budget={b}: {trimmed_b_names}"
                )
        empty_ctx = legacy_rapper.get_context(
            tok_omega, {"PIANO": [tok_omega.get("<SME>"), tok_omega.get("<TE>")]}, "<FUTURE_M>", max_total_tokens=None
        )
        assert empty_ctx == [], f"Expected empty context when only bare <SME> remained, got {empty_ctx}"

        # 6. [R6B-MED-04] NTFS Junction alias (abspath != dst, samefile == True) MUST NOT self-delete model.pth
        junc_local = os.path.join(tmp_r6, "JuncLocal")
        junc_roaming = os.path.join(tmp_r6, "JuncRoaming")
        junc_host_compass = os.path.join(junc_roaming, "COMPASS_AI")
        junc_host_models = os.path.join(junc_host_compass, "models")
        junc_shadow_parent = os.path.join(
            junc_local,
            "Packages",
            "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0",
            "LocalCache",
            "Roaming",
        )
        os.makedirs(os.path.join(junc_host_models, "MORTM45D_80M_sft_gen"), exist_ok=True)
        os.makedirs(junc_shadow_parent, exist_ok=True)
        junc_pth = os.path.join(junc_host_models, "MORTM45D_80M_sft_gen", "model.pth")
        with open(junc_pth, "wb") as fh:
            fh.write(b"SURVIVE_JUNCTION_ALIAS_R6B")
        junc_shadow_compass = os.path.join(junc_shadow_parent, "COMPASS_AI")
        mklink_res = subprocess.run(
            ["cmd", "/c", "mklink", "/J", junc_shadow_compass, junc_host_compass],
            capture_output=True,
            text=True,
        )
        if mklink_res.returncode == 0 and os.path.isdir(junc_shadow_compass):
            try:
                assert os.path.abspath(junc_shadow_compass) != os.path.abspath(junc_host_compass)
                assert os.path.samefile(junc_shadow_compass, junc_host_compass)
                with patch.dict(
                    os.environ, {"LOCALAPPDATA": junc_local, "APPDATA": junc_roaming}, clear=False
                ):
                    os.environ.pop("COMPASS_MODELS_DIR", None)
                    with patch.object(model_mod, "_is_msix_python", return_value=False), patch.object(
                        model_mod, "resolve_host_write_path", side_effect=lambda p: os.path.abspath(p)
                    ):
                        model_mod._clean_msix_localcache_shadow(
                            junc_host_models, os.path.join(ROOT_DIR, "data", "models")
                        )
                assert os.path.isfile(junc_pth), (
                    "[R6B-MED-04] Junction-aliased shadow self-deleted live model.pth!"
                )
            finally:
                subprocess.run(
                    ["cmd", "/c", "rmdir", junc_shadow_compass],
                    capture_output=True,
                    check=False,
                )

        # 7. [R6B-HIGH-01] Queued /generate request cancelled BEFORE worker execution NEVER leaks semaphore/temp/events
        j1_entered = threading.Event()
        j1_release = threading.Event()
        j2_worker_ran = False

        def _blocking_j1_or_j2(**kwargs):
            nonlocal j2_worker_ran
            if kwargs.get("job_id") == "job_queued_2_r6b":
                j2_worker_ran = True
                raise GenerationCancelledError("J2 should have been cancelled in queue")
            j1_entered.set()
            j1_release.wait(timeout=3.0)
            raise GenerationCancelledError("J1 done")

        temp_root = tempfile.gettempdir()
        before_in_dirs = {
            nm for nm in os.listdir(temp_root) if nm.startswith("compass_in_")
        }
        sem_q_before = app_mod._GEN_QUEUE_SEMAPHORE._value

        async def _run_queued_cancel_test():
            with patch.object(app_mod.model_controller, "generate", side_effect=_blocking_j1_or_j2):
                t1 = asyncio.create_task(
                    app_mod.generate_music(
                        request=_DummyReq(),
                        meta=json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta"}),
                        meta_json=None,
                        job_id="job_active_1_r6b",
                        past_midi=None,
                        conditions_midi=None,
                        future_midi=None,
                    )
                )
                while not j1_entered.is_set():
                    await asyncio.sleep(0.01)

                # Now _INFER_EXECUTOR's single worker is occupied by t1; submit t2 so it queues in _INFER_EXECUTOR
                t2 = asyncio.create_task(
                    app_mod.generate_music(
                        request=_DummyReq(),
                        meta=json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta"}),
                        meta_json=None,
                        job_id="job_queued_2_r6b",
                        past_midi=None,
                        conditions_midi=None,
                        future_midi=None,
                    )
                )
                await asyncio.sleep(0.05)
                t2.cancel()
                try:
                    await t2
                except asyncio.CancelledError:
                    pass

                j1_release.set()
                try:
                    await t1
                except HTTPException:
                    pass

        asyncio.run(_run_queued_cancel_test())
        deadline_q = time.monotonic() + 2.0
        while time.monotonic() < deadline_q and app_mod._GEN_QUEUE_SEMAPHORE._value != sem_q_before:
            time.sleep(0.02)
        after_in_dirs = {
            nm for nm in os.listdir(temp_root) if nm.startswith("compass_in_")
        }
        leaked_in_dirs = after_in_dirs - before_in_dirs
        assert j2_worker_ran is False, "Queued J2 should not have executed on worker after cancel"
        assert app_mod._GEN_QUEUE_SEMAPHORE._value == sem_q_before, (
            f"Queued cancel leaked semaphore: before={sem_q_before}, after={app_mod._GEN_QUEUE_SEMAPHORE._value}"
        )
        assert not leaked_in_dirs, f"Queued cancel leaked %TEMP% dirs: {leaked_in_dirs}"
        assert "job_queued_2_r6b" not in app_mod.model_controller._pending_cancel_events

        # 8. [R6C-MED-01] Win32 \\?\ / \\.\ / \\?\UNC\ prefix disambiguation & MSIX Store data-safety guard
        if sys.platform.startswith("win"):
            ext_root = os.path.join(tmp_r6, "WinDevicePrefixes")
            ext_roaming = os.path.join(ext_root, "Roaming")
            ext_local = os.path.join(ext_root, "Local")
            ext_models = os.path.join(ext_roaming, "COMPASS_AI", "models")
            ext_shadow_models = os.path.join(
                ext_local,
                "Packages",
                "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0",
                "LocalCache",
                "Roaming",
                "COMPASS_AI",
                "models",
            )
            os.makedirs(os.path.join(ext_shadow_models, "MORTM45D_80M_sft_gen"), exist_ok=True)
            os.makedirs(ext_models, exist_ok=True)
            ext_shadow_pth = os.path.join(ext_shadow_models, "MORTM45D_80M_sft_gen", "model.pth")
            with open(ext_shadow_pth, "wb") as fh:
                fh.write(b"MSIX_EXT_PREFIX_WEIGHTS_SURVIVE")

            drv, tail_models = os.path.splitdrive(os.path.abspath(ext_models))
            ext_q_models = f"\\\\?\\{os.path.abspath(ext_models)}"
            ext_dot_models = f"\\\\.\\{os.path.abspath(ext_models)}"
            unc_models = f"\\\\127.0.0.1\\{drv[0]}${tail_models}"
            ext_unc_models = f"\\\\?\\UNC\\127.0.0.1\\{drv[0]}${tail_models}"
            localhost_unc_models = f"\\\\localhost\\{drv[0]}${tail_models}"

            assert model_mod._is_smb_unc_path(ext_models) is False
            assert model_mod._is_smb_unc_path(ext_q_models) is False
            assert model_mod._is_smb_unc_path(ext_dot_models) is False
            assert model_mod._is_smb_unc_path("\\\\?\\Volume{01234567-89ab-cdef-0123-456789abcdef}\\foo") is False
            assert model_mod._is_smb_unc_path(unc_models) is True
            assert model_mod._is_smb_unc_path(ext_unc_models) is True
            assert model_mod._is_smb_unc_path(localhost_unc_models) is True

            canon_base = model_mod._canonical_local_realpath(ext_models)
            assert model_mod._canonical_local_realpath(ext_q_models) == canon_base
            assert model_mod._canonical_local_realpath(ext_dot_models) == canon_base
            assert model_mod._canonical_local_realpath(unc_models) == canon_base
            assert model_mod._canonical_local_realpath(ext_unc_models) == canon_base
            assert model_mod._canonical_local_realpath(localhost_unc_models) == canon_base

            # Containment & non-existent file alias across \\?\C:\ and C:\ / \\127.0.0.1\C$\
            assert (
                model_mod._paths_alias_or_overlap(
                    ext_roaming, ext_q_models, check_containment=True
                )
                is True
            )
            nonexistent_sf = os.path.join(ext_q_models, "ghost.pth")
            nonexistent_df = os.path.join(unc_models, "ghost.pth")
            assert (
                model_mod._paths_alias_or_overlap(
                    nonexistent_sf, nonexistent_df, check_containment=False
                )
                is True
            )

            # MSIX Python with UNC unavailable + \\?\C:\ APPDATA MUST NOT delete LocalCache model.pth!
            with patch.dict(
                os.environ,
                {"APPDATA": f"\\\\?\\{os.path.abspath(ext_roaming)}", "LOCALAPPDATA": f"\\\\?\\{os.path.abspath(ext_local)}"},
                clear=False,
            ):
                os.environ.pop("COMPASS_MODELS_DIR", None)
                with patch.object(model_mod, "_is_msix_python", return_value=True), patch.object(
                    model_mod.os.path,
                    "isdir",
                    side_effect=lambda p, _orig=os.path.isdir: False
                    if str(p).startswith("\\\\127.0.0.1\\")
                    else _orig(p),
                ), patch.object(model_mod.sys, "version_info", (3, 12, 8, "final", 0)):
                    resolved_no_unc = model_mod.resolve_host_write_path(ext_q_models)
                    assert model_mod._is_smb_unc_path(resolved_no_unc) is False
                    shell_ext_dir = model_mod.resolve_physical_models_dir_for_shell(ext_q_models)
                    assert "Python.3.12_" in shell_ext_dir, f"Expected LocalCache 3.12 dir, got {shell_ext_dir}"
                    model_mod._clean_msix_localcache_shadow(
                        ext_q_models, os.path.join(ROOT_DIR, "data", "models")
                    )
                assert os.path.isfile(ext_shadow_pth), (
                    "[R6C-MED-01] MSIX Python without UNC deleted LocalCache model.pth under \\\\?\\ prefix!"
                )

        # 9. [R6C-LOW-01] MORTM45DRapper._trim_framed_context_tokens pre-empty & unclosed <INST_*> frame guards
        from models.mortm.mortm45d import MORTM45DRapper

        rapper_45d_probe = MORTM45DRapper.__new__(MORTM45DRapper)
        pre_empty_tokens = [tok_omega.get("<INST_PIANO>"), tok_omega.get("<ESEQ>")]
        assert (
            rapper_45d_probe._trim_framed_context_tokens(
                tok_omega, pre_empty_tokens, max_budget=32, keep_tail=False
            )
            == []
        )
        unclosed_tokens = [
            tok_omega.get("<INST_PIANO>"),
            tok_omega.get("<SME>"),
            tok_omega.get("s_0"),
            tok_omega.get("p_60"),
            tok_omega.get("d_24"),
            tok_omega.get("<INST_SAX>"),
            tok_omega.get("<SME>"),
            tok_omega.get("s_0"),
            tok_omega.get("p_64"),
            tok_omega.get("d_24"),
            tok_omega.get("<ESEQ>"),
        ]
        reframed_unclosed = rapper_45d_probe._trim_framed_context_tokens(
            tok_omega, unclosed_tokens, max_budget=64, keep_tail=False
        )
        reframed_names = [tok_omega.rev_get(int(t)) for t in reframed_unclosed]
        assert reframed_names == [
            "<INST_PIANO>",
            "<SME>",
            "s_0",
            "p_60",
            "d_24",
            "<ESEQ>",
            "<INST_SAX>",
            "<SME>",
            "s_0",
            "p_64",
            "d_24",
            "<ESEQ>",
        ], f"Unexpected reframed tokens: {reframed_names}"
    finally:
        shutil.rmtree(tmp_r6, ignore_errors=True)

    print("  [PASS] All Round-6 Verification-Critic backend remediations verified")


def test_12_round5_critic_r4_remediations():
    """
    Physical regression suite for all Round-4 Verification-Critic backend findings:
      - [R4-HIGH-01] Subdirectory NTFS junction containment-alias bomb (`models\\models\\...`) & sibling `%APPDATA%` exfiltration blocked
      - [R4-MED-01] Orphan leading `<BLANK>` after `<INST_*>` when slicing `start_measure > 0` on empty bar eliminated
      - [R4-MED-02] Unified meter validation (`5/4`, `6/4`, `0/4`, `1/16`, `4/3`) + fail-fast pre-lock rejection
      - [R4-MED-04] `POST /cancel_generate` zero silent global kill-all escalation + pre-strip `?job_id=%20%20` shadowing fix
      - [R4-MED-06] `9600 PPQN` 55-bar Pro Tools SMF (`2,112,000` ticks) accepted & rescaled to `480 PPQN`
      - [R4-LOW-01] Cross-version MSIX `LocalCache` isolation + `\\\\?\\Volume{GUID}\\` device-prefix stripping
    """
    from fastapi.testclient import TestClient
    from pydantic import ValidationError
    from unittest.mock import patch
    import re
    import struct
    import subprocess
    import app as app_mod
    import model as model_mod
    from models.mortm.mortm45d import MORTM45DRapper
    from mortm.models.modules.progress import GLOBAL_GENERATION_PROGRESS
    from mortm.train.custom_token import validate_supported_time_signature
    from mortm.utils.convert import MIDIConverter
    from mortm.utils.de_convert import ct_token_to_midi
    from rapper import AbstractModelRapper

    print("\n--- [TEST 12] Round-5 Verification-Critic (R4) Backend Remediations ---")
    tmp_r4 = tempfile.mkdtemp(prefix="compass_r4_critic_")
    try:
        # 1. [R4-HIGH-01] Subdirectory NTFS junction containment-alias bomb & sibling %APPDATA% exfiltration
        if sys.platform.startswith("win"):
            for mode in ("self_containment", "sibling_roaming"):
                sub_root = os.path.join(tmp_r4, f"junc_{mode}")
                roaming = os.path.join(sub_root, "Roaming")
                local = os.path.join(sub_root, "Local")
                host_models = os.path.join(roaming, "COMPASS_AI", "models")
                host_45d = os.path.join(host_models, "MORTM45D_80M_sft_gen")
                os.makedirs(host_45d, exist_ok=True)
                host_pth = os.path.join(host_45d, "model.pth")
                with open(host_pth, "wb") as fh:
                    fh.write(b"HOST_WEIGHTS_SAFE")

                sibling_dir = os.path.join(roaming, "Bitwarden")
                os.makedirs(sibling_dir, exist_ok=True)
                sibling_secret = os.path.join(sibling_dir, "vault.json")
                with open(sibling_secret, "wb") as fh:
                    fh.write(b'{"secret": "DO_NOT_EXFILTRATE"}')

                shadow_compass = os.path.join(
                    local,
                    "Packages",
                    "PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0",
                    "LocalCache",
                    "Roaming",
                    "COMPASS_AI",
                )
                os.makedirs(shadow_compass, exist_ok=True)
                shadow_junc = os.path.join(shadow_compass, "models")
                junc_target = os.path.join(roaming, "COMPASS_AI") if mode == "self_containment" else roaming
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", shadow_junc, junc_target],
                    check=True,
                    capture_output=True,
                )
                try:
                    assert model_mod._is_reparse_point(shadow_junc) is True
                    with patch.dict(
                        os.environ, {"LOCALAPPDATA": local, "APPDATA": roaming}, clear=False
                    ):
                        os.environ.pop("COMPASS_MODELS_DIR", None)
                        with patch.object(model_mod, "_is_msix_python", return_value=False):
                            model_mod._clean_msix_localcache_shadow(
                                host_models, os.path.join(ROOT_DIR, "data", "models")
                            )
                    assert not os.path.exists(os.path.join(host_models, "models")), (
                        f"[R4-HIGH-01] Self-referential nested models/models created in mode={mode}!"
                    )
                    assert not os.path.exists(os.path.join(host_models, "Bitwarden", "vault.json")), (
                        "[R4-HIGH-01] Sibling Roaming Bitwarden/vault.json exfiltrated into COMPASS_AI/models!"
                    )
                    assert os.path.isfile(host_pth)
                    assert os.path.isfile(sibling_secret)
                finally:
                    subprocess.run(
                        ["cmd", "/c", "rmdir", shadow_junc],
                        check=False,
                        capture_output=True,
                    )

        # 2. [R4-MED-01] Orphan leading <BLANK> when slicing start_measure > 0 on an empty bar
        tok_omega = Tokenizer(omega_converter(TO_TOKEN))
        pm_bar1_empty = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
        inst_b1 = pretty_midi.Instrument(program=0)
        # Bar 0 (0.0s..0.5s) has a note; Bar 1 (2.0s..4.0s) is empty; Bar 2 (4.0s..4.5s) has a note
        inst_b1.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        inst_b1.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=4.0, end=4.5))
        pm_bar1_empty.instruments.append(inst_b1)
        bar1_empty_path = os.path.join(tmp_r4, "bar1_empty.mid")
        pm_bar1_empty.write(bar1_empty_path)

        rapper_45d = MORTM45DRapper.__new__(MORTM45DRapper)
        sliced_m1_m2, found_m1_m2, _ = rapper_45d._extract_context_midi_tokens(
            tok_omega,
            bar1_empty_path,
            start_measure=1,
            measure_count=2,
            default_time_signature=(4, 4),
        )
        sliced_names = [tok_omega.rev_get(int(t)) for t in sliced_m1_m2]
        assert sliced_names == [
            "<INST_PIANO>",
            "<SME>",
            "s_0",
            "p_64",
            "d_24",
            "<ESEQ>",
        ], f"[R4-MED-01] Unexpected tokens when Bar 1 is empty: {sliced_names}"
        assert found_m1_m2 == ["PIANO"]

        # Slicing ONLY empty Bar 1 (`start_measure=1, measure_count=1`) must emit [] (never zero-note <INST_PIANO><BLANK><ESEQ>)
        sliced_m1_only, found_m1_only, _ = rapper_45d._extract_context_midi_tokens(
            tok_omega,
            bar1_empty_path,
            start_measure=1,
            measure_count=1,
            default_time_signature=(4, 4),
        )
        assert sliced_m1_only == [] and found_m1_only == [], (
            f"[R4-MED-01] Expected [] for empty-bar-only slice, got {sliced_m1_only}"
        )

        # Boundary trimmer (`keep_tail=True`) landing on <BLANK> must never leave orphan leading <BLANK>
        inner_with_blank = [
            tok_omega.get("<SME>"),
            tok_omega.get("s_0"),
            tok_omega.get("p_60"),
            tok_omega.get("d_24"),
            tok_omega.get("<SME>"),
            tok_omega.get("<BLANK>"),
            tok_omega.get("<SME>"),
            tok_omega.get("s_0"),
            tok_omega.get("p_67"),
            tok_omega.get("d_24"),
        ]
        trimmed_tail = AbstractModelRapper.trim_music_tokens_on_event_boundary(
            tok_omega, inner_with_blank, 5, keep_tail=True
        )
        trimmed_tail_names = [tok_omega.rev_get(int(t)) for t in trimmed_tail]
        assert trimmed_tail_names == ["s_0", "p_67", "d_24"], trimmed_tail_names

        # 3. [R4-MED-02] Unified meter validation + fail-fast pre-lock rejection
        assert validate_supported_time_signature(4, 4) == (4, 4)
        assert validate_supported_time_signature(7, 8) == (7, 8)
        assert validate_supported_time_signature(12, 16) == (12, 16)
        assert validate_supported_time_signature(1, 8) == (1, 8)
        for bad_num, bad_den in [(5, 4), (6, 4), (12, 8), (0, 4), (4, 0), (1, 16), (4, 3), (True, 4)]:
            try:
                validate_supported_time_signature(bad_num, bad_den)
                raise AssertionError(f"Expected meter ({bad_num}/{bad_den}) to be rejected!")
            except ValueError:
                pass

        for bad_num, bad_den in [(5, 4), (6, 4), (0, 4), (1, 16), (4, 3)]:
            try:
                GenerateMeta(
                    model_type="MORTM4.5D-80M-SFT-Gen",
                    time_signature_numerator=bad_num,
                    time_signature_denominator=bad_den,
                )
                raise AssertionError(f"GenerateMeta accepted unsupported meter {bad_num}/{bad_den}!")
            except ValidationError:
                pass

        # Explicit 0 must NOT silently coerce to 4 via `or 4`
        bogus_meta = GenerateMeta.model_construct(
            model_type="MORTM4.5D-80M-SFT-Gen",
            time_signature_numerator=0,
            time_signature_denominator=4,
        )
        try:
            AbstractModelRapper.resolve_time_signature(bogus_meta)
            raise AssertionError("resolve_time_signature coerced explicit 0 -> 4!")
        except ValueError:
            pass

        pm_54 = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
        pm_54.time_signature_changes.append(pretty_midi.TimeSignature(5, 4, 0.0))
        inst_54 = pretty_midi.Instrument(program=0)
        inst_54.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5))
        pm_54.instruments.append(inst_54)
        midi_54_path = os.path.join(tmp_r4, "five_four_ctx.mid")
        pm_54.write(midi_54_path)

        conv_54 = MIDIConverter(tok_omega, directory="", file_name=midi_54_path, program_list=["PIANO"])
        assert conv_54.is_error is True and "拍子" in (conv_54.error_reason or "")
        conv_54_forced = MIDIConverter(
            tok_omega,
            directory="",
            file_name=midi_54_path,
            program_list=["PIANO"],
            force_time_signature=(4, 4),
        )
        assert conv_54_forced.is_error is False and conv_54_forced.time_signature == (4, 4)

        try:
            ct_token_to_midi(
                tok_omega,
                np.array([tok_omega.get("<INST_PIANO>"), tok_omega.get("<ESEQ>")], dtype=np.int64),
                os.path.join(tmp_r4, "bad_54_out.mid"),
                tempo=120,
                time_signature=(5, 4),
            )
            raise AssertionError("ct_token_to_midi accepted 5/4!")
        except ValueError:
            pass

        client = TestClient(app)
        with open(midi_54_path, "rb") as fh_54:
            raw_54 = fh_54.read()
        with patch.object(
            app_mod.model_controller,
            "_get_or_create_rapper",
            side_effect=AssertionError("_get_or_create_rapper must NOT run for 5/4 MIDI!"),
        ):
            r_54_gen = client.post(
                "/generate",
                data={"meta": json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta_past"})},
                files={"past_midi": ("five_four.mid", raw_54, "audio/midi")},
            )
        assert r_54_gen.status_code == 400 and "拍子" in r_54_gen.json()["detail"], (
            f"Expected fail-fast HTTP 400 for 5/4 MIDI, got {r_54_gen.status_code}: {r_54_gen.text}"
        )

        # 4. [R4-MED-04] POST /cancel_generate zero silent global kill-all escalation + whitespace query shadowing fix
        ev_innocent = app_mod.model_controller.register_cancel_event(job_id="job_innocent_r4")
        ev_target = app_mod.model_controller.register_cancel_event(job_id="job_target_r4")
        try:
            assert client.post("/cancel_generate", json={"job_id": "   "}).status_code == 400
            assert client.post("/cancel_generate", json={"job_id": 123}).status_code == 400
            assert client.post("/cancel_generate", json={"job_id": "bad id!"}).status_code == 400
            assert (
                client.post(
                    "/cancel_generate",
                    content=b"{corrupt_json",
                    headers={"Content-Type": "application/json"},
                ).status_code
                == 400
            )
            assert client.post("/cancel_generate", json={"jobId_typo": "job_target_r4"}).status_code == 400
            assert (
                client.post(
                    "/cancel_generate",
                    json={"job_id": "job_target_r4", "pad": "x" * 5000},
                ).status_code
                == 413
            )
            assert client.post("/cancel_generate?job_id=%20%20").status_code == 400
            assert ev_innocent.is_set() is False, "[R4-MED-04] Innocent sibling job was killed!"
            assert ev_target.is_set() is False

            # Whitespace ?job_id=%20%20 must NOT shadow valid X-Compass-Job-Id header
            r_hdr_fallthrough = client.post(
                "/cancel_generate?job_id=%20%20",
                headers={"X-Compass-Job-Id": "job_target_r4"},
            )
            assert r_hdr_fallthrough.status_code == 200
            assert r_hdr_fallthrough.json()["job_id"] == "job_target_r4"
            assert ev_target.is_set() is True
            assert ev_innocent.is_set() is False, "[R4-MED-04] Whitespace query shadowed header into kill-all!"

            GLOBAL_GENERATION_PROGRESS.start_run("run_r4_hdr", "進行中...", job_id="job_innocent_r4")
            try:
                r_prog_hdr = client.get(
                    "/generation_progress?job_id=%20%20",
                    headers={"X-Compass-Job-Id": "job_innocent_r4"},
                )
                assert r_prog_hdr.status_code == 200 and r_prog_hdr.json()["active"] is True
            finally:
                GLOBAL_GENERATION_PROGRESS.update(
                    active=False, stage="done", message="Done", run_id="run_r4_hdr"
                )
        finally:
            app_mod.model_controller.unregister_cancel_event(ev_innocent, job_id="job_innocent_r4")
            app_mod.model_controller.unregister_cancel_event(ev_target, job_id="job_target_r4")

        # 5. [R4-MED-06] 9600 PPQN 55-bar Pro Tools SMF (2,112,000 ticks = 220 quarter notes) accepted & rescaled to 480 PPQN
        ticks_55_bars_9600 = 55 * 4 * 9600  # 2,112,000 ticks
        vlq_2112000 = app_mod._encode_smf_vlq(ticks_55_bars_9600)
        mtrk_9600 = (
            b"\x00\xff\x51\x03\x07\xa1\x20"  # 120 BPM
            + vlq_2112000
            + b"\x90\x3c\x64"
            + app_mod._encode_smf_vlq(9600)
            + b"\x80\x3c\x00\x00\xff\x2f\x00"
        )
        smf_9600 = (
            struct.pack(">4sIHHH", b"MThd", 6, 0, 1, 9600)
            + struct.pack(">4sI", b"MTrk", len(mtrk_9600))
            + mtrk_9600
        )
        assert app_mod._validate_smf_binary_structure(smf_9600) is True, (
            "[R4-MED-06] Valid 55-bar 9600 PPQN Pro Tools SMF was rejected!"
        )
        norm_480_bytes = app_mod._normalize_high_ppqn_smf_bytes(smf_9600, target_ppqn=480)
        norm_480_path = os.path.join(tmp_r4, "protools_9600_norm.mid")
        with open(norm_480_path, "wb") as fh:
            fh.write(norm_480_bytes)
        pm_norm = pretty_midi.PrettyMIDI(norm_480_path)
        assert pm_norm.resolution == 480
        assert len(pm_norm.instruments) == 1 and len(pm_norm.instruments[0].notes) == 1
        assert abs(pm_norm.instruments[0].notes[0].start - 110.0) < 1e-3

        # >4160 quarter notes at 9600 PPQN (e.g. 50,000,000 ticks) still rejected in <5ms
        mtrk_9600_bomb = app_mod._encode_smf_vlq(50_000_000) + b"\x90\x3c\x64\x00\x80\x3c\x00\x00\xff\x2f\x00"
        smf_9600_bomb = (
            struct.pack(">4sIHHH", b"MThd", 6, 0, 1, 9600)
            + struct.pack(">4sI", b"MTrk", len(mtrk_9600_bomb))
            + mtrk_9600_bomb
        )
        assert app_mod._validate_smf_binary_structure(smf_9600_bomb) is False

        # 6. [R4-LOW-01] Cross-version MSIX LocalCache isolation & \\?\Volume{GUID}\ prefix stripping
        if sys.platform.startswith("win"):
            iso_roaming = os.path.join(tmp_r4, "IsoRoaming")
            iso_local = os.path.join(tmp_r4, "IsoLocal")
            iso_models_nonexistent = os.path.join(iso_roaming, "COMPASS_AI", "models")
            stale_310_shadow = os.path.join(
                iso_local,
                "Packages",
                "PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0",
                "LocalCache",
                "Roaming",
                "COMPASS_AI",
                "models",
            )
            os.makedirs(stale_310_shadow, exist_ok=True)
            with patch.dict(
                os.environ, {"APPDATA": iso_roaming, "LOCALAPPDATA": iso_local}, clear=False
            ):
                os.environ.pop("COMPASS_MODELS_DIR", None)
                # Standard Win32 Python (even before iso_models_nonexistent exists) must return abs_dir
                with patch.object(model_mod, "_is_msix_python", return_value=False):
                    assert model_mod.resolve_physical_models_dir_for_shell(
                        iso_models_nonexistent
                    ) == os.path.abspath(iso_models_nonexistent)
                # MSIX Python 3.13 without UNC and without a 3.13 LocalCache must NOT fall back to 3.10 LocalCache
                with patch.object(model_mod, "_is_msix_python", return_value=True), patch.object(
                    model_mod.os.path,
                    "isdir",
                    side_effect=lambda p, _orig=os.path.isdir: False
                    if str(p).startswith("\\\\127.0.0.1\\")
                    else _orig(p),
                ), patch.object(model_mod.sys, "version_info", (3, 13, 1, "final", 0)):
                    assert model_mod.resolve_physical_models_dir_for_shell(
                        iso_models_nonexistent
                    ) == os.path.abspath(iso_models_nonexistent)

            mv_out = subprocess.check_output(["mountvol"], text=True)
            m_vol = re.search(r"(\\\\\?\\Volume\{[0-9a-fA-F-]+\}\\)[\s\r\n]+([A-Za-z]):\\", mv_out)
            if m_vol:
                vol_prefix, drv_letter = m_vol.group(1), m_vol.group(2)
                vol_test_path = vol_prefix + r"Users\Public\compass_vol_probe"
                stripped_vol = model_mod._strip_win_device_prefix(vol_test_path)
                assert stripped_vol.lower() == f"{drv_letter.lower()}:\\users\\public\\compass_vol_probe", (
                    f"Expected Volume GUID stripped to {drv_letter}:\\..., got {stripped_vol}"
                )
    finally:
        shutil.rmtree(tmp_r4, ignore_errors=True)

    print("  [PASS] All Round-5 Verification-Critic (R4) backend remediations verified")


def test_13_round6_fact_checker_r5_remediations():
    """
    Round-6 physical regression tests for ALL 4 backend Round-5 Fact-Checker findings:
    - [R5-HIGH-01] `POST /cancel_generate` rejects unscoped empty/whitespace/`{}` bodies, non-bool `cancel_all`,
      duplicate `getlist("job_id")` / `getlist("x-compass-job-id")`, and `cancel_all` + query/header mixes
      without ever killing sibling jobs; explicit `{"cancel_all": True}` alone succeeds with HTTP 200
    - [R5-MED-01] `_extract_context_midi_tokens` returns `([], [])` when `start_measure >= total_measures`
      or `measure_count <= 0`, preventing backward-clamped Bar 0 hallucinations and multi-track desync
    - [R5-MED-02] Linear SMPTE SMF (`0x8000` division) ignores `0xFF 0x51` Set Tempo (with lossless delta carry)
      and avoids `max(24, ...)` clamping on low-subframe SMPTE (`24fps * 1tpf`) for exact `1.0s` wall-clock parity
    - [R5-LOW-01] `validate_supported_time_signature` raises `ValueError` (never `OverflowError`) on
      `inf`, `-inf`, `nan`, `1e300`, `10**400`, `1 << 20000`, and 10,000-digit strings
    """
    print("\n--- [TEST 13] Round-6 Fact-Checker (R5) Backend Remediations ---")
    import struct
    import app as app_mod
    from models.mortm.mortm45d import MORTM45DRapper
    from mortm.train.custom_token import validate_supported_time_signature
    from mortm.train.tokenizer import TO_TOKEN, Tokenizer, omega_converter

    tmp_r5 = tempfile.mkdtemp(prefix="compass_r5_audit_")
    try:
        # 1. [R5-HIGH-01] POST /cancel_generate strict job_id / explicit bool cancel_all enforcement
        with app_mod._cancel_limiter._lock:
            app_mod._cancel_limiter._buckets["loopback"].clear()

        client = TestClient(app)
        ev_innocent = app_mod.model_controller.register_cancel_event(job_id="job_innocent_r5")
        ev_target = app_mod.model_controller.register_cancel_event(job_id="job_target_r5")
        try:
            assert client.post("/cancel_generate").status_code == 400
            assert (
                client.post(
                    "/cancel_generate",
                    content=b"   ",
                    headers={"Content-Type": "application/json"},
                ).status_code
                == 400
            )
            assert client.post("/cancel_generate", json={}).status_code == 400
            assert client.post("/cancel_generate", json={"cancel_all": 1}).status_code == 400
            assert client.post("/cancel_generate", json={"cancel_all": 1.0}).status_code == 400
            assert client.post("/cancel_generate", json={"cancel_all": False}).status_code == 400
            assert (
                client.post(
                    "/cancel_generate", json={"cancel_all": True, "job_id": "job_target_r5"}
                ).status_code
                == 400
            )
            assert (
                client.post(
                    "/cancel_generate?job_id=job_target_r5&job_id=job_innocent_r5"
                ).status_code
                == 400
            )
            assert (
                client.post(
                    "/cancel_generate?job_id=%20%20", json={"cancel_all": True}
                ).status_code
                == 400
            )
            assert (
                client.post(
                    "/cancel_generate",
                    headers={"X-Compass-Job-Id": "   "},
                    json={"cancel_all": True},
                ).status_code
                == 400
            )
            assert ev_innocent.is_set() is False, "[R5-HIGH-01] Innocent sibling job was killed!"
            assert ev_target.is_set() is False

            # Scoped target cancel cancels ONLY ev_target
            r_scoped = client.post("/cancel_generate", json={"job_id": "job_target_r5"})
            assert r_scoped.status_code == 200 and r_scoped.json()["cancelled"] is True
            assert ev_target.is_set() is True
            assert ev_innocent.is_set() is False

            # Explicit {"cancel_all": True} (exact bool True, no query/header) cancels remaining active jobs
            r_all = client.post("/cancel_generate", json={"cancel_all": True})
            assert r_all.status_code == 200 and r_all.json()["cancelled"] is True
            assert ev_innocent.is_set() is True
        finally:
            app_mod.model_controller.unregister_cancel_event(ev_innocent, job_id="job_innocent_r5")
            app_mod.model_controller.unregister_cancel_event(ev_target, job_id="job_target_r5")

        # 2. [R5-MED-01] _extract_context_midi_tokens never clamps backward when start_measure >= total_measures
        tok_omega = Tokenizer(omega_converter(TO_TOKEN))
        rapper_45d = MORTM45DRapper.__new__(MORTM45DRapper)

        pm_bar0_only = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
        inst_b0 = pretty_midi.Instrument(program=0)
        inst_b0.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        pm_bar0_only.instruments.append(inst_b0)
        bar0_only_path = os.path.join(tmp_r5, "bar0_only.mid")
        pm_bar0_only.write(bar0_only_path)

        toks_past_end, progs_past_end, _ = rapper_45d._extract_context_midi_tokens(
            tok_omega,
            bar0_only_path,
            start_measure=1,
            measure_count=1,
            default_time_signature=(4, 4),
        )
        assert toks_past_end == [] and progs_past_end == [], (
            f"[R5-MED-01] Expected ([], []) when start_measure >= total_measures, got {toks_past_end}, {progs_past_end}"
        )

        toks_zero_cnt, progs_zero_cnt, _ = rapper_45d._extract_context_midi_tokens(
            tok_omega,
            bar0_only_path,
            start_measure=0,
            measure_count=0,
            default_time_signature=(4, 4),
        )
        assert toks_zero_cnt == [] and progs_zero_cnt == []

        # Multi-track: PIANO plays in Bar 0 & Bar 2; ALTO SAX (program 65) stops after Bar 0
        pm_multi = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
        inst_piano = pretty_midi.Instrument(program=0)
        inst_piano.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        inst_piano.notes.append(pretty_midi.Note(velocity=100, pitch=67, start=4.0, end=4.5))
        inst_sax = pretty_midi.Instrument(program=65)
        inst_sax.notes.append(pretty_midi.Note(velocity=100, pitch=72, start=0.0, end=0.5))
        pm_multi.instruments.extend([inst_piano, inst_sax])
        multi_path = os.path.join(tmp_r5, "multi_trailing_silence.mid")
        pm_multi.write(multi_path)

        toks_multi, progs_multi, _ = rapper_45d._extract_context_midi_tokens(
            tok_omega,
            multi_path,
            start_measure=1,
            measure_count=2,
            default_time_signature=(4, 4),
        )
        tok_names_multi = [tok_omega.rev_get(int(t)) for t in toks_multi]
        assert progs_multi == ["PIANO"], f"[R5-MED-01] Expected only ['PIANO'], got {progs_multi}"
        assert "<INST_SAX>" not in tok_names_multi and "p_72" not in tok_names_multi

        # 3. [R5-MED-02] Linear SMPTE SMF wall-clock parity (0xFF 0x51 ignored with lossless delta carry; 24fps*1tpf unclamped)
        # Subcase A: -25 fps * 40 tpf (0xE728 -> 1000 ticks/sec, tpq=500.0) + 250-tick delta before 60 BPM FF 51 + 250-tick delta before 1000-tick note
        mtrk_smpte_25_40 = (
            app_mod._encode_smf_vlq(250)  # 0.25s before Set Tempo
            + b"\xff\x51\x03\x0f\x42\x40"  # 60 BPM (1,000,000 us/quarter) -> MUST be ignored on SMPTE!
            + app_mod._encode_smf_vlq(250)  # +0.25s -> cumulative 0.50s NoteOn
            + b"\x90\x3c\x64"
            + app_mod._encode_smf_vlq(1000)  # +1.00s -> cumulative 1.50s NoteOff
            + b"\x80\x3c\x00\x00\xff\x2f\x00"
        )
        smf_smpte_25_40 = (
            struct.pack(">4sIHHH", b"MThd", 6, 0, 1, 0xE728)
            + struct.pack(">4sI", b"MTrk", len(mtrk_smpte_25_40))
            + mtrk_smpte_25_40
        )
        assert app_mod._validate_smf_binary_structure(smf_smpte_25_40) is True
        norm_smpte_25_40 = app_mod._normalize_high_ppqn_smf_bytes(smf_smpte_25_40, target_ppqn=480)
        smpte_25_40_path = os.path.join(tmp_r5, "smpte_25_40_norm.mid")
        with open(smpte_25_40_path, "wb") as fh:
            fh.write(norm_smpte_25_40)
        pm_smpte_a = pretty_midi.PrettyMIDI(smpte_25_40_path)
        note_a = pm_smpte_a.instruments[0].notes[0]
        assert abs(note_a.start - 0.5) < 1e-3, f"[R5-MED-02] Delta before skipped FF 51 lost! start={note_a.start}"
        assert abs(note_a.end - 1.5) < 1e-3, f"[R5-MED-02] SMPTE FF 51 distorted note end! end={note_a.end}"

        # Subcase B: -24 fps * 1 tpf (0xE801 -> 24 ticks/sec, tpq=12.0) + 24-tick (1.0s) note
        mtrk_smpte_24_1 = (
            b"\x00\x90\x40\x64"
            + app_mod._encode_smf_vlq(24)
            + b"\x80\x40\x00\x00\xff\x2f\x00"
        )
        smf_smpte_24_1 = (
            struct.pack(">4sIHHH", b"MThd", 6, 0, 1, 0xE801)
            + struct.pack(">4sI", b"MTrk", len(mtrk_smpte_24_1))
            + mtrk_smpte_24_1
        )
        assert app_mod._validate_smf_binary_structure(smf_smpte_24_1) is True
        norm_smpte_24_1 = app_mod._normalize_high_ppqn_smf_bytes(smf_smpte_24_1, target_ppqn=480)
        smpte_24_1_path = os.path.join(tmp_r5, "smpte_24_1_norm.mid")
        with open(smpte_24_1_path, "wb") as fh:
            fh.write(norm_smpte_24_1)
        pm_smpte_b = pretty_midi.PrettyMIDI(smpte_24_1_path)
        note_b = pm_smpte_b.instruments[0].notes[0]
        assert abs(note_b.start - 0.0) < 1e-3 and abs(note_b.end - 1.0) < 1e-3, (
            f"[R5-MED-02] Low-subframe 24fps*1tpf SMPTE distorted! start={note_b.start}, end={note_b.end}"
        )

        # 4. [R5-LOW-01] validate_supported_time_signature never leaks OverflowError on inf/nan/huge ints
        for bad_num, bad_den in [
            (float("inf"), 4),
            (4, float("inf")),
            (float("-inf"), 4),
            (float("nan"), 4),
            (4, float("nan")),
            (1e300, 4),
            (10**400, 4),
            (4, 10**400),
            (1 << 20000, 4),
            ("9" * 10000, 4),
        ]:
            try:
                validate_supported_time_signature(bad_num, bad_den)
                raise AssertionError(f"Expected ValueError for ({bad_num!r}, {bad_den!r})")
            except ValueError:
                pass
    finally:
        shutil.rmtree(tmp_r5, ignore_errors=True)

    print("  [PASS] All Round-6 Fact-Checker (R5) backend remediations verified")


def test_14_round7_fact_checker_r6_remediations():
    """
    Round-7 physical regression tests for ALL 4 backend Round-6 Fact-Checker findings:
    - [R6-MED-01] Symmetric fail-closed `job_id` validation across `GET /generation_progress`, `POST /cancel_generate`,
      and `POST /generate` (rejecting malformed IDs, comma-folded headers, duplicate query/header/form entries, and
      source conflicts with HTTP 400; masking `job_id`/`run_id` on genuinely unscoped `/generation_progress`)
    - [R6-MED-02] `MORTM45DRapper.preprocessing` reconciles `task_str` when silent / out-of-range context slices
      produce `[]` (`infill` -> `meta_past` / `meta_future` / `meta`, `meta_past`/`meta_future`/`inst_comp` -> `meta`)
      so zero-context runs never emit an OOD `<thinking>` CoT prompt without conditioning blocks
    - [R6-LOW-01] Fractional SMPTE (`29.97` drop-frame `0xE328` / `0xE301`) `int(MAX_SMF_QUARTER_NOTES * tpq)` floor +
      `min(max_target_ticks, ...)` clamp, SMPTE `FF 51 03 00 00 00` ignore parity, and `uint16` division bounds guard
    - [R6-LOW-02] `validate_supported_time_signature` scalar/bool/ndarray/malicious-`__eq__` hardening with zero
      `DeprecationWarning` under `-W error` and zero leaked `TypeError` / `OverflowError`
    """
    print("\n--- [TEST 14] Round-7 Fact-Checker (R6) Backend Remediations ---")
    from decimal import Decimal
    from fractions import Fraction
    import numbers
    import struct
    from types import SimpleNamespace
    import app as app_mod
    from models.mortm.mortm45d import MORTM45DRapper
    from mortm.models.modules.progress import GLOBAL_GENERATION_PROGRESS
    from mortm.train.custom_token import validate_supported_time_signature
    from mortm.train.tokenizer import TO_TOKEN, Tokenizer, omega_converter

    tmp_r6 = tempfile.mkdtemp(prefix="compass_r6_audit_")
    try:
        # 1. [R6-MED-01] Symmetric fail-closed job_id validation on /generation_progress & /generate + unscoped masking
        with app_mod._generate_limiter._lock:
            app_mod._generate_limiter._buckets["loopback"].clear()
        with app_mod._cancel_limiter._lock:
            app_mod._cancel_limiter._buckets["loopback"].clear()

        client = TestClient(app)
        GLOBAL_GENERATION_PROGRESS.start_run("run_secret_r6", "推論進行中...", job_id="job_secret_r6")
        try:
            # Genuinely unscoped /generation_progress returns 200 with active status, but masks job_id & run_id
            r_unscoped = client.get("/generation_progress")
            assert r_unscoped.status_code == 200
            unscoped_json = r_unscoped.json()
            assert unscoped_json["active"] is True and unscoped_json["stage"] == "loading_model"
            assert unscoped_json["job_id"] is None and unscoped_json["run_id"] is None

            # Malformed, comma-folded, whitespace-only, duplicate, or conflicting job_id -> HTTP 400 (never leaks job_secret_r6!)
            for bad_resp in [
                client.get("/generation_progress?job_id=bad!id"),
                client.get("/generation_progress", headers={"X-Compass-Job-Id": "bad!id"}),
                client.get("/generation_progress", headers={"X-Compass-Job-Id": "job_1, job_2"}),
                client.get("/generation_progress?job_id=%20%20"),
                client.get("/generation_progress", headers={"X-Compass-Job-Id": "   "}),
                client.get("/generation_progress?job_id=job_2&job_id=%20%20"),
                client.get("/generation_progress?job_id=job_secret_r6&job_id=job_secret_r6"),
                client.get(
                    "/generation_progress?job_id=job_a",
                    headers={"X-Compass-Job-Id": "job_b"},
                ),
            ]:
                assert bad_resp.status_code == 400, f"Expected 400, got {bad_resp.status_code}: {bad_resp.text}"

            # Raw ASGI duplicate headers (whitespace + job_2) rejected with HTTP 400 on /generation_progress & /cancel_generate
            async def _raw_asgi_status(method: str, path: str, raw_headers: list) -> int:
                statuses: list = []

                async def receive():
                    return {"type": "http.request", "body": b"", "more_body": False}

                async def send(msg):
                    if msg["type"] == "http.response.start":
                        statuses.append(msg["status"])

                scope = {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": method,
                    "scheme": "http",
                    "path": path,
                    "raw_path": path.encode("ascii"),
                    "query_string": b"",
                    "headers": raw_headers,
                    "client": ("127.0.0.1", 54321),
                    "server": ("127.0.0.1", 8000),
                }
                await app(scope, receive, send)
                return statuses[0]

            import asyncio as _aio

            assert (
                _aio.run(
                    _raw_asgi_status(
                        "GET",
                        "/generation_progress",
                        [(b"x-compass-job-id", b"   "), (b"x-compass-job-id", b"job_2")],
                    )
                )
                == 400
            )
            assert (
                _aio.run(
                    _raw_asgi_status(
                        "POST",
                        "/cancel_generate",
                        [(b"x-compass-job-id", b"   "), (b"x-compass-job-id", b"job_2")],
                    )
                )
                == 400
            )

            # Valid ?job_id=%20%20 paired with single valid X-Compass-Job-Id still falls through cleanly (HTTP 200)
            r_valid_scope = client.get(
                "/generation_progress?job_id=%20%20",
                headers={"X-Compass-Job-Id": "job_secret_r6"},
            )
            assert r_valid_scope.status_code == 200 and r_valid_scope.json()["job_id"] == "job_secret_r6"
        finally:
            GLOBAL_GENERATION_PROGRESS.update(
                active=False, stage="done", message="完了", run_id="run_secret_r6"
            )

        # POST /generate rejects malformed, conflicting, or duplicate job_id BEFORE touching model controller
        meta_payload = json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta"})
        with patch.object(
            app_mod.model_controller,
            "generate",
            side_effect=AssertionError("model_controller.generate must NOT run when job_id is invalid!"),
        ):
            assert (
                client.post("/generate", data={"meta": meta_payload, "job_id": "bad!id"}).status_code
                == 400
            )
            assert (
                client.post(
                    "/generate",
                    data={"meta": meta_payload, "job_id": "job_form"},
                    headers={"X-Compass-Job-Id": "job_header"},
                ).status_code
                == 400
            )
            assert (
                client.post(
                    "/generate",
                    data={"meta": meta_payload, "job_id": ["job_a", "job_b"]},
                ).status_code
                == 400
            )
            assert (
                client.post(
                    "/generate",
                    files=[
                        ("meta", (None, meta_payload)),
                        ("job_id", (None, "job_a")),
                        ("job_id", (None, "job_b")),
                    ],
                ).status_code
                == 400
            )
            assert (
                client.post(
                    "/generate",
                    data={"meta": meta_payload},
                    files={"job_id": ("spoof.txt", b"job_file", "text/plain")},
                ).status_code
                == 400
            )
            assert (
                client.post("/generate?job_id=%20%20", data={"meta": meta_payload}).status_code
                == 400
            )
            assert (
                client.post(
                    "/generate",
                    data={"meta": meta_payload},
                    headers={"X-Compass-Job-Id": "   "},
                ).status_code
                == 400
            )
            assert (
                client.post(
                    "/generate",
                    data={"meta": meta_payload, "job_id": "   "},
                ).status_code
                == 400
            )

        # 2. [R6-MED-02] MORTM45DRapper.preprocessing reconciles silent context slices to avoid OOD zero-context CoT
        tok_omega = Tokenizer(omega_converter(TO_TOKEN))
        rapper_45d = MORTM45DRapper.__new__(MORTM45DRapper)
        rapper_45d.meta = {"tag": {"model": "sft_gen"}}
        rapper_45d.model = SimpleNamespace(args=SimpleNamespace(position_length=4096))
        rapper_45d.last_reason = []

        pm_bar0 = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
        inst_p0 = pretty_midi.Instrument(program=0)
        inst_p0.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        pm_bar0.instruments.append(inst_p0)
        bar0_path = os.path.join(tmp_r6, "silent_after_bar0.mid")
        pm_bar0.write(bar0_path)

        # Silent slice (start_measure=2 -> []) on meta_past, meta_future, infill, inst_comp reconciles to 'meta' (use_cot=False)
        for orig_task, p_arg, c_arg, f_arg in [
            ("meta_past", bar0_path, None, None),
            ("meta_future", None, None, bar0_path),
            ("infill", bar0_path, None, bar0_path),
            ("inst_comp", None, bar0_path, None),
        ]:
            meta_silent = GenerateMeta(
                model_type="MORTM4.5D-80M-SFT-Gen",
                task=orig_task,
                thinking=True,
                start_measure=2,
                measure_count=2,
                future_start_measure=2,
                future_measure_count=2,
                num_gems=1,
            )
            prep = rapper_45d.preprocessing(p_arg, c_arg, f_arg, meta_silent)
            seq_names = [prep["tokenizer"].rev_get(int(t)) for t in prep["sequence"][0]]
            assert prep["task"] == "meta", f"Expected 'meta' reconciliation for {orig_task}, got {prep['task']}"
            assert prep["use_cot"] is False
            assert "<thinking>" not in seq_names
            assert seq_names == [
                "<EOS>",
                "<SYSTEM>",
                "<INST_PIANO>",
                "<NOTE_DENSE_4>",
                "<GEN_MEASURE_COUNT_8>",
                "k_Unknown",
                "<TAG_END>",
                "<MGEN>",
            ], f"Unexpected reconciled prompt for {orig_task}: {seq_names}"
            assert rapper_45d.last_reason[0]["task"] == "meta" and rapper_45d.last_reason[0]["thinking"] is False

        # Infill where past is non-empty (Bar 0) and future is silent (Bar 2) reconciles to 'meta_past' (keeps CoT + <PAST_M>)
        meta_half_infill = GenerateMeta(
            model_type="MORTM4.5D-80M-SFT-Gen",
            task="infill",
            thinking=True,
            start_measure=0,
            measure_count=1,
            future_start_measure=2,
            future_measure_count=1,
            num_gems=1,
        )
        prep_half = rapper_45d.preprocessing(bar0_path, None, bar0_path, meta_half_infill)
        half_names = [prep_half["tokenizer"].rev_get(int(t)) for t in prep_half["sequence"][0]]
        assert prep_half["task"] == "meta_past" and prep_half["use_cot"] is True
        assert "<PAST_M>" in half_names and "<FUTURE_M>" not in half_names and "<thinking>" in half_names

        # 3. [R6-LOW-01] Fractional SMPTE (29.97 drop-frame 0xE301 / 0xE328) horizon parity + zero-tempo ignore + uint16 guard
        assert app_mod._extract_smf_ticks_per_quarter(False) is None
        assert app_mod._extract_smf_ticks_per_quarter(True) is None
        assert app_mod._extract_smf_ticks_per_quarter(-1) is None
        assert app_mod._extract_smf_ticks_per_quarter(0x10000) is None

        for smpte_div in (0xE301, 0xE328):
            tpq = app_mod._extract_smf_ticks_per_quarter(smpte_div)
            assert tpq is not None
            max_raw = int(app_mod.MAX_SMF_QUARTER_NOTES * tpq)
            mtrk_edge = (
                b"\x00\xff\x51\x03\x00\x00\x00"  # zero tempo ignored on linear SMPTE
                + app_mod._encode_smf_vlq(max_raw)
                + b"\x90\x3c\x64\x00\x80\x3c\x00\x00\xff\x2f\x00"
            )
            smf_edge = (
                struct.pack(">4sIHHH", b"MThd", 6, 0, 1, smpte_div)
                + struct.pack(">4sI", b"MTrk", len(mtrk_edge))
                + mtrk_edge
            )
            assert app_mod._validate_smf_binary_structure(smf_edge) is True
            norm_edge = app_mod._normalize_high_ppqn_smf_bytes(smf_edge, target_ppqn=480)
            assert app_mod._validate_smf_binary_structure(norm_edge) is True

            mtrk_over = app_mod._encode_smf_vlq(max_raw + 1) + b"\xff\x2f\x00"
            smf_over = (
                struct.pack(">4sIHHH", b"MThd", 6, 0, 1, smpte_div)
                + struct.pack(">4sI", b"MTrk", len(mtrk_over))
                + mtrk_over
            )
            assert app_mod._validate_smf_binary_structure(smf_over) is False

        # Metrical (non-SMPTE) zero-tempo event MUST still be rejected
        mtrk_zero_metrical = b"\x00\xff\x51\x03\x00\x00\x00\x00\xff\x2f\x00"
        smf_zero_metrical = (
            struct.pack(">4sIHHH", b"MThd", 6, 0, 1, 480)
            + struct.pack(">4sI", b"MTrk", len(mtrk_zero_metrical))
            + mtrk_zero_metrical
        )
        assert app_mod._validate_smf_binary_structure(smf_zero_metrical) is False

        # 4. [R6-LOW-02] validate_supported_time_signature scalar/ndarray/malicious-__eq__ hardening
        assert validate_supported_time_signature(np.int64(3), np.int64(4)) == (3, 4)
        assert validate_supported_time_signature(Decimal("6"), Decimal("8")) == (6, 8)
        assert validate_supported_time_signature(Fraction(8, 2), 4) == (4, 4)

        class _WeirdSymbolicFour:
            def __int__(self):
                return 4

            def __eq__(self, other):
                return other == 4

        class _BadBoolResult:
            def __bool__(self):
                return 1  # invalid non-bool return from __bool__

        class _MaliciousReal(numbers.Real):
            def __int__(self):
                return 4

            def __float__(self):
                return 4.0

            def __trunc__(self):
                return 4

            def __floor__(self):
                return 4

            def __ceil__(self):
                return 4

            def __round__(self, ndigits=None):
                return 4

            def __abs__(self):
                return 4

            def __pos__(self):
                return 4

            def __neg__(self):
                return -4

            def __add__(self, other):
                return 4

            def __radd__(self, other):
                return 4

            def __mul__(self, other):
                return 4

            def __rmul__(self, other):
                return 4

            def __truediv__(self, other):
                return 1.0

            def __rtruediv__(self, other):
                return 1.0

            def __floordiv__(self, other):
                return 1

            def __rfloordiv__(self, other):
                return 1

            def __mod__(self, other):
                return 0

            def __rmod__(self, other):
                return 0

            def __pow__(self, exponent):
                return 1

            def __rpow__(self, base):
                return 1

            def __lt__(self, other):
                return False

            def __le__(self, other):
                return True

            def __eq__(self, other):
                return _BadBoolResult()

        for bad_n, bad_d in [
            (np.array([4]), 4),
            (4, np.array([4])),
            (np.array([4, 4]), 4),
            (np.array(4), 4),
            (np.bool_(True), 4),
            (_WeirdSymbolicFour(), 4),
            (4, _MaliciousReal()),
            (_MaliciousReal(), 4),
            (Decimal("Infinity"), 4),
            (Decimal("NaN"), 4),
            (Decimal("4.5"), 4),
        ]:
            try:
                validate_supported_time_signature(bad_n, bad_d)
                raise AssertionError(f"Expected ValueError for ({bad_n!r}, {bad_d!r})")
            except ValueError as ve:
                assert "拍子" in str(ve)
    finally:
        shutil.rmtree(tmp_r6, ignore_errors=True)

    print("  [PASS] All Round-7 Fact-Checker (R6) backend remediations verified")


def test_15_round8_fact_checker_r7_remediations():
    """
    Round-8 physical regression tests for ALL 3 backend Round-7 Fact-Checker findings:
    - [R7-LOW-01] `validate_supported_time_signature` accepts `np.float64(4.0)` / `np.float32(4.0)` / `np.float16(4.0)`
      while catching malicious `.ndim` descriptors and huge finite `Decimal` / `Fraction` inputs in O(1) without leaking
      `RuntimeError` / `OverflowError` / `DeprecationWarning`
    - [R7-LOW-02] `MORTM45DRapper.preprocessing` preserves complementary `target_insts` (`['SAX']` for PIANO `const_midi`,
      `['PIANO']` for SAX `const_midi`) even when Stage-2 budget trimming reconciles `inst_comp -> meta` at tiny `position_length`
    - [R7-LOW-03] `_normalize_job_id` validates full-string charset (`<= 4096`, `^[A-Za-z0-9_.:-]+$`) BEFORE `[:128]`
      truncation, rejecting `("a" * 128) + ", job_2"` and `("a" * 128) + "!evil"` with HTTP 400 while preserving `[R3-LOW-03]`
    """
    print("\n--- [TEST 15] Round-8 Fact-Checker (R7) Backend Remediations ---")
    from decimal import Decimal
    from fractions import Fraction
    from types import SimpleNamespace
    import app as app_mod
    from models.mortm.mortm45d import MORTM45DRapper
    from mortm.train.custom_token import validate_supported_time_signature

    tmp_r7 = tempfile.mkdtemp(prefix="compass_r7_audit_")
    try:
        # 1. [R7-LOW-01] np.float64(4.0) accepted; malicious .ndim & huge Decimal("1E+9999999") rejected with ValueError
        assert validate_supported_time_signature(np.float64(4.0), 4) == (4, 4)
        assert validate_supported_time_signature(4, np.float64(4.0)) == (4, 4)
        assert validate_supported_time_signature(np.float32(5.0), np.float64(32.0)) == (5, 32)
        assert validate_supported_time_signature(np.float16(7.0), np.int64(32)) == (7, 32)

        class _BadNdimObj:
            @property
            def ndim(self):
                raise RuntimeError("boom in ndim")

        for bad_n, bad_d in [
            (np.float64(4.5), 4),
            (_BadNdimObj(), 4),
            (4, _BadNdimObj()),
            (Decimal("1E+9999999"), 4),
            (Decimal("-1E+9999999"), 4),
            (Fraction(10**500, 1), 4),
        ]:
            t0 = time.perf_counter()
            try:
                validate_supported_time_signature(bad_n, bad_d)
                raise AssertionError(f"Expected ValueError for ({bad_n!r}, {bad_d!r})")
            except ValueError as ve:
                assert "拍子" in str(ve)
            assert (time.perf_counter() - t0) < 0.05

        # 2. [R7-LOW-02] Stage-2 inst_comp -> meta budget trim preserves complementary target_insts
        for cond_prog_num, expected_comp in [(0, ["SAX"]), (65, ["PIANO"])]:
            pm_cond = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
            inst_c = pretty_midi.Instrument(program=cond_prog_num)
            inst_c.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=20.0, end=20.5))
            pm_cond.instruments.append(inst_c)
            cond_path = os.path.join(tmp_r7, f"cond_bar10_{cond_prog_num}.mid")
            pm_cond.write(cond_path)

            for pos_len, expected_task, expected_cot in [(4096, "inst_comp", True), (32, "meta", False)]:
                rapper = MORTM45DRapper.__new__(MORTM45DRapper)
                rapper.meta = {"tag": {"model": "sft_gen"}}
                rapper.model = SimpleNamespace(args=SimpleNamespace(position_length=pos_len))
                rapper.last_reason = []
                meta_ic = GenerateMeta(
                    model_type="MORTM4.5D-80M-SFT-Gen",
                    task="inst_comp",
                    program=[],
                    thinking=True,
                    start_measure=0,
                    measure_count=12,
                    num_gems=1,
                )
                prep_ic = rapper.preprocessing(None, cond_path, None, meta_ic)
                assert prep_ic["task"] == expected_task
                assert prep_ic["use_cot"] is expected_cot
                assert prep_ic["target_insts"] == expected_comp, (
                    f"Expected complementary {expected_comp} at pos_len={pos_len}, got {prep_ic['target_insts']}"
                )
                assert rapper.last_reason[0]["instruments"] == expected_comp

        # 3. [R7-LOW-03] >128-char comma-folded / malformed suffix rejected with HTTP 400 while valid >128-char truncates cleanly
        assert app_mod._normalize_job_id("  " + ("j" * 180) + "  ") == "j" * 128
        assert app_mod._normalize_job_id(("a" * 128) + ", job_2") is None
        assert app_mod._normalize_job_id(("a" * 128) + "!evil") is None
        assert app_mod._normalize_job_id("a" * 5000) is None

        with app_mod._generate_limiter._lock:
            app_mod._generate_limiter._buckets["loopback"].clear()
        with app_mod._cancel_limiter._lock:
            app_mod._cancel_limiter._buckets["loopback"].clear()

        client = TestClient(app)
        meta_payload = json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta"})
        for bad_long_id in [("a" * 128) + ", job_2", ("a" * 128) + "!evil", "a" * 4100]:
            assert (
                client.get("/generation_progress", headers={"X-Compass-Job-Id": bad_long_id}).status_code
                == 400
            )
            assert (
                client.post("/cancel_generate", headers={"X-Compass-Job-Id": bad_long_id}).status_code
                == 400
            )
            assert (
                client.post("/generate", data={"meta": meta_payload, "job_id": bad_long_id}).status_code
                == 400
            )
    finally:
        shutil.rmtree(tmp_r7, ignore_errors=True)

    print("  [PASS] All Round-8 Fact-Checker (R7) backend remediations verified")


def test_16_round9_fact_checker_r8_remediations():
    print("\n--- [TEST 16] Round-9 Fact-Checker (R8) Remediations (MORTM45Rapper Key Normalization & Length Guard) ---")
    from types import SimpleNamespace
    from models.mortm.mortm45 import MORTM45Rapper
    from mortm.models.modules.backend import HardwareBackendFactory

    cpu_backend = HardwareBackendFactory.create("cpu")
    model_dir = os.path.join(ROOT_DIR, "data", "models", "MORTM45D_80M_sft_gen")

    for model_tag in ("pretrained", "generation"):
        rapper = MORTM45Rapper.__new__(MORTM45Rapper)
        rapper.backend = cpu_backend
        rapper.meta = {
            "model_folder_path": model_dir,
            "model_name": f"MORTM4.5-{model_tag}",
            "tag": {
                "instruments": ["PIANO", "SAX"],
                "model": model_tag,
                "task": ["Meta2MIDI"],
            },
        }
        rapper.model = SimpleNamespace(args=SimpleNamespace(position_length=512))
        rapper.last_reason = []

        # 1. Overlong / unknown custom key (>32 chars) normalizes to "Unknown" (<= 32 chars)
        meta_long_key = GenerateMeta(
            model_type=f"MORTM4.5-{model_tag}",
            task="Meta2MIDI",
            program=["PIANO"],
            key="C_major_" + ("x" * 40),
            num_gems=2,
        )
        prep_long = rapper.preprocessing(None, None, None, meta_long_key)
        assert len(rapper.last_reason) == 2
        for r in rapper.last_reason:
            assert r["key"] == "Unknown"
            assert len(r["key"]) <= 32
        tok = prep_long["tokenizer"]
        seq0 = (
            prep_long["sequence"][0].detach().cpu().numpy().tolist()
            if hasattr(prep_long["sequence"][0], "detach")
            else np.asarray(prep_long["sequence"][0]).flatten().tolist()
        )
        assert tok.get("k_Unknown") in seq0

        # 2. Human-readable "C major" / "A minor" normalizes to "CM" / "Am"
        for raw_k, expected_k in (("C major", "CM"), ("A minor", "Am"), ("EbM", "EbM")):
            meta_k = GenerateMeta(
                model_type=f"MORTM4.5-{model_tag}",
                task="Meta2MIDI",
                program=["PIANO"],
                key=raw_k,
                num_gems=1,
            )
            prep_k = rapper.preprocessing(None, None, None, meta_k)
            assert rapper.last_reason[0]["key"] == expected_k
            seq_k = (
                prep_k["sequence"][0].detach().cpu().numpy().tolist()
                if hasattr(prep_k["sequence"][0], "detach")
                else np.asarray(prep_k["sequence"][0]).flatten().tolist()
            )
            assert prep_k["tokenizer"].get(f"k_{expected_k}") in seq_k

        # 3. "Auto" / whitespace / None normalizes to None
        for blank_k in (None, "", "   ", "Auto", "auto"):
            meta_blank = GenerateMeta(
                model_type=f"MORTM4.5-{model_tag}",
                task="Meta2MIDI",
                program=["PIANO"],
                key=blank_k,
                num_gems=1,
            )
            rapper.preprocessing(None, None, None, meta_blank)
            assert rapper.last_reason[0]["key"] is None

    print("  [PASS] All Round-9 Fact-Checker (R8) backend remediations verified")


def test_17_round19_generate_meta_recursion_and_size_guards():
    print("\n--- [TEST 17] Round-19 POST /generate RecursionError & 64KB Meta Size Guards ([R18-MED-03]) ---")
    import app as app_mod
    from fastapi.testclient import TestClient

    app_mod._generate_limiter._buckets.clear()
    secret = os.environ.get("COMPASS_HANDSHAKE_TOKEN", "").strip()
    hdrs = {"X-Compass-Handshake": secret} if secret else {}
    client = TestClient(app_mod.app)

    try:
        # 1. Deeply nested JSON arrays & objects must return HTTP 400 (never uncaught RecursionError -> 500)
        deep_arr = "[" * 2000 + "]" * 2000
        r_arr = client.post("/generate", data={"meta": deep_arr}, headers=hdrs)
        assert r_arr.status_code == 400, f"Expected HTTP 400 for deep array JSON, got {r_arr.status_code}"

        deep_obj = '{"a":' * 2000 + "1" + "}" * 2000
        r_obj = client.post("/generate", data={"meta": deep_obj}, headers=hdrs)
        assert r_obj.status_code == 400, f"Expected HTTP 400 for deep object JSON, got {r_obj.status_code}"

        # 2. Oversized > 64KB raw meta string must fail fast with HTTP 413 before json.loads
        huge_meta = '{"model_type":"MORTM4.5D-80M-SFT-Gen","pad":"' + ("A" * 70000) + '"}'
        r_huge = client.post("/generate", data={"meta": huge_meta}, headers=hdrs)
        assert r_huge.status_code == 413, f"Expected HTTP 413 for >64KB meta, got {r_huge.status_code}"
    finally:
        app_mod._generate_limiter._buckets.clear()

    print("  [PASS] POST /generate RecursionError (HTTP 400) & >64KB ceiling (HTTP 413) verified")


def test_18_round21_multipart_shadowing_and_smuggling_guards():
    print(
        "\n--- [TEST 18] Round-21 POST /generate Multipart Duplicate, Conflict & Unknown-Field Guards ([R20-LOW-01]) ---"
    )
    import asyncio
    import struct
    import app as app_mod
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    app_mod._generate_limiter._buckets.clear()
    secret = os.environ.get("COMPASS_HANDSHAKE_TOKEN", "").strip()
    hdrs = {"X-Compass-Handshake": secret} if secret else {}
    client = TestClient(app_mod.app)
    valid_meta = json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta"})
    valid_smf = (
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", 12)
        + b"\x00\x90\x3c\x64\x60\x80\x3c\x00\x00\xff\x2f\x00"
    )

    try:
        # 1. Duplicate `meta` fields (oversized first shadowed by valid second) -> HTTP 400
        r_dup_meta = client.post(
            "/generate",
            files=[("meta", (None, "X" * 100_000)), ("meta", (None, valid_meta))],
            headers=hdrs,
        )
        assert r_dup_meta.status_code == 400, f"Expected 400 on duplicate meta, got {r_dup_meta.status_code}"

        # 2. Simultaneous `meta` and `meta_json` -> HTTP 400
        r_both_meta = client.post(
            "/generate",
            data={"meta": valid_meta, "meta_json": "NOT_JSON{{{"},
            headers=hdrs,
        )
        assert r_both_meta.status_code == 400, f"Expected 400 on meta+meta_json, got {r_both_meta.status_code}"

        # 3. Duplicate `past_midi` (malicious first shadowed by valid second) -> HTTP 400
        r_dup_file = client.post(
            "/generate",
            files=[
                ("meta", (None, valid_meta)),
                ("past_midi", ("evil.exe", b"MZ_EXEC")),
                ("past_midi", ("ok.mid", valid_smf)),
            ],
            headers=hdrs,
        )
        assert r_dup_file.status_code == 400, f"Expected 400 on duplicate past_midi, got {r_dup_file.status_code}"

        # 4. Unknown multipart field (`smuggled`) -> HTTP 400
        r_smuggled = client.post(
            "/generate",
            files=[("meta", (None, valid_meta)), ("smuggled", ("evil.exe", b"MZ_EXEC"))],
            headers=hdrs,
        )
        assert r_smuggled.status_code == 400, f"Expected 400 on unknown field, got {r_smuggled.status_code}"

        # 5. Empty / whitespace filename non-empty payload smuggling -> HTTP 400 / 422
        r_anon_file = client.post(
            "/generate",
            files=[
                ("meta", (None, valid_meta)),
                ("past_midi", ("", b"MZ_EXEC", "application/octet-stream")),
            ],
            headers=hdrs,
        )
        assert r_anon_file.status_code in (400, 422), (
            f"Expected 400/422 on empty-filename payload, got {r_anon_file.status_code}"
        )
        r_ws_file = client.post(
            "/generate",
            files=[
                ("meta", (None, valid_meta)),
                ("past_midi", ("   ", b"MZ_EXEC", "application/octet-stream")),
            ],
            headers=hdrs,
        )
        assert r_ws_file.status_code == 400, (
            f"Expected 400 on whitespace-filename payload, got {r_ws_file.status_code}"
        )

        # 6. Direct coroutine invocation (`_DummyReq`) with simultaneous `meta` + `meta_json` -> HTTPException(400)
        class _DummyDirectReq:
            headers = hdrs
            query_params = {}
            client = type("C", (), {"host": "127.0.0.1"})()

            async def is_disconnected(self):
                return False

        async def _check_direct_conflict():
            try:
                await app_mod.generate_music(
                    request=_DummyDirectReq(),
                    meta=valid_meta,
                    meta_json=valid_meta,
                    job_id="job_direct_conflict",
                    past_midi=None,
                    conditions_midi=None,
                    future_midi=None,
                )
                raise AssertionError("Expected HTTPException(400) on direct meta + meta_json")
            except HTTPException as exc:
                assert exc.status_code == 400

        asyncio.run(_check_direct_conflict())
    finally:
        app_mod._generate_limiter._buckets.clear()

    print("  [PASS] POST /generate multipart duplicate, conflict, & smuggling guards verified")


def test_19_round22_pre_auth_zero_spool_and_disconnect_cleanup():
    print(
        "\n--- [TEST 19] Round-22 Pre-Auth Zero-Spool, 34MB Streaming Cap & ClientDisconnect Handle Cleanup ([R21-HIGH-01]) ---"
    )
    import asyncio
    import gc
    from tempfile import SpooledTemporaryFile as RealSpooledTemporaryFile
    from unittest.mock import patch
    import starlette.formparsers as sf_mod
    import app as app_mod

    app_mod._generate_limiter._buckets.clear()
    prev_token = os.environ.get("COMPASS_HANDSHAKE_TOKEN")
    test_secret = "round22_test_secret_token_9911"
    os.environ["COMPASS_HANDSHAKE_TOKEN"] = test_secret
    valid_meta = json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta"})

    created_spooled = []

    def _tracked_spooled(*args, **kwargs):
        f = RealSpooledTemporaryFile(*args, **kwargs)
        created_spooled.append(f)
        return f

    async def _invoke_asgi(headers_list, chunks, disconnect_at=None, hang_after_chunks=False):
        receive_calls = 0
        bytes_delivered = 0
        idx = 0
        resp_status = None
        resp_body = bytearray()

        async def _receive():
            nonlocal receive_calls, bytes_delivered, idx
            receive_calls += 1
            if disconnect_at is not None and idx == disconnect_at:
                return {"type": "http.disconnect"}
            if idx < len(chunks):
                ch = chunks[idx]
                idx += 1
                bytes_delivered += len(ch)
                more = idx < len(chunks) or disconnect_at is not None or hang_after_chunks
                return {"type": "http.request", "body": ch, "more_body": more}
            if hang_after_chunks:
                await asyncio.sleep(60.0)
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(msg):
            nonlocal resp_status
            if msg["type"] == "http.response.start":
                resp_status = msg["status"]
            elif msg["type"] == "http.response.body":
                resp_body.extend(msg.get("body", b""))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/generate",
            "raw_path": b"/generate",
            "query_string": b"",
            "headers": headers_list,
            "client": ("127.0.0.1", 54321),
            "server": ("127.0.0.1", 8000),
        }
        await app_mod.app(scope, _receive, _send)
        return resp_status, receive_calls, bytes_delivered, bytes(resp_body)

    boundary = b"----CompassR22Boundary"
    part_prefix = (
        b"------CompassR22Boundary\r\n"
        b'Content-Disposition: form-data; name="meta"\r\n\r\n'
        + valid_meta.encode("utf-8")
        + b"\r\n------CompassR22Boundary\r\n"
        b'Content-Disposition: form-data; name="past_midi"; filename="big.mid"\r\n'
        b"Content-Type: audio/midi\r\n\r\n"
    )
    part_suffix = b"\r\n------CompassR22Boundary--\r\n"
    mb_chunk = b"A" * (1024 * 1024)

    try:
        with patch.object(sf_mod, "SpooledTemporaryFile", side_effect=_tracked_spooled):
            # 1. Unauthenticated CORS Simple Request (`Origin: https://evil.example`, no token) -> 403 with 0 bytes read & 0 files
            created_spooled.clear()
            status_unauth, recv_unauth, bytes_unauth, _ = asyncio.run(
                _invoke_asgi(
                    [
                        (b"origin", b"https://evil.example"),
                        (b"content-type", b"multipart/form-data; boundary=" + boundary),
                    ],
                    [part_prefix, mb_chunk, mb_chunk, mb_chunk, part_suffix],
                )
            )
            assert status_unauth == 403, f"Expected 403, got {status_unauth}"
            assert recv_unauth == 0 and bytes_unauth == 0, (
                f"Pre-auth read {bytes_unauth} bytes across {recv_unauth} receive() calls!"
            )
            assert len(created_spooled) == 0, "Pre-auth created SpooledTemporaryFile!"

            # 2. Rate-limited client -> 429 with 0 bytes read & 0 files
            now_mono = time.monotonic()
            with app_mod._generate_limiter._lock:
                app_mod._generate_limiter._buckets["loopback"].clear()
                for _ in range(app_mod._generate_limiter.max_requests):
                    app_mod._generate_limiter._buckets["loopback"].append(now_mono)
            status_rl, recv_rl, bytes_rl, _ = asyncio.run(
                _invoke_asgi(
                    [
                        (b"x-compass-handshake", test_secret.encode("utf-8")),
                        (b"content-type", b"multipart/form-data; boundary=" + boundary),
                    ],
                    [part_prefix, mb_chunk, mb_chunk, part_suffix],
                )
            )
            assert status_rl == 429, f"Expected 429, got {status_rl}"
            assert recv_rl == 0 and bytes_rl == 0
            assert len(created_spooled) == 0
            app_mod._generate_limiter._buckets.clear()

            # 3. Authenticated streaming multipart exceeding 34 MB -> aborts mid-stream with 413 & closes handles without GC
            created_spooled.clear()
            chunks_40mb = [part_prefix] + ([mb_chunk] * 40) + [part_suffix]
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                status_413, _recv_413, bytes_413, _ = asyncio.run(
                    _invoke_asgi(
                        [
                            (b"x-compass-handshake", test_secret.encode("utf-8")),
                            (b"content-type", b"multipart/form-data; boundary=" + boundary),
                        ],
                        chunks_40mb,
                    )
                )
                assert status_413 == 413, f"Expected 413 on >34MB stream, got {status_413}"
                assert bytes_413 <= (35 * 1024 * 1024 + len(part_prefix)), (
                    f"Stream was not aborted early! Read {bytes_413} bytes"
                )
                assert len(created_spooled) == 1
                assert created_spooled[0].closed is True, (
                    "SpooledTemporaryFile leaked open after >34MB 413 abort without GC!"
                )
            finally:
                if gc_was_enabled:
                    gc.enable()
            app_mod._generate_limiter._buckets.clear()

            # 4. Authenticated mid-upload ClientDisconnect after rolling >2MB to %TEMP% -> 400 & immediate closed == True without GC
            created_spooled.clear()
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                status_disc, _recv_disc, _bytes_disc, _ = asyncio.run(
                    _invoke_asgi(
                        [
                            (b"x-compass-handshake", test_secret.encode("utf-8")),
                            (b"content-type", b"multipart/form-data; boundary=" + boundary),
                        ],
                        [part_prefix, mb_chunk, mb_chunk, mb_chunk],
                        disconnect_at=4,
                    )
                )
                assert status_disc == 400, f"Expected 400 on ClientDisconnect, got {status_disc}"
                assert len(created_spooled) == 1
                assert getattr(created_spooled[0], "_rolled", False) is True
                assert created_spooled[0].closed is True, (
                    "SpooledTemporaryFile leaked open on mid-upload ClientDisconnect without GC!"
                )
            finally:
                if gc_was_enabled:
                    gc.enable()
            app_mod._generate_limiter._buckets.clear()

            # 5. Authenticated mid-upload asyncio.CancelledError after rolling >2MB to %TEMP% -> immediate closed == True without GC
            created_spooled.clear()

            async def _check_cancel_mid_upload():
                t = asyncio.create_task(
                    _invoke_asgi(
                        [
                            (b"x-compass-handshake", test_secret.encode("utf-8")),
                            (b"content-type", b"multipart/form-data; boundary=" + boundary),
                        ],
                        [part_prefix, mb_chunk, mb_chunk, mb_chunk],
                        hang_after_chunks=True,
                    )
                )
                while not (len(created_spooled) == 1 and getattr(created_spooled[0], "_rolled", False)):
                    await asyncio.sleep(0.01)
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                asyncio.run(_check_cancel_mid_upload())
                assert len(created_spooled) == 1
                assert created_spooled[0].closed is True, (
                    "SpooledTemporaryFile leaked open on mid-upload CancelledError without GC!"
                )
            finally:
                if gc_was_enabled:
                    gc.enable()
    finally:
        if prev_token is None:
            os.environ.pop("COMPASS_HANDSHAKE_TOKEN", None)
        else:
            os.environ["COMPASS_HANDSHAKE_TOKEN"] = prev_token
        app_mod._generate_limiter._buckets.clear()

    print("  [PASS] Pre-auth 0-byte rejection, 34MB streaming cap, & immediate disconnect/cancel handle closure verified")


def test_20_round23_truncated_multipart_semicolon_and_header_guards():
    print(
        "\n--- [TEST 20] Round-23 Truncated Multipart Clean-EOF, Semicolon URL-Encoded, & Non-ASCII Header Guards ([R22-HIGH-01], [R22-HIGH-02], [R22-MED-01]) ---"
    )
    import asyncio
    import gc
    import struct
    import tempfile
    from tempfile import SpooledTemporaryFile as RealSpooledTemporaryFile
    from unittest.mock import patch
    import starlette.formparsers as sf_mod
    import app as app_mod

    app_mod._generate_limiter._buckets.clear()
    prev_token = os.environ.get("COMPASS_HANDSHAKE_TOKEN")
    test_secret = "round23_test_secret_token_8842"
    os.environ["COMPASS_HANDSHAKE_TOKEN"] = test_secret
    valid_meta = json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta_past"})
    valid_smf = (
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", 12)
        + b"\x00\x90\x3c\x64\x60\x80\x3c\x00\x00\xff\x2f\x00"
    )

    created_spooled = []

    def _tracked_spooled(*args, **kwargs):
        f = RealSpooledTemporaryFile(*args, **kwargs)
        created_spooled.append(f)
        return f

    async def _invoke_asgi(headers_list, chunks):
        receive_calls = 0
        bytes_delivered = 0
        idx = 0
        resp_status = None
        resp_body = bytearray()

        async def _receive():
            nonlocal receive_calls, bytes_delivered, idx
            receive_calls += 1
            if idx < len(chunks):
                ch = chunks[idx]
                idx += 1
                bytes_delivered += len(ch)
                return {"type": "http.request", "body": ch, "more_body": idx < len(chunks)}
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(msg):
            nonlocal resp_status
            if msg["type"] == "http.response.start":
                resp_status = msg["status"]
            elif msg["type"] == "http.response.body":
                resp_body.extend(msg.get("body", b""))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/generate",
            "raw_path": b"/generate",
            "query_string": b"",
            "headers": headers_list,
            "client": ("127.0.0.1", 54322),
            "server": ("127.0.0.1", 8000),
        }
        await app_mod.app(scope, _receive, _send)
        return resp_status, receive_calls, bytes_delivered, bytes(resp_body)

    boundary = b"----CompassR23Boundary"
    part_prefix = (
        b"------CompassR23Boundary\r\n"
        b'Content-Disposition: form-data; name="meta"\r\n\r\n'
        + valid_meta.encode("utf-8")
        + b"\r\n------CompassR23Boundary\r\n"
        b'Content-Disposition: form-data; name="past_midi"; filename="ok.mid"\r\n'
        b"Content-Type: audio/midi\r\n\r\n"
    )
    part_suffix = b"\r\n------CompassR23Boundary--\r\n"
    mb_chunk = b"A" * (1024 * 1024)

    try:
        with patch.object(sf_mod, "SpooledTemporaryFile", side_effect=_tracked_spooled):
            # 1A. Truncated clean-EOF multipart (missing --boundary--) -> 400, 0 generate calls, & closed == True without GC
            created_spooled.clear()
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                with patch.object(
                    app_mod.model_controller,
                    "generate",
                    side_effect=AssertionError("generate must NEVER run on truncated multipart!"),
                ):
                    status_trunc, _, _, _ = asyncio.run(
                        _invoke_asgi(
                            [
                                (b"x-compass-handshake", test_secret.encode("utf-8")),
                                (b"content-type", b"multipart/form-data; boundary=" + boundary),
                            ],
                            [part_prefix, mb_chunk, mb_chunk],
                        )
                    )
                assert status_trunc == 400, f"Expected 400 on truncated multipart, got {status_trunc}"
                assert len(created_spooled) == 1
                assert getattr(created_spooled[0], "_rolled", False) is True
                assert created_spooled[0].closed is True, (
                    "Truncated clean-EOF SpooledTemporaryFile leaked open without GC!"
                )
            finally:
                if gc_was_enabled:
                    gc.enable()
            app_mod._generate_limiter._buckets.clear()

            # 1B. Valid multipart whose final part is a MIDI file -> 200 OK & closed == True
            created_spooled.clear()

            def _stub_generate(**_kwargs):
                out_dir = tempfile.mkdtemp(prefix="compass_r23_out_")
                out_path = os.path.join(out_dir, "output_0.mid")
                with open(out_path, "wb") as fh:
                    fh.write(valid_smf)
                return out_path, "audio/midi", {"source": "request"}

            with patch.object(app_mod.model_controller, "generate", side_effect=_stub_generate):
                status_ok, _, _, _ = asyncio.run(
                    _invoke_asgi(
                        [
                            (b"x-compass-handshake", test_secret.encode("utf-8")),
                            (b"content-type", b"multipart/form-data; boundary=" + boundary),
                        ],
                        [part_prefix, valid_smf, part_suffix],
                    )
                )
            assert status_ok == 200, f"Expected 200 on valid file-last multipart, got {status_ok}"
            assert len(created_spooled) == 1 and created_spooled[0].closed is True
            app_mod._generate_limiter._buckets.clear()

            # 2. Semicolon separator flood (`b"a;" * 200_000`) -> 400 in < 100 ms (no O(N^2) QuerystringParser freeze)
            t0_semi = time.perf_counter()
            status_semi, _, _, _ = asyncio.run(
                _invoke_asgi(
                    [
                        (b"x-compass-handshake", test_secret.encode("utf-8")),
                        (b"content-type", b"application/x-www-form-urlencoded"),
                    ],
                    [b"a=1;" * 200_000],
                )
            )
            dt_semi_ms = (time.perf_counter() - t0_semi) * 1000.0
            assert status_semi == 400, f"Expected 400 on >64 ';' separators, got {status_semi}"
            assert dt_semi_ms < 100.0, f"Semicolon check took {dt_semi_ms:.1f} ms!"
            app_mod._generate_limiter._buckets.clear()

            # 3. Non-ASCII Content-Type (> U+00FF & U+0080..U+00FF 400B boundary) and blank Content-Length: "   " -> 400 with 0 bytes read
            for bad_headers in (
                [
                    (b"x-compass-handshake", test_secret.encode("utf-8")),
                    (b"content-type", "text/あ".encode("utf-8")),
                ],
                [
                    (b"x-compass-handshake", test_secret.encode("utf-8")),
                    (b"content-type", b"multipart/form-data; boundary=" + (b"\xc3\xa9" * 200)),
                ],
                [
                    (b"x-compass-handshake", test_secret.encode("utf-8")),
                    (b"content-type", b"application/x-www-form-urlencoded"),
                    (b"content-length", b"   "),
                ],
            ):
                st_bad, recv_bad, bytes_bad, _ = asyncio.run(
                    _invoke_asgi(bad_headers, [mb_chunk])
                )
                assert st_bad == 400, f"Expected 400 for malformed header {bad_headers}, got {st_bad}"
                assert recv_bad == 0 and bytes_bad == 0
                app_mod._generate_limiter._buckets.clear()
    finally:
        if prev_token is None:
            os.environ.pop("COMPASS_HANDSHAKE_TOKEN", None)
        else:
            os.environ["COMPASS_HANDSHAKE_TOKEN"] = prev_token
        app_mod._generate_limiter._buckets.clear()

    print("  [PASS] Truncated clean-EOF multipart, semicolon separator cap, & non-ASCII/blank header guards verified")


def test_21_round24_charset_punycode_cancel_disconnect_and_epilogue_guards():
    print(
        "\n--- [TEST 21] Round-24 Charset Allowlist (Punycode DoS), /cancel_generate Disconnect & CL Guards, & Multipart Epilogue Smuggling ([R23-HIGH-01], [R23-HIGH-02], [R23-LOW-01]) ---"
    )
    import asyncio
    import gc
    import struct
    import tempfile
    from tempfile import SpooledTemporaryFile as RealSpooledTemporaryFile
    from unittest.mock import patch
    import starlette.formparsers as sf_mod
    import app as app_mod

    app_mod._generate_limiter._buckets.clear()
    app_mod._cancel_limiter._buckets.clear()
    prev_token = os.environ.get("COMPASS_HANDSHAKE_TOKEN")
    test_secret = "round24_test_secret_token_9951"
    os.environ["COMPASS_HANDSHAKE_TOKEN"] = test_secret
    valid_meta = json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta_past"})
    valid_smf = (
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", 12)
        + b"\x00\x90\x3c\x64\x60\x80\x3c\x00\x00\xff\x2f\x00"
    )

    created_spooled = []

    def _tracked_spooled(*args, **kwargs):
        f = RealSpooledTemporaryFile(*args, **kwargs)
        created_spooled.append(f)
        return f

    async def _invoke_route(path_str, headers_list, chunks, disconnect_after_first=False):
        receive_calls = 0
        bytes_delivered = 0
        idx = 0
        resp_status = None
        resp_body = bytearray()

        async def _receive():
            nonlocal receive_calls, bytes_delivered, idx
            receive_calls += 1
            if idx < len(chunks):
                ch = chunks[idx]
                idx += 1
                bytes_delivered += len(ch)
                return {
                    "type": "http.request",
                    "body": ch,
                    "more_body": True if disconnect_after_first else (idx < len(chunks)),
                }
            if disconnect_after_first:
                return {"type": "http.disconnect"}
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(msg):
            nonlocal resp_status
            if msg["type"] == "http.response.start":
                resp_status = msg["status"]
            elif msg["type"] == "http.response.body":
                resp_body.extend(msg.get("body", b""))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path_str,
            "raw_path": path_str.encode("ascii"),
            "query_string": b"",
            "headers": headers_list,
            "client": ("127.0.0.1", 54324),
            "server": ("127.0.0.1", 8000),
        }
        await app_mod.app(scope, _receive, _send)
        return resp_status, receive_calls, bytes_delivered, bytes(resp_body)

    try:
        # 1. [R23-HIGH-01] Reject charset=punycode, unicode_escape, utf-7 at 0 bytes read (< 20 ms)
        puny_payload = (
            b"--abc\r\n"
            b'Content-Disposition: form-data; name="meta-"\r\n\r\n'
            + (b"a" * 300_000)
            + b"\r\n--abc--\r\n"
        )
        for bad_ct in (
            b"multipart/form-data; boundary=abc; charset=punycode",
            b"multipart/form-data; boundary=abc; charset=unicode_escape",
            b"multipart/form-data; boundary=abc; charset=utf-7",
            b"application/x-www-form-urlencoded; charset=punycode",
        ):
            t0 = time.perf_counter()
            st_cs, recv_cs, bytes_cs, _ = asyncio.run(
                _invoke_route(
                    "/generate",
                    [
                        (b"x-compass-handshake", test_secret.encode("utf-8")),
                        (b"content-type", bad_ct),
                    ],
                    [puny_payload],
                )
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert st_cs == 400, f"Expected 400 on {bad_ct!r}, got {st_cs}"
            assert recv_cs == 0 and bytes_cs == 0
            assert dt_ms < 25.0, f"Charset gate took {dt_ms:.2f} ms"
            app_mod._generate_limiter._buckets.clear()

        # 2. [R23-HIGH-02] POST /cancel_generate ClientDisconnect -> 400 (never 500) & Content-Length pre-validation
        st_disc, _, _, _ = asyncio.run(
            _invoke_route(
                "/cancel_generate",
                [
                    (b"x-compass-handshake", test_secret.encode("utf-8")),
                    (b"content-type", b"application/json"),
                ],
                [b'{"job_id":"abc'],
                disconnect_after_first=True,
            )
        )
        assert st_disc == 400, f"Expected 400 on /cancel_generate ClientDisconnect, got {st_disc}"
        app_mod._cancel_limiter._buckets.clear()

        for expected_st, cancel_cl_headers in (
            (
                400,
                [
                    (b"x-compass-handshake", test_secret.encode("utf-8")),
                    (b"content-length", b"20"),
                    (b"content-length", b"99999"),
                ],
            ),
            (
                400,
                [
                    (b"x-compass-handshake", test_secret.encode("utf-8")),
                    (b"content-length", b"   "),
                ],
            ),
            (
                400,
                [
                    (b"x-compass-handshake", test_secret.encode("utf-8")),
                    (b"content-length", b"9" * 5000),
                ],
            ),
            (
                413,
                [
                    (b"x-compass-handshake", test_secret.encode("utf-8")),
                    (b"content-length", b"100000"),
                ],
            ),
        ):
            st_cl, recv_cl, bytes_cl, _ = asyncio.run(
                _invoke_route("/cancel_generate", cancel_cl_headers, [b'{"cancel_all":true}'])
            )
            assert st_cl == expected_st, f"Expected {expected_st}, got {st_cl}"
            assert recv_cl == 0 and bytes_cl == 0
            app_mod._cancel_limiter._buckets.clear()

        # 3. [R23-LOW-01] Control-char Content-Type ("\r\n", "\x0c") -> 400 & multipart epilogue smuggling -> 400
        for ctrl_ct in (b"\r\n", b"\x0c"):
            st_ctrl, recv_ctrl, bytes_ctrl, _ = asyncio.run(
                _invoke_route(
                    "/generate",
                    [
                        (b"x-compass-handshake", test_secret.encode("utf-8")),
                        (b"content-type", ctrl_ct),
                        (b"content-length", b"0"),
                    ],
                    [b""],
                )
            )
            assert st_ctrl == 400, f"Expected 400 for control-char Content-Type {ctrl_ct!r}, got {st_ctrl}"
            assert recv_ctrl == 0 and bytes_ctrl == 0
            app_mod._generate_limiter._buckets.clear()

        boundary = b"----CompassR24Bnd"
        valid_body = (
            b"------CompassR24Bnd\r\n"
            b'Content-Disposition: form-data; name="meta"\r\n\r\n'
            + valid_meta.encode("utf-8")
            + b"\r\n------CompassR24Bnd\r\n"
            b'Content-Disposition: form-data; name="past_midi"; filename="ok.mid"\r\n'
            b"Content-Type: audio/midi\r\n\r\n"
            + (b"A" * (1024 * 1024 + 64))
            + b"\r\n------CompassR24Bnd--"
        )
        smuggled_tail = (
            b"\r\n------CompassR24Bnd\r\n"
            b'Content-Disposition: form-data; name="meta"\r\n\r\nevil\r\n'
            b"------CompassR24Bnd--\r\n"
        )
        with patch.object(sf_mod, "SpooledTemporaryFile", side_effect=_tracked_spooled):
            for smuggled_chunks in (
                [valid_body + smuggled_tail],
                [valid_body, smuggled_tail],
                [valid_body + b"\r\n" + (b" " * 80)],
            ):
                created_spooled.clear()
                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    st_smug, _, _, _ = asyncio.run(
                        _invoke_route(
                            "/generate",
                            [
                                (b"x-compass-handshake", test_secret.encode("utf-8")),
                                (b"content-type", b"multipart/form-data; boundary=" + boundary),
                            ],
                            smuggled_chunks,
                        )
                    )
                    assert st_smug == 400, f"Expected 400 on smuggled epilogue, got {st_smug}"
                    assert len(created_spooled) == 1 and created_spooled[0].closed is True
                finally:
                    if gc_was_enabled:
                        gc.enable()
                app_mod._generate_limiter._buckets.clear()

            # Valid charset=UTF-8 + split "\r" / "\n" epilogue still returns 200 OK
            def _stub_generate(**_kwargs):
                out_dir = tempfile.mkdtemp(prefix="compass_r24_out_")
                out_path = os.path.join(out_dir, "output_0.mid")
                with open(out_path, "wb") as fh:
                    fh.write(valid_smf)
                return out_path, "audio/midi", {"source": "request"}

            valid_ok_prefix = (
                b"------CompassR24Bnd\r\n"
                b'Content-Disposition: form-data; name="meta"\r\n\r\n'
                + valid_meta.encode("utf-8")
                + b"\r\n------CompassR24Bnd\r\n"
                b'Content-Disposition: form-data; name="past_midi"; filename="ok.mid"\r\n'
                b"Content-Type: audio/midi\r\n\r\n"
                + valid_smf
                + b"\r\n------CompassR24Bnd--"
            )
            with patch.object(app_mod.model_controller, "generate", side_effect=_stub_generate):
                st_ok, _, _, _ = asyncio.run(
                    _invoke_route(
                        "/generate",
                        [
                            (b"x-compass-handshake", test_secret.encode("utf-8")),
                            (b"content-type", b"multipart/form-data; boundary=" + boundary + b"; charset=UTF-8"),
                        ],
                        [valid_ok_prefix, b"\r", b"\n"],
                    )
                )
            assert st_ok == 200, f"Expected 200 on valid UTF-8 split-CRLF epilogue, got {st_ok}"
    finally:
        if prev_token is None:
            os.environ.pop("COMPASS_HANDSHAKE_TOKEN", None)
        else:
            os.environ["COMPASS_HANDSHAKE_TOKEN"] = prev_token
        app_mod._generate_limiter._buckets.clear()
        app_mod._cancel_limiter._buckets.clear()

    print("  [PASS] Charset allowlist, /cancel_generate ClientDisconnect & CL guards, and multipart epilogue guards verified")


def test_22_round25_rfc2231_punycode_dup_charset_and_leading_crlf_zero_part():
    print(
        "\n--- [TEST 22] Round-25 RFC 2231 Punycode DoS, Duplicate Charset/Header Guards, Quoted UTF-8 Stars, & Zero-Part Leading-CRLF ([R24-HIGH-01], [R24-LOW-02]) ---"
    )
    import asyncio
    import struct
    import tempfile
    from unittest.mock import patch
    import app as app_mod

    app_mod._generate_limiter._buckets.clear()
    app_mod._cancel_limiter._buckets.clear()
    prev_token = os.environ.get("COMPASS_HANDSHAKE_TOKEN")
    test_secret = "round25_test_secret_token_7712"
    os.environ["COMPASS_HANDSHAKE_TOKEN"] = test_secret
    valid_meta = json.dumps({"model_type": "MORTM4.5D-80M-SFT-Gen", "task": "meta_past"})
    valid_smf = (
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", 12)
        + b"\x00\x90\x3c\x64\x60\x80\x3c\x00\x00\xff\x2f\x00"
    )

    async def _invoke_route(path_str, headers_list, chunks):
        receive_calls = 0
        bytes_delivered = 0
        idx = 0
        resp_status = None
        resp_body = bytearray()

        async def _receive():
            nonlocal receive_calls, bytes_delivered, idx
            receive_calls += 1
            if idx < len(chunks):
                ch = chunks[idx]
                idx += 1
                bytes_delivered += len(ch)
                return {
                    "type": "http.request",
                    "body": ch,
                    "more_body": idx < len(chunks),
                }
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(msg):
            nonlocal resp_status
            if msg["type"] == "http.response.start":
                resp_status = msg["status"]
            elif msg["type"] == "http.response.body":
                resp_body.extend(msg.get("body", b""))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path_str,
            "raw_path": path_str.encode("ascii"),
            "query_string": b"",
            "headers": headers_list,
            "client": ("127.0.0.1", 54325),
            "server": ("127.0.0.1", 8000),
        }
        await app_mod.app(scope, _receive, _send)
        return resp_status, receive_calls, bytes_delivered, bytes(resp_body)

    try:
        # 1. [R24-HIGH-01] 24-part RFC 2231 punycode / unicode_escape / filename*= / ext*0*= bombs rejected in < 20 ms
        puny_bomb = b"punycode''" + (b"a" * 15800) + b"-a"
        parts_24 = []
        for idx in range(24):
            param_attr = (
                b"ext*="
                if idx % 3 == 0
                else (b"filename*=" if idx % 3 == 1 else b"ext*0*=")
            )
            codec_payload = (
                puny_bomb
                if idx % 3 != 2
                else (b"unicode_escape''" + (b"\\u0041" * 2500))
            )
            parts_24.append(
                b"--r25bnd\r\nContent-Disposition: form-data; name=\"meta\"; "
                + param_attr
                + codec_payload
                + b"\r\n\r\nx\r\n"
            )
        bomb_body = b"".join(parts_24) + b"--r25bnd--\r\n"

        t0 = time.perf_counter()
        st_bomb, _, _, _ = asyncio.run(
            _invoke_route(
                "/generate",
                [
                    (b"x-compass-handshake", test_secret.encode("utf-8")),
                    (b"content-type", b"multipart/form-data; boundary=r25bnd"),
                ],
                [bomb_body],
            )
        )
        bomb_ms = (time.perf_counter() - t0) * 1000.0
        assert st_bomb == 400, f"Expected 400 on RFC 2231 punycode bomb, got {st_bomb}"
        assert bomb_ms < 20.0, f"RFC 2231 gate took {bomb_ms:.2f} ms (expected < 20 ms)"
        app_mod._generate_limiter._buckets.clear()

        # 2. [R24-HIGH-01] Duplicate charset=, RFC 2231 on Content-Type, RFC 2822 comments, & unclosed quotes rejected at 0 bytes read
        for route_path, bad_ct in (
            ("/generate", b"multipart/form-data; boundary=abc; charset=punycode; charset=utf-8"),
            ("/generate", b"application/x-www-form-urlencoded; charset=punycode; charset=utf-8"),
            ("/generate", b"multipart/form-data; boundary=abc; ext*=" + puny_bomb[:400]),
            ("/generate", b"multipart/form-data; boundary=abc; filename*(comment)=utf-8''x"),
            ("/generate", b'multipart/form-data; boundary="unclosed'),
            ("/generate", b"multipart/form-data; boundary=abc; stray_segment"),
            ("/cancel_generate", b"application/json; charset=punycode; charset=utf-8"),
            ("/cancel_generate", b"application/json; ext*=punycode''aaaa-a"),
        ):
            st_hdr, recv_hdr, bytes_hdr, _ = asyncio.run(
                _invoke_route(
                    route_path,
                    [
                        (b"x-compass-handshake", test_secret.encode("utf-8")),
                        (b"content-type", bad_ct),
                    ],
                    [b'{"cancel_all":true}'],
                )
            )
            assert st_hdr == 400, f"Expected 400 on {route_path} {bad_ct[:60]!r}, got {st_hdr}"
            assert recv_hdr == 0 and bytes_hdr == 0
            app_mod._generate_limiter._buckets.clear()
            app_mod._cancel_limiter._buckets.clear()

        # Duplicate Content-Disposition / part Content-Type charset=punycode rejected with 400
        for bad_part_body in (
            (
                b"--r25bnd\r\n"
                b'Content-Disposition: form-data; name="evil"\r\n'
                b'Content-Disposition: form-data; name="meta"\r\n\r\n'
                + valid_meta.encode("utf-8")
                + b"\r\n--r25bnd--\r\n"
            ),
            (
                b"--r25bnd\r\n"
                b'Content-Disposition: form-data; name="past_midi"; filename="ok.mid"\r\n'
                b"Content-Type: audio/midi; charset=punycode\r\n\r\n"
                + valid_smf
                + b"\r\n--r25bnd--\r\n"
            ),
        ):
            st_dup_hdr, _, _, _ = asyncio.run(
                _invoke_route(
                    "/generate",
                    [
                        (b"x-compass-handshake", test_secret.encode("utf-8")),
                        (b"content-type", b"multipart/form-data; boundary=r25bnd"),
                    ],
                    [bad_part_body],
                )
            )
            assert st_dup_hdr == 400
            app_mod._generate_limiter._buckets.clear()

        # 3. [R24-HIGH-01] Valid quoted boundary="bnd*with*stars" & UTF-8 filename="jp_日本語_1*2.mid" -> 200 OK
        def _stub_generate(**_kwargs):
            out_dir = tempfile.mkdtemp(prefix="compass_r25_out_")
            out_path = os.path.join(out_dir, "output_0.mid")
            with open(out_path, "wb") as fh:
                fh.write(valid_smf)
            return out_path, "audio/midi", {"source": "request"}

        star_bnd = b"bnd*with*stars"
        utf8_star_body = (
            b"--"
            + star_bnd
            + b"\r\nContent-Disposition: form-data; name=\"meta\"\r\n\r\n"
            + valid_meta.encode("utf-8")
            + b"\r\n--"
            + star_bnd
            + b"\r\nContent-Disposition: form-data; name=\"past_midi\"; filename=\""
            + "jp_日本語_1*2.mid".encode("utf-8")
            + b"\"\r\nContent-Type: audio/midi\r\n\r\n"
            + valid_smf
            + b"\r\n--"
            + star_bnd
            + b"--\r\n"
        )
        with patch.object(app_mod.model_controller, "generate", side_effect=_stub_generate):
            st_utf8, _, _, _ = asyncio.run(
                _invoke_route(
                    "/generate",
                    [
                        (b"x-compass-handshake", test_secret.encode("utf-8")),
                        (
                            b"content-type",
                            b'multipart/form-data; boundary="bnd*with*stars"; charset=utf-8',
                        ),
                    ],
                    [utf8_star_body],
                )
            )
        assert st_utf8 == 200, f"Expected 200 on quoted stars + UTF-8 filename, got {st_utf8}"
        app_mod._generate_limiter._buckets.clear()

        # 4. [R24-LOW-02] Zero-part leading CRLF b"\r\n--bnd--\r\n" -> 422 while b"\r\n--bnd--\r\nX" -> 400
        for zero_ok_chunks in (
            [b"--bnd--\r\n"],
            [b"\r\n--bnd--\r\n"],
            [b"\r", b"\n", b"--bnd--", b"\r\n"],
        ):
            st_z422, _, _, _ = asyncio.run(
                _invoke_route(
                    "/generate",
                    [
                        (b"x-compass-handshake", test_secret.encode("utf-8")),
                        (b"content-type", b"multipart/form-data; boundary=bnd"),
                    ],
                    zero_ok_chunks,
                )
            )
            assert st_z422 == 422, f"Expected 422 on zero-part {zero_ok_chunks!r}, got {st_z422}"
            app_mod._generate_limiter._buckets.clear()

        for zero_bad_chunks in (
            [b"\r\n--bnd--\r\nX"],
            [b"\r", b"\n", b"--bnd--", b"\r", b"\n", b"X"],
        ):
            st_z400, _, _, _ = asyncio.run(
                _invoke_route(
                    "/generate",
                    [
                        (b"x-compass-handshake", test_secret.encode("utf-8")),
                        (b"content-type", b"multipart/form-data; boundary=bnd"),
                    ],
                    zero_bad_chunks,
                )
            )
            assert st_z400 == 400, f"Expected 400 on smuggled zero-part {zero_bad_chunks!r}, got {st_z400}"
            app_mod._generate_limiter._buckets.clear()
    finally:
        if prev_token is None:
            os.environ.pop("COMPASS_HANDSHAKE_TOKEN", None)
        else:
            os.environ["COMPASS_HANDSHAKE_TOKEN"] = prev_token
        app_mod._generate_limiter._buckets.clear()
        app_mod._cancel_limiter._buckets.clear()

    print("  [PASS] RFC 2231 punycode gate, duplicate charset/header rejection, UTF-8 quoted stars, and zero-part leading CRLF verified")


def test_23_round26_backslash_boundary_quote_preservation():
    """[R25-LOW-03] Quoted multipart boundaries ending in literal backslashes (`boundary="abc\\\\"`) must not swallow `; charset=...`."""
    print("\n=== [Test 23] Round-26 Backslash-Boundary Quote & Charset Preservation ===")
    import asyncio
    import json
    import struct
    import tempfile
    from unittest.mock import patch
    import app as app_mod

    prev_token = os.environ.get("COMPASS_HANDSHAKE_TOKEN")
    test_secret = "f" * 64
    os.environ["COMPASS_HANDSHAKE_TOKEN"] = test_secret
    app_mod._generate_limiter._buckets.clear()

    valid_meta = json.dumps(
        {
            "model_type": "MORTM4.5D-80M-SFT-Gen",
            "task": "meta_past",
            "program": ["PIANO"],
            "tempo": 120,
            "num_gems": 1,
        }
    )
    valid_smf = (
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", 12)
        + b"\x00\x90\x3c\x64\x60\x80\x3c\x00\x00\xff\x2f\x00"
    )

    async def _invoke_generate(headers_list, chunks):
        idx = 0
        resp_status = None

        async def _receive():
            nonlocal idx
            if idx < len(chunks):
                ch = chunks[idx]
                idx += 1
                return {"type": "http.request", "body": ch, "more_body": idx < len(chunks)}
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(msg):
            nonlocal resp_status
            if msg["type"] == "http.response.start":
                resp_status = msg["status"]

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/generate",
            "raw_path": b"/generate",
            "query_string": b"",
            "headers": headers_list,
            "client": ("127.0.0.1", 54326),
            "server": ("127.0.0.1", 8000),
        }
        await app_mod.app(scope, _receive, _send)
        return resp_status

    def _stub_generate(**_kwargs):
        out_dir = tempfile.mkdtemp(prefix="compass_r26_out_")
        out_path = os.path.join(out_dir, "output_0.mid")
        with open(out_path, "wb") as fh:
            fh.write(valid_smf)
        return out_path, "audio/midi", {"source": "request"}

    try:
        for raw_bnd, ct_header in (
            (b"abc\\", b'multipart/form-data; boundary="abc\\\\"; charset=utf-8'),
            (b"abc\\\\", b'multipart/form-data; charset=utf-8; boundary="abc\\\\\\\\"'),
            (b'a\\"b\\', b'multipart/form-data; boundary="a\\\\\\"b\\\\"'),
        ):
            mp_body = (
                b"--"
                + raw_bnd
                + b'\r\nContent-Disposition: form-data; name="meta"\r\n\r\n'
                + valid_meta.encode("utf-8")
                + b"\r\n--"
                + raw_bnd
                + b'\r\nContent-Disposition: form-data; name="past_midi"; filename="'
                + "jp_日本語.mid".encode("utf-8")
                + b'"\r\nContent-Type: audio/midi\r\n\r\n'
                + valid_smf
                + b"\r\n--"
                + raw_bnd
                + b"--\r\n"
            )
            with patch.object(app_mod.model_controller, "generate", side_effect=_stub_generate):
                st_ok = asyncio.run(
                    _invoke_generate(
                        [
                            (b"x-compass-handshake", test_secret.encode("utf-8")),
                            (b"content-type", ct_header),
                        ],
                        [mp_body],
                    )
                )
            assert st_ok == 200, f"Expected 200 for boundary={raw_bnd!r}, got {st_ok}"
            app_mod._generate_limiter._buckets.clear()
    finally:
        if prev_token is None:
            os.environ.pop("COMPASS_HANDSHAKE_TOKEN", None)
        else:
            os.environ["COMPASS_HANDSHAKE_TOKEN"] = prev_token
        app_mod._generate_limiter._buckets.clear()

    print("  [PASS] Trailing-backslash & escaped-quote multipart boundaries preserved cleanly")


if __name__ == "__main__":
    test_1_appdata_models_dir_and_custom_dropin()
    test_2_lora_folding_mathematical_equivalence()
    test_2b_1measure_midi_and_omega_vocab_and_alibi_mask()
    test_3_all_5_sft_tasks_and_foundation_across_backends()
    test_4_fastapi_endpoints_and_security_guardrails()
    test_5_speed_benchmarks()
    test_6_adversarial_round3_regressions()
    test_7_round2_critic_remediations()
    test_8_round3_fact_checker_remediations()
    test_9_round4_fact_checker_remediations()
    test_10_round5_zero_banned_ast_imports_and_zero_multipart_warnings()
    test_11_round6_critic_remediations()
    test_12_round5_critic_r4_remediations()
    test_13_round6_fact_checker_r5_remediations()
    test_14_round7_fact_checker_r6_remediations()
    test_15_round8_fact_checker_r7_remediations()
    test_16_round9_fact_checker_r8_remediations()
    test_17_round19_generate_meta_recursion_and_size_guards()
    test_18_round21_multipart_shadowing_and_smuggling_guards()
    test_19_round22_pre_auth_zero_spool_and_disconnect_cleanup()
    test_20_round23_truncated_multipart_semicolon_and_header_guards()
    test_21_round24_charset_punycode_cancel_disconnect_and_epilogue_guards()
    test_22_round25_rfc2231_punycode_dup_charset_and_leading_crlf_zero_part()
    test_23_round26_backslash_boundary_quote_preservation()
    print("\nALL PHYSICAL RUNTIME TESTS PASSED!")
