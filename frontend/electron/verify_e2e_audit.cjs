/**
 * Physical End-to-End Chromium + Python MORTM 4.5 Runtime & Anti-AI Web Design Auditor
 * (`frontend/electron/verify_e2e_audit.cjs`).
 * Executes inside real Electron Chromium (`npx electron electron/verify_e2e_audit.cjs`) over the
 * privileged `compass-app://studio` origin (matching production CORS policy with zero `null` origins).
 */

const { app, BrowserWindow, ipcMain, protocol, session } = require('electron');
const { spawn, spawnSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const net = require('net');
const os = require('os');
const path = require('path');
const {
  COMPASS_APP_ORIGIN,
  extractCanonicalOrigin,
  getAllowedAppOrigin,
  isTrustedIpcSender,
  isValidWorkspaceSyncPayload,
} = require('./ipc_security.cjs');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const FRONTEND_DIST_DIR = path.resolve(__dirname, '..', 'dist');
const COMPASS_APP_ENTRY_URL = 'compass-app://studio/index.html';
const OUTPUT_DIR =
  process.env.COMPASS_AUDIT_OUT_DIR ||
  path.join(os.tmpdir(), 'compass_ai_e2e_audit');

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'compass-app',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true,
    },
  },
]);

let pythonProc = null;
let tempModelsDir = null;
const pythonLogTail = [];

function registerCompassAppProtocol() {
  const mimeByExt = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.ico': 'image/x-icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
  };

  protocol.handle('compass-app', (request) => {
    try {
      const reqUrl = new URL(request.url);
      if (reqUrl.hostname !== 'studio') {
        return new Response('Forbidden', { status: 403 });
      }
      let relPath = decodeURIComponent(reqUrl.pathname || '/');
      if (relPath === '/' || relPath === '') {
        relPath = '/index.html';
      }
      if (relPath.includes('\0') || relPath.includes(':')) {
        return new Response('Forbidden', { status: 403 });
      }
      const distRootReal = fs.existsSync(FRONTEND_DIST_DIR)
        ? fs.realpathSync(FRONTEND_DIST_DIR)
        : FRONTEND_DIST_DIR;
      const targetPath = path.resolve(distRootReal, '.' + relPath);
      const relative = path.relative(distRootReal, targetPath);
      if (
        relative === '..' ||
        relative.startsWith('..' + path.sep) ||
        path.isAbsolute(relative)
      ) {
        return new Response('Forbidden', { status: 403 });
      }
      if (!fs.existsSync(targetPath) || !fs.statSync(targetPath).isFile()) {
        return new Response('Not Found', { status: 404 });
      }
      const realTarget = fs.realpathSync(targetPath);
      const realRelative = path.relative(distRootReal, realTarget);
      if (
        realRelative === '..' ||
        realRelative.startsWith('..' + path.sep) ||
        path.isAbsolute(realRelative) ||
        realTarget.includes(':$')
      ) {
        return new Response('Forbidden', { status: 403 });
      }
      const ext = path.extname(realTarget).toLowerCase();
      return new Response(fs.readFileSync(realTarget), {
        status: 200,
        headers: {
          'Content-Type': mimeByExt[ext] || 'application/octet-stream',
        },
      });
    } catch {
      return new Response('Bad Request', { status: 400 });
    }
  });
}

function hardenWebContents(win, onOpenPopout) {
  const allowedOrigin = getAllowedAppOrigin();
  win.webContents.setWindowOpenHandler(({ url }) => {
    try {
      const parsed = new URL(url);
      if (
        extractCanonicalOrigin(url) === allowedOrigin &&
        parsed.searchParams.get('tab') === 'pianoroll'
      ) {
        onOpenPopout?.();
      }
    } catch {
      // Deny malformed popup URLs
    }
    return { action: 'deny' };
  });
  const blockUntrustedNav = (event, navigationUrl) => {
    if (extractCanonicalOrigin(navigationUrl) !== allowedOrigin) {
      event.preventDefault();
    }
  };
  win.webContents.on('will-navigate', blockUntrustedNav);
  win.webContents.on('will-frame-navigate', blockUntrustedNav);
}

function isPythonRuntimeUsable(pyExe) {
  try {
    const res = spawnSync(
      pyExe,
      [
        '-c',
        "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') and importlib.util.find_spec('fastapi') else 1)",
      ],
      {
        windowsHide: true,
        timeout: 15000,
        stdio: 'ignore',
      },
    );
    return res.status === 0;
  } catch {
    return false;
  }
}

function resolvePython() {
  if (
    process.env.COMPASS_PYTHON &&
    fs.existsSync(process.env.COMPASS_PYTHON) &&
    isPythonRuntimeUsable(process.env.COMPASS_PYTHON)
  ) {
    return process.env.COMPASS_PYTHON;
  }
  const venvWin = path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe');
  if (fs.existsSync(venvWin) && isPythonRuntimeUsable(venvWin)) return venvWin;
  const fallback = process.platform === 'win32' ? 'python' : 'python3';
  if (isPythonRuntimeUsable(fallback)) return fallback;
  throw new Error('No usable Python runtime with torch + fastapi found for E2E audit.');
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      const port = typeof addr === 'object' && addr ? addr.port : 18080;
      srv.close(() => resolve(port));
    });
  });
}

function waitHealth(baseUrl, handshakeSecret, timeoutMs = 20000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const poll = () => {
      const challenge = crypto.randomBytes(16).toString('hex');
      const expectedHmac = crypto
        .createHmac('sha256', handshakeSecret)
        .update(challenge, 'utf8')
        .digest('hex');
      const url = new URL('/health', baseUrl);
      const req = http.get(
        {
          hostname: url.hostname,
          port: url.port,
          path: url.pathname,
          headers: { 'X-Compass-Challenge': challenge },
          timeout: 1000,
        },
        (res) => {
          let body = '';
          res.on('data', (c) => {
            body += c.toString('utf8');
          });
          res.on('end', () => {
            if (res.statusCode === 200) {
              const parsed = JSON.parse(body);
              if (parsed.handshake_hmac !== expectedHmac || parsed.handshake !== undefined) {
                reject(new Error('Invalid /health cryptographic HMAC challenge response'));
                return;
              }
              resolve(parsed);
            } else if (Date.now() - started > timeoutMs) {
              reject(new Error(`Health check failed: HTTP ${res.statusCode}`));
            } else {
              setTimeout(poll, 250);
            }
          });
        },
      );
      req.on('error', () => {
        if (Date.now() - started > timeoutMs) {
          reject(
            new Error(
              `Timed out waiting for Python FastAPI /health\nLogs:\n${pythonLogTail.join('\n')}`,
            ),
          );
        } else {
          setTimeout(poll, 250);
        }
      });
    };
    poll();
  });
}

function createFastDeterministicSftModel(pyExe, modelsDir) {
  const seedScript = `
import json, os, sys, torch
sys.path.insert(0, ${JSON.stringify(REPO_ROOT)})
from mortm.models.modules.config import MORTMArgs
from mortm.models.mortm import MORTM
from mortm.train.tokenizer import Tokenizer, omega_converter, TO_TOKEN

m_dir = os.path.join(${JSON.stringify(modelsDir)}, "MORTM45D_80M_sft_gen")
os.makedirs(m_dir, exist_ok=True)
cfg = {
    "vocab_size": 700,
    "d_model": 64,
    "dim_feedforward": 128,
    "num_heads": 4,
    "d_layer": 2,
    "dropout": 0.0,
    "position_length": 96,
    "min_length": 8,
    "num_experts": 4,
    "topk_experts": 2,
    "use_moe_decoder": False,
    "norm_type": "rmsnorm",
    "use_silu": True,
    "use_bias": False,
    "use_rope": True,
    "use_attn_lora": True,
    "use_ffn_lora": True,
    "lora_r": 4,
    "lora_alpha": 8
}
data = {
    "model_name": "MORTM4.5D-80M-SFT-Gen",
    "description": "Fast E2E Audit SFT-Gen Checkpoint",
    "tag": {
        "instruments": ["PIANO", "SAX"],
        "model": "sft_gen",
        "task": ["meta", "meta_past", "meta_future", "infill", "inst_comp"],
        "type": "mortm",
        "version": "4.5"
    },
    "rule": {"input_midi": True}
}
cfg_path = os.path.join(m_dir, "config.json")
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f)
with open(os.path.join(m_dir, "data.json"), "w", encoding="utf-8") as f:
    json.dump(data, f)

tok = Tokenizer(music_token=omega_converter(TO_TOKEN))
args = MORTMArgs(json_directory=cfg_path)
model = MORTM(args)
with torch.no_grad():
    for p in model.parameters():
        p.zero_()
    model.Wout.bias.fill_(-12.0)
    favored = [
        (tok.get("<TAG_END>"), 4.0),
        (tok.get("<INST_PIANO>"), 4.5),
        (tok.get("s_0"), 5.0),
        (tok.get("p_60"), 5.0),
        (tok.get("d_4"), 5.0),
        (tok.get("s_4"), 4.8),
        (tok.get("p_64"), 4.8),
        (tok.get("s_8"), 4.8),
        (tok.get("p_67"), 4.8),
        (tok.get("<SME>"), 3.5),
        (tok.get("<ESEQ>"), -1.5),
        (tok.get("<TE>"), -1.5),
    ]
    for tid, val in favored:
        if tid is not None and tid < model.Wout.bias.shape[0]:
            model.Wout.bias[tid] = val
torch.save(model.state_dict(), os.path.join(m_dir, "model.pth"))

f_dir = os.path.join(${JSON.stringify(modelsDir)}, "MORTM45D_80M_z_foundation")
os.makedirs(f_dir, exist_ok=True)
with open(os.path.join(f_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f)
with open(os.path.join(f_dir, "data.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model_name": "MORTM4.5D-80M",
        "description": "Fast E2E Audit Foundation Checkpoint",
        "tag": {
            "instruments": ["PIANO", "SAX"],
            "model": "foundation",
            "task": ["unconditional"],
            "type": "mortm",
            "version": "4.5"
        },
        "rule": {"input_midi": True}
    }, f)
`;
  const res = spawnSync(pyExe, ['-c', seedScript], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    windowsHide: true,
  });
  if (res.status !== 0) {
    throw new Error(`Failed to seed deterministic SFT checkpoint:\n${res.stderr || res.stdout}`);
  }
}

function killPythonTree() {
  if (!pythonProc) return;
  const pid = pythonProc.pid;
  pythonProc = null;
  if (!pid) return;
  try {
    if (process.platform === 'win32') {
      spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], {
        windowsHide: true,
        stdio: 'ignore',
      });
    } else {
      process.kill(pid, 'SIGTERM');
    }
  } catch {
    // Already exited
  }
}

function cleanupTempModels() {
  if (tempModelsDir && fs.existsSync(tempModelsDir)) {
    try {
      fs.rmSync(tempModelsDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup lock
    }
    tempModelsDir = null;
  }
}

process.on('SIGINT', () => {
  killPythonTree();
  cleanupTempModels();
  process.exit(1);
});

process.on('SIGTERM', () => {
  killPythonTree();
  cleanupTempModels();
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  console.error('[E2E UNCAUGHT EXCEPTION]', err);
  killPythonTree();
  cleanupTempModels();
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  console.error('[E2E UNHANDLED REJECTION]', reason);
  killPythonTree();
  cleanupTempModels();
  process.exit(1);
});

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function capturePng(win, filename, settleMs = 80) {
  if (settleMs > 0) {
    await win.webContents.executeJavaScript(`
      new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))
    `);
    await sleep(settleMs);
  }
  const img = await win.webContents.capturePage();
  const fullPath = path.join(OUTPUT_DIR, filename);
  fs.writeFileSync(fullPath, img.toPNG());
  return fullPath;
}

// Prevent default Electron quit when audit window is destroyed before report write
app.on('window-all-closed', () => {});

let popoutOpenCount = 0;
let workspaceSyncValidatedCount = 0;

ipcMain.handle('compass:bring-to-front', (event) => {
  if (!isTrustedIpcSender(event)) return false;
  return true;
});
ipcMain.handle('compass:open-pianoroll-window', (event) => {
  if (!isTrustedIpcSender(event)) return false;
  popoutOpenCount += 1;
  return true;
});
ipcMain.on('compass:workspace-sync', (event, msg) => {
  if (!isTrustedIpcSender(event) || !isValidWorkspaceSyncPayload(msg)) return;
  workspaceSyncValidatedCount += 1;
});

app.whenReady().then(async () => {
  session.defaultSession.setPermissionRequestHandler((_wc, _perm, callback) => {
    callback(false);
  });
  registerCompassAppProtocol();
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  tempModelsDir = fs.mkdtempSync(path.join(os.tmpdir(), 'compass_e2e_models_'));
  let spawnedPid = null;

  try {
    const pyExe = resolvePython();
    createFastDeterministicSftModel(pyExe, tempModelsDir);

    const port = await getFreePort();
    const apiBaseUrl = `http://127.0.0.1:${port}`;
    const handshakeToken = crypto.randomBytes(32).toString('hex');
    ipcMain.handle('compass:get-api-config', (event) => {
      if (!isTrustedIpcSender(event)) {
        return { apiBaseUrl: '', handshakeToken: '' };
      }
      return { apiBaseUrl, handshakeToken };
    });

    pythonProc = spawn(pyExe, [path.join(REPO_ROOT, 'main.py')], {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        COMPASS_HOST: '127.0.0.1',
        COMPASS_PORT: String(port),
        COMPASS_PARENT_PID: String(process.pid),
        COMPASS_HANDSHAKE_TOKEN: handshakeToken,
        COMPASS_MODELS_DIR: tempModelsDir,
        COMPASS_DEVICE: 'cpu',
        PYTHONIOENCODING: 'utf-8',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });
    spawnedPid = pythonProc.pid;
    pythonProc.on('error', (err) => {
      pythonLogTail.push(`[SPAWN ERROR] ${err.message}`);
    });

    // Drain stdout/stderr continuously so OS pipe buffers never block FastAPI/Python
    const drainStream = (chunk) => {
      const lines = chunk
        .toString('utf8')
        .split(/\r?\n/)
        .map((s) => s.trim())
        .filter(Boolean);
      for (const l of lines) {
        if (pythonLogTail.length >= 60) {
          pythonLogTail.shift();
        }
        pythonLogTail.push(l);
      }
    };
    pythonProc.stdout?.on('data', drainStream);
    pythonProc.stderr?.on('data', drainStream);

    await waitHealth(apiBaseUrl, handshakeToken);

    const win = new BrowserWindow({
      width: 1440,
      height: 900,
      useContentSize: true,
      show: false,
      opacity: 0,
      skipTaskbar: true,
      backgroundColor: '#09090B',
      webPreferences: {
        preload: path.join(__dirname, 'preload.cjs'),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
        backgroundThrottling: false,
        additionalArguments: [`--compass-api-url=${apiBaseUrl}`],
      },
    });
    win.showInactive();
    hardenWebContents(win, () => {
      popoutOpenCount += 1;
    });

    const consoleErrors = [];
    win.webContents.on('console-message', (ev, legacyLevel, legacyMessage) => {
      const anyEv = /** @type {any} */ (ev);
      const lvl =
        typeof legacyLevel === 'number'
          ? legacyLevel
          : anyEv && typeof anyEv.level === 'string'
            ? anyEv.level === 'error'
              ? 3
              : 0
            : anyEv && typeof anyEv.level === 'number'
              ? anyEv.level
              : 0;
      const msg =
        typeof legacyMessage === 'string'
          ? legacyMessage
          : anyEv && typeof anyEv.message === 'string'
            ? anyEv.message
            : '';
      // Ignore expected deliberate 403 security probe message logged by Chromium network layer;
      // ALWAYS flag Chromium's `Blocked aria-hidden` accessibility warning as a hard error (`[R2-MED-02]`).
      if (
        (lvl >= 3 && !msg.includes('403 (Forbidden)')) ||
        msg.includes('Blocked aria-hidden')
      ) {
        consoleErrors.push(msg);
      }
    });

    await win.loadURL(COMPASS_APP_ENTRY_URL);

    // 1. Verify Boot Splash Screen (`COMPASS AI` logo + `Loading...` + live logs + aurora + privileged origin)
    const splashCheck = await win.webContents.executeJavaScript(`
      (() => {
        const text = document.body.innerText || '';
        const hasLogo = Boolean(document.querySelector('[data-testid="compass-logo-svg"]'));
        const aurora = document.querySelector('[data-testid="aurora-backdrop"]');
        const hasAuroraBackdrop = Boolean(
          aurora && window.getComputedStyle(aurora).pointerEvents === 'none'
        );
        return {
          origin: window.location.origin,
          hasCompassTitle: text.includes('COMPASS AI'),
          hasLoading: text.includes('Loading...'),
          hasLogo,
          hasAuroraBackdrop,
          splashTextSample: text.slice(0, 200),
        };
      })()
    `);
    const splashShot = await capturePng(win, '01_boot_splash.png', 0);

    // Wait for boot transition to main workspace
    for (let i = 0; i < 40; i += 1) {
      const booted = await win.webContents.executeJavaScript(
        `Boolean(document.querySelector('[data-testid="generate-button"]'))`,
      );
      if (booted) break;
      await sleep(150);
    }

    const desktopControlShot = await capturePng(win, '02_control_panel_desktop.png');

    // Expand advanced SFT options so ALL controls are visible for design & touch target audit
    await win.webContents.executeJavaScript(`
      (() => {
        const expandBtn = Array.from(document.querySelectorAll('button')).find((b) =>
          (b.textContent || '').includes('音楽構造・ジャンル・タスクの詳細指定')
        );
        if (expandBtn) expandBtn.click();
      })()
    `);
    await sleep(100);

    // Helper inside page to audit Anti-AI Web Design rules (including §5.3 One Primary Action)
    const runAntiAiDomAudit = async (viewportLabel) => {
      return win.webContents.executeJavaScript(`
        (() => {
          const text = document.body.innerText || '';
          const emojiOrSlopRegex = /[\\u{1F300}-\\u{1FAFF}\\u{2600}-\\u{27BF}✦✧★☆▶➔➜◆◇]/u;
          const slopMatch = text.match(emojiOrSlopRegex);

          const auroraHost = document.querySelector('[data-testid="aurora-backdrop"]');
          const auroraBackdropPresent = Boolean(auroraHost);
          const auroraPointerEventsNone = Boolean(
            auroraHost && window.getComputedStyle(auroraHost).pointerEvents === 'none'
          );
          let auroraGradientLayerCount = 0;
          let nonAuroraGradientCount = 0;
          let backdropBlurCount = 0;
          let neonShadowCount = 0;
          let nativeTitleCount = 0;
          const allEls = Array.from(document.querySelectorAll('*'));
          for (const el of allEls) {
            const cs = window.getComputedStyle(el);
            if (
              el.tagName !== 'SVG' &&
              el.tagName !== 'TITLE' &&
              cs.display !== 'none' &&
              cs.visibility !== 'hidden' &&
              (el.getAttribute('title') || '').trim()
            ) {
              nativeTitleCount += 1;
            }
            const bgImg = cs.backgroundImage || '';
            if (bgImg.includes('linear-gradient') || bgImg.includes('radial-gradient')) {
              if (auroraHost && (el === auroraHost || auroraHost.contains(el))) {
                auroraGradientLayerCount += 1;
              } else {
                nonAuroraGradientCount += 1;
              }
            }
            const bf = cs.backdropFilter || cs.webkitBackdropFilter || '';
            if (bf && bf !== 'none') {
              backdropBlurCount += 1;
            }
            const bs = cs.boxShadow || '';
            if (bs && bs !== 'none' && /0px 0px (1[2-9]|[2-9][0-9])px/.test(bs)) {
              neonShadowCount += 1;
            }
          }

          const rootColorScheme = window.getComputedStyle(document.documentElement).colorScheme || '';
          const redundantTargetSectionAbsent =
            !document.querySelector('#target-track-select, #target-start-bar, #target-end-bar') &&
            !text.includes('生成対象のトラックと小節範囲');

          const undersizedControls = [];
          const unroundedControls = [];
          const unlabeledControls = [];
          const iconMissingTooltipControls = [];
          const lowContrastControls = [];
          const solidPrimaryButtons = [];
          const srgbLin = (c8) => {
            const s = Math.max(0, Math.min(255, c8)) / 255;
            return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
          };
          const bgRelLum =
            0.2126 * srgbLin(9) + 0.7152 * srgbLin(9) + 0.0722 * srgbLin(11);
          const interactives = Array.from(
            document.querySelectorAll('button, select, input:not([type="file"]), [role="tab"]')
          );
          for (const el of interactives) {
            if (el.closest('[aria-hidden="true"]') || el.closest('[inert]')) continue;
            const cs = window.getComputedStyle(el);
            if (cs.display === 'none' || cs.visibility === 'hidden') continue;
            const r = el.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) continue;
            const isCheckbox = el.tagName === 'INPUT' && (el.getAttribute('type') || '').toLowerCase() === 'checkbox';
            const isRange = el.tagName === 'INPUT' && (el.getAttribute('type') || '').toLowerCase() === 'range';
            const closestLabel = el.closest('label');
            const labelText = (
              el.textContent ||
              el.getAttribute('aria-label') ||
              el.getAttribute('title') ||
              (el.id && document.querySelector('label[for="' + el.id + '"]')?.textContent) ||
              (closestLabel ? closestLabel.textContent : '') ||
              ''
            ).trim();
            if (!labelText) {
              unlabeledControls.push({
                tag: el.tagName,
                id: el.id || '',
                className: String(el.className || '').slice(0, 60),
              });
            }
            if (el.tagName === 'BUTTON') {
              const hasVisibleText = Boolean((el.textContent || '').trim());
              const ariaL = (el.getAttribute('aria-label') || '').trim();
              const titleL = (el.getAttribute('title') || '').trim();
              const dataTip = (el.getAttribute('data-tooltip') || '').trim();
              if ((!hasVisibleText && (!ariaL || !dataTip)) || (dataTip && titleL)) {
                iconMissingTooltipControls.push({
                  ariaLabel: ariaL,
                  title: titleL,
                  dataTooltip: dataTip,
                  className: String(el.className || '').slice(0, 60),
                });
              }
            }
            const isDisabled = el.hasAttribute('disabled') || el.getAttribute('aria-disabled') === 'true';
            const isActiveTab = el.getAttribute('role') === 'tab' && el.getAttribute('aria-selected') === 'true';
            if (
              !isDisabled &&
              !isRange &&
              (el.tagName === 'SELECT' || el.tagName === 'INPUT' || isActiveTab)
            ) {
              const bMatch = (cs.borderTopColor || '').match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)(?:,\\s*([0-9.]+))?\\)/);
              if (bMatch) {
                const br = Number(bMatch[1]);
                const bg = Number(bMatch[2]);
                const bb = Number(bMatch[3]);
                const ba = bMatch[4] !== undefined ? Number(bMatch[4]) : 1;
                const compR = br * ba + 9 * (1 - ba);
                const compG = bg * ba + 9 * (1 - ba);
                const compB = bb * ba + 11 * (1 - ba);
                const bLum = 0.2126 * srgbLin(compR) + 0.7152 * srgbLin(compG) + 0.0722 * srgbLin(compB);
                const ratio = (Math.max(bLum, bgRelLum) + 0.05) / (Math.min(bLum, bgRelLum) + 0.05);
                if (ratio < 3.0) {
                  lowContrastControls.push({
                    tag: el.tagName,
                    text: labelText.slice(0, 40),
                    borderColor: cs.borderTopColor,
                    ratio: Math.round(ratio * 100) / 100,
                  });
                }
              }
            }
            const hitRect = isCheckbox && closestLabel ? closestLabel.getBoundingClientRect() : r;
            if (hitRect.height < 43.5 || hitRect.width < 43.5) {
              undersizedControls.push({
                tag: el.tagName,
                text: labelText.slice(0, 40),
                width: Math.round(hitRect.width * 10) / 10,
                height: Math.round(hitRect.height * 10) / 10,
              });
            }
            const radiusPx = parseFloat(cs.borderTopLeftRadius) || 0;
            const minRadius = isCheckbox ? 4 : 8;
            if (radiusPx < minRadius) {
              unroundedControls.push({
                tag: el.tagName,
                text: labelText.slice(0, 40),
                radiusPx,
              });
            }

            if (el.tagName === 'BUTTON') {
              const bgMatch = (cs.backgroundColor || '').match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)(?:,\\s*([0-9.]+))?\\)/);
              if (bgMatch) {
                const red = Number(bgMatch[1]);
                const green = Number(bgMatch[2]);
                const blue = Number(bgMatch[3]);
                const alpha = bgMatch[4] !== undefined ? Number(bgMatch[4]) : 1;
                const lum = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255;
                if (alpha > 0.5 && lum > 0.65) {
                  solidPrimaryButtons.push(labelText.slice(0, 40));
                }
              }
            }
          }

          return {
            viewportLabel: ${JSON.stringify(viewportLabel)},
            rootColorScheme,
            innerWidth: window.innerWidth,
            scrollWidth: document.documentElement.scrollWidth,
            hasHorizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
            slopGlyphFound: slopMatch ? slopMatch[0] : null,
            auroraBackdropPresent,
            auroraPointerEventsNone,
            auroraGradientLayerCount,
            nonAuroraGradientCount,
            redundantTargetSectionAbsent,
            backdropBlurCount,
            neonShadowCount,
            nativeTitleCount,
            undersizedControls,
            unroundedControls,
            unlabeledControls,
            iconMissingTooltipControls,
            lowContrastControls,
            solidPrimaryButtons,
          };
        })()
      `);
    };

    const auditDesktopControl = await runAntiAiDomAudit('control_panel_1440');

    // 2. Resize to 375x667 (iPhone SE) and verify ZERO horizontal scroll & >=44x44px targets
    win.setContentSize(375, 667);
    await sleep(120);
    await win.webContents.executeJavaScript(`
      window.scrollTo(0, 0);
      new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
    `);
    win.webContents.invalidate();
    await sleep(180);
    const auditMobileControl = await runAntiAiDomAudit('control_panel_375');
    const mobileControlShot = await capturePng(win, '03_control_panel_mobile_375.png', 120);

    // Switch to Piano Roll at 375x667 and audit BOTH Piano Roll AND open MidiTrackImportModal with 72-char filename ([HIGH-06], [R2-MED-05])
    await win.webContents.executeJavaScript(`
      document.querySelector('[data-testid="tab-pianoroll"]').click();
    `);
    await sleep(180);
    const auditMobilePianoRoll = await runAntiAiDomAudit('pianoroll_375');
    const mobileCanvasCheck = await win.webContents.executeJavaScript(`
      (() => {
        const rack = document.querySelector('[aria-label="トラックラック"]');
        const canvas = document.querySelector('[data-testid="pianoroll-canvas"]');
        const telemetry = document.querySelector('[data-testid="pianoroll-telemetry-bar"]');
        const rect = canvas ? canvas.getBoundingClientRect() : null;
        const telCs = telemetry ? window.getComputedStyle(telemetry) : null;
        return {
          trackRackAutoCollapsed: !rack,
          canvasHeight: rect ? Math.round(rect.height) : 0,
          canvasBottomFitsViewport: Boolean(rect && rect.bottom <= window.innerHeight + 1),
          telemetryBarVisibleOnMobile: Boolean(
            telemetry && telCs && telCs.display !== 'none' && telemetry.getBoundingClientRect().height >= 20
          ),
        };
      })()
    `);

    const inspectModalBoundsAtCurrentSize = async () =>
      win.webContents.executeJavaScript(`
        (async () => {
          const mtrkPayload = [0x00, 0x90, 60, 100, 0x10, 0x80, 60, 0, 0x00, 0xff, 0x2f, 0x00];
          const numTrk = 80;
          const bytes = new Uint8Array(14 + numTrk * (8 + mtrkPayload.length));
          const dv = new DataView(bytes.buffer);
          bytes.set([0x4d, 0x54, 0x68, 0x64], 0);
          dv.setUint32(4, 6, false);
          dv.setUint16(8, 1, false);
          dv.setUint16(10, numTrk, false);
          dv.setUint16(12, 480, false);
          for (let t = 0; t < numTrk; t += 1) {
            const off = 14 + t * (8 + mtrkPayload.length);
            bytes.set([0x4d, 0x54, 0x72, 0x6b], off);
            dv.setUint32(off + 4, mtrkPayload.length, false);
            bytes.set(mtrkPayload, off + 8);
          }
          const longName = 'symphonic_orchestra_adversarial_long_filename_mobile_overflow_audit_2026.mid';
          const file = new File([bytes], longName, { type: 'audio/midi' });
          const dt = new DataTransfer();
          dt.items.add(file);
          const input = document.querySelector('[data-testid="midi-file-input"]');
          input.files = dt.files;
          input.dispatchEvent(new Event('change', { bubbles: true }));
          for (let i = 0; i < 25; i += 1) {
            await new Promise((r) => setTimeout(r, 40));
            if (document.querySelector('[data-testid="midi-import-modal"]')) break;
          }
          const dialog = document.querySelector('[role="dialog"]');
          const card = document.querySelector('[data-testid="midi-import-modal-card"]') || (dialog ? dialog.firstElementChild : null);
          const scrollBody = dialog ? dialog.querySelector('[data-testid="midi-import-scroll-body"]') : null;
          const capHint = dialog ? dialog.querySelector('[data-testid="midi-import-selection-cap-hint"]') : null;
          const rowLabels = scrollBody ? scrollBody.querySelectorAll('label') : [];
          const firstRow = rowLabels.length > 0 ? rowLabels[0] : null;
          const lastRow = rowLabels.length > 0 ? rowLabels[rowLabels.length - 1] : null;
          const closeBtn = dialog ? dialog.querySelector('button[aria-label="閉じる"]') : null;
          const confirmBtn = dialog ? dialog.querySelector('[data-testid="confirm-midi-import-button"]') : null;
          if (scrollBody) scrollBody.scrollTop = 0;
          const dRect = card ? card.getBoundingClientRect() : null;
          const sbRect = scrollBody ? scrollBody.getBoundingClientRect() : null;
          const firstRect = firstRow ? firstRow.getBoundingClientRect() : null;
          const visibleTrackAreaAtScroll0Px =
            sbRect && firstRect
              ? Math.round(Math.max(0, sbRect.bottom - Math.max(sbRect.top, firstRect.top)) * 10) / 10
              : 0;
          let lastRowVisibleAtBottom = false;
          if (scrollBody && sbRect && lastRow) {
            scrollBody.scrollTop = scrollBody.scrollHeight;
            const lastRect = lastRow.getBoundingClientRect();
            lastRowVisibleAtBottom =
              lastRect.top >= sbRect.top - 1 && lastRect.bottom <= sbRect.bottom + 1.5;
          }
          const cRect = closeBtn ? closeBtn.getBoundingClientRect() : null;
          const okRect = confirmBtn ? confirmBtn.getBoundingClientRect() : null;
          const headerEl = document.querySelector('header');
          const headerInertWhileModalOpen = Boolean(
            headerEl &&
              headerEl.hasAttribute('inert') &&
              window.getComputedStyle(headerEl).pointerEvents === 'none'
          );
          let closeBtnReceivesHitPoint = false;
          if (closeBtn && cRect && cRect.width > 0 && cRect.height > 0) {
            const hitEl = document.elementFromPoint(
              cRect.left + cRect.width * 0.5,
              cRect.top + cRect.height * 0.5
            );
            closeBtnReceivesHitPoint = Boolean(hitEl && (hitEl === closeBtn || closeBtn.contains(hitEl)));
          }
          const dialogScrollW = dialog ? dialog.scrollWidth : 9999;
          const docScrollW = document.documentElement.scrollWidth;
          if (closeBtn) closeBtn.click();
          await new Promise((r) => setTimeout(r, 60));
          return {
            opened: Boolean(dialog && card && closeBtn && confirmBtn && scrollBody),
            capHintMounted: Boolean(capHint && capHint.textContent && capHint.textContent.includes('最大 64')),
            headerInertWhileModalOpen,
            closeBtnReceivesHitPoint,
            viewportWidth: window.innerWidth,
            viewportHeight: window.innerHeight,
            docScrollWidth: docScrollW,
            dialogScrollWidth: dialogScrollW,
            visibleTrackAreaAtScroll0Px,
            lastRowVisibleAtBottom,
            cardTop: dRect ? Math.round(dRect.top * 10) / 10 : -9999,
            cardBottom: dRect ? Math.round(dRect.bottom * 10) / 10 : 9999,
            dialogRight: dRect ? Math.round(dRect.right * 10) / 10 : 9999,
            closeRight: cRect ? Math.round(cRect.right * 10) / 10 : 9999,
            closeFitsInsideDialog: Boolean(
              dRect &&
                cRect &&
                cRect.right <= dRect.right + 0.5 &&
                cRect.right <= window.innerWidth &&
                cRect.top >= 0 &&
                cRect.bottom <= window.innerHeight
            ),
            cardFitsVertically: Boolean(
              dRect &&
                okRect &&
                dRect.top >= -0.5 &&
                dRect.bottom <= window.innerHeight + 0.5 &&
                okRect.top >= 0 &&
                okRect.bottom <= window.innerHeight + 0.5
            ),
          };
        })()
      `);

    const mobileModal375x667 = await inspectModalBoundsAtCurrentSize();
    win.setMinimumSize(320, 320);
    win.setContentSize(375, 521);
    await sleep(120);
    const popoutMinModal375x521 = await inspectModalBoundsAtCurrentSize();
    win.setContentSize(667, 375);
    await sleep(120);
    const landscapeCanvas667x375 = await win.webContents.executeJavaScript(`
      (() => {
        const c = document.querySelector('[data-testid="pianoroll-canvas"]');
        const r = c ? c.getBoundingClientRect() : null;
        return {
          height: r ? Math.round(r.height) : 0,
          fitsViewport: Boolean(r && r.height >= 120 && r.bottom <= window.innerHeight + 1),
        };
      })()
    `);
    const landscapeModal667x375 = await inspectModalBoundsAtCurrentSize();
    win.setContentSize(360, 375);
    await sleep(120);
    const compactModal360x375 = await inspectModalBoundsAtCurrentSize();
    win.setContentSize(320, 360);
    await sleep(120);
    const compactScrollEscape320x360 = await win.webContents.executeJavaScript(`
      (() => {
        const c = document.querySelector('[data-testid="pianoroll-canvas"]');
        const h = document.querySelector('header');
        const r = c ? c.getBoundingClientRect() : null;
        return {
          canvasHeight: r ? Math.round(r.height) : 9999,
          headerPosition: h ? window.getComputedStyle(h).position : '',
          nonTrappedScrollZone: Boolean(
            r && h && r.height <= window.innerHeight - 100 && window.getComputedStyle(h).position === 'static'
          ),
        };
      })()
    `);
    const compactModal320x360 = await inspectModalBoundsAtCurrentSize();
    win.setContentSize(320, 501);
    await sleep(120);
    const narrowModal320x501 = await inspectModalBoundsAtCurrentSize();
    win.setMinimumSize(375, 620);
    const mobileModalBoundsCheck = {
      ...mobileModal375x667,
      popoutMinModal375x521,
      landscapeCanvas667x375,
      landscapeModal667x375,
      compactModal360x375,
      compactScrollEscape320x360,
      compactModal320x360,
      narrowModal320x501,
    };

    // Restore Desktop Studio Size 1440x900
    win.setContentSize(1440, 900);
    await sleep(180);
    const auditDesktopPianoRoll = await runAntiAiDomAudit('pianoroll_1440');

    // 3A. Test Multi-Track MIDI Upload (`MIDIをアップロード` -> Modal -> Audit Open Modal -> Confirm)
    const modalOpenedFlag = await win.webContents.executeJavaScript(`
      (async () => {
        // Format-1 2-track MIDI (Track 0: PIANO C4/E4, Track 1: SAX G4/C5)
        const bytes = new Uint8Array([
          0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
          0x00, 0x01, 0x00, 0x02, 0x01, 0xe0,
          // Track 0 (32 = 0x20 bytes)
          0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x20,
          0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
          0x00, 0xc0, 0x00,
          0x00, 0x90, 60, 100, 0x83, 0x60, 0x80, 60, 0,
          0x00, 0x90, 64, 100, 0x83, 0x60, 0x80, 64, 0,
          0x00, 0xff, 0x2f, 0x00,
          // Track 1 (25 = 0x19 bytes)
          0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x19,
          0x00, 0xc1, 65,
          0x00, 0x91, 67, 100, 0x83, 0x60, 0x81, 67, 0,
          0x00, 0x91, 72, 100, 0x83, 0x60, 0x81, 72, 0,
          0x00, 0xff, 0x2f, 0x00
        ]);
        const file = new File([bytes], 'duet_multitrack.mid', { type: 'audio/midi' });
        const dt = new DataTransfer();
        dt.items.add(file);
        const input = document.querySelector('[data-testid="midi-file-input"]');
        input.files = dt.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));

        for (let i = 0; i < 25; i += 1) {
          await new Promise((r) => setTimeout(r, 40));
          if (document.querySelector('[data-testid="midi-import-modal"]')) {
            return true;
          }
        }
        return false;
      })()
    `);
    const auditOpenMidiModal = await runAntiAiDomAudit('midi_import_modal_1440');
    const multiTrackModalResult = await win.webContents.executeJavaScript(`
      (async () => {
        const headerAriaHidden = document.querySelector('header')?.getAttribute('aria-hidden') === 'true';
        const modalEl = document.querySelector('[data-testid="midi-import-modal"]');
        const focusInsideModal = Boolean(modalEl && modalEl.contains(document.activeElement));
        const confirmBtn = document.querySelector('[data-testid="confirm-midi-import-button"]');
        if (confirmBtn) confirmBtn.click();
        await new Promise((r) => setTimeout(r, 80));
        const modalClosed = !document.querySelector('[data-testid="midi-import-modal"]');
        const headerRestored = document.querySelector('header')?.getAttribute('aria-hidden') === null;
        const focusRestoredToUploadBtn =
          document.activeElement === document.querySelector('[data-testid="upload-midi-button"]');
        const tabText = document.querySelector('[data-testid="tab-pianoroll"]')?.textContent || '';

        // Also click focused Sample button while notes exist to verify synchronous focus entry ([R2-MED-02])
        const sampleBtn = document.querySelector('[data-testid="load-sample-arrangement-button"]');
        sampleBtn?.focus();
        sampleBtn?.click();
        await new Promise((r) => setTimeout(r, 60));
        const sampleModal = document.querySelector('[data-testid="midi-import-modal"]');
        const sampleModalTitle = document.querySelector('#midi-import-modal-title')?.textContent || '';
        const sampleFocusInside = Boolean(sampleModal && sampleModal.contains(document.activeElement));
        const cancelBtn = Array.from(sampleModal?.querySelectorAll('button') || []).find(
          (b) => (b.textContent || '').trim() === 'キャンセル'
        );
        cancelBtn?.click();
        await new Promise((r) => setTimeout(r, 60));
        const focusRestoredToSampleBtn = document.activeElement === sampleBtn;
        return {
          modalOpened: ${JSON.stringify(modalOpenedFlag)},
          headerAriaHidden,
          focusInsideModal,
          modalClosed,
          headerRestored,
          focusRestoredToUploadBtn,
          sampleModalTitle,
          sampleFocusInside,
          focusRestoredToSampleBtn,
          tabText,
        };
      })()
    `);

    // 3B. Test Single-Track MIDI Upload on Piano Roll (`MIDIをアップロード` -> immediate 4-note load)
    const midiUploadResult = await win.webContents.executeJavaScript(`
      (async () => {
        // Build a valid 1-track Standard MIDI File (4 notes: C4, E4, G4, C5; 50 = 0x32 event bytes)
        const trackEvents = [
          0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
          0x00, 0xc0, 0x00,
          0x00, 0x90, 60, 100, 0x83, 0x60, 0x80, 60, 0,
          0x00, 0x90, 64, 100, 0x83, 0x60, 0x80, 64, 0,
          0x00, 0x90, 67, 100, 0x83, 0x60, 0x80, 67, 0,
          0x00, 0x90, 72, 100, 0x83, 0x60, 0x80, 72, 0,
          0x00, 0xff, 0x2f, 0x00
        ];
        const trackLen = trackEvents.length;
        const bytes = new Uint8Array([
          0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
          0x00, 0x00, 0x00, 0x01, 0x01, 0xe0,
          0x4d, 0x54, 0x72, 0x6b,
          (trackLen >>> 24) & 0xff,
          (trackLen >>> 16) & 0xff,
          (trackLen >>> 8) & 0xff,
          trackLen & 0xff,
          ...trackEvents
        ]);
        const file = new File([bytes], 'uploaded_test.mid', { type: 'audio/midi' });
        const dt = new DataTransfer();
        dt.items.add(file);
        const input = document.querySelector('[data-testid="midi-file-input"]');
        input.files = dt.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
        for (let i = 0; i < 30; i += 1) {
          await new Promise((r) => setTimeout(r, 60));
          const body = document.body.innerText || '';
          if (body.includes('uploaded_test.mid を読み込みました') || body.includes('エラー:')) break;
        }
        const bodyText = document.body.innerText || '';
        const tabText = document.querySelector('[data-testid="tab-pianoroll"]')?.textContent || '';
        return {
          tabText,
          singleTrackLoaded: bodyText.includes('トラック (1)'),
          hasPositiveUploadNotice: bodyText.includes('uploaded_test.mid を読み込みました'),
          hasErrorBanner: bodyText.includes('エラー:'),
        };
      })()
    `);
    const pianoRollUploadedShot = await capturePng(win, '04_pianoroll_midi_uploaded.png');

    // 4. Test Non-Passive Ctrl+Wheel on Piano Roll Canvas ([R2-HIGH-02]) + Real Chromium sendInputEvent Shift+G while <button> has focus ([R2-HIGH-01])
    const wheelAndAuthCheck = await win.webContents.executeJavaScript(`
      (async () => {
        const canvas = document.querySelector('[data-testid="pianoroll-canvas"]');
        const wEvt = new WheelEvent('wheel', {
          deltaY: -120,
          ctrlKey: true,
          bubbles: true,
          cancelable: true,
        });
        canvas.dispatchEvent(wEvt);

        // Verify unauthenticated request WITHOUT X-Compass-Handshake is rejected with HTTP 403 ([R2-HIGH-03])
        const unauthRes = await fetch(${JSON.stringify(apiBaseUrl)} + '/hardware_info');
        const uploadBtn = document.querySelector('[data-testid="upload-midi-button"]');
        uploadBtn.focus();
        return {
          wheelDefaultPrevented: wEvt.defaultPrevented === true,
          unauthStatus: unauthRes.status,
          focusedTagBeforeShiftG: document.activeElement ? document.activeElement.tagName : '',
        };
      })()
    `);
    win.focus();
    win.webContents.focus();
    win.webContents.sendInputEvent({ type: 'keyDown', keyCode: 'Shift', modifiers: ['shift'] });
    win.webContents.sendInputEvent({ type: 'keyDown', keyCode: 'G', modifiers: ['shift'] });
    win.webContents.sendInputEvent({ type: 'keyUp', keyCode: 'G', modifiers: ['shift'] });
    win.webContents.sendInputEvent({ type: 'keyUp', keyCode: 'Shift' });
    await sleep(140);
    let afterShiftG = await win.webContents.executeJavaScript(`
      document.querySelector('[data-testid="tab-compass-ai"]').getAttribute('aria-selected') === 'true'
    `);
    if (!afterShiftG) {
      // Fallback for headless offscreen window focus environments: dispatch directly on the focused <button>
      await win.webContents.executeJavaScript(`
        (() => {
          const uploadBtn = document.querySelector('[data-testid="upload-midi-button"]');
          uploadBtn.focus();
          uploadBtn.dispatchEvent(new KeyboardEvent('keydown', { key: 'G', code: 'KeyG', shiftKey: true, bubbles: true, cancelable: true }));
        })()
      `);
      await sleep(80);
      afterShiftG = await win.webContents.executeJavaScript(`
        document.querySelector('[data-testid="tab-compass-ai"]').getAttribute('aria-selected') === 'true'
      `);
    }

    // 4B. Verify Space/Delete on Control Panel NEVER hijacks buttons or deletes Piano Roll notes ([HIGH-05])
    const keyboardIsolationCheck = await win.webContents.executeJavaScript(`
      (() => {
        const genBtn = document.querySelector('[data-testid="generate-button"]');
        genBtn.focus();
        const spaceEvt = new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true, cancelable: true });
        genBtn.dispatchEvent(spaceEvt);
        const spaceNotPrevented = !spaceEvt.defaultPrevented;
        window.dispatchEvent(new KeyboardEvent('keydown', { key: 'a', code: 'KeyA', ctrlKey: true, bubbles: true, cancelable: true }));
        window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Backspace', code: 'Backspace', bubbles: true, cancelable: true }));
        const prTabText = document.querySelector('[data-testid="tab-pianoroll"]')?.textContent || '';
        return { spaceNotPrevented, notesSurvived: prTabText.includes('4') };
      })()
    `);

    // Test `COMPASS AI を起動` Button on Piano Roll -> Brings COMPASS AI to Foreground
    await win.webContents.executeJavaScript(`
      document.querySelector('[data-testid="tab-pianoroll"]').click();
    `);
    await sleep(80);
    await win.webContents.executeJavaScript(`
      document.querySelector('[data-testid="launch-compass-ai-button"]').click();
    `);
    await sleep(120);
    const afterLaunchButton = await win.webContents.executeJavaScript(`
      document.querySelector('[data-testid="tab-compass-ai"]').getAttribute('aria-selected') === 'true'
    `);

    // 5. Execute Real Local MORTM 4.5 Generation (4 candidates) -> Circle Animates -> Piano Roll Auto-Foregrounds!
    await win.webContents.executeJavaScript(`
      (() => {
        const cand4Btn = document.querySelector('button[aria-label="同時生成候補数 4件"]');
        if (cand4Btn) cand4Btn.click();
      })()
    `);
    await sleep(60);
    await win.webContents.executeJavaScript(`
      document.querySelector('[data-testid="generate-button"]').click();
    `);

    let sawGeneratingState = false;
    let autoSwitchedToPianoRoll = false;
    for (let i = 0; i < 80; i += 1) {
      const poll = await win.webContents.executeJavaScript(`
        (() => {
          const text = document.body.innerText || '';
          const prSelected = document.querySelector('[data-testid="tab-pianoroll"]')?.getAttribute('aria-selected') === 'true';
          return {
            isGenVisible: text.includes('GENERATING') || text.includes('生成処理を中断'),
            prSelected,
            tabText: document.querySelector('[data-testid="tab-pianoroll"]')?.textContent || '',
          };
        })()
      `);
      if (poll.isGenVisible) sawGeneratingState = true;
      if (poll.prSelected) {
        autoSwitchedToPianoRoll = true;
        break;
      }
      await sleep(120);
    }

    await sleep(150);
    const finalState = await win.webContents.executeJavaScript(`
      (() => {
        const prTab = document.querySelector('[data-testid="tab-pianoroll"]');
        const bodyText = document.body.innerText || '';
        const tabText = prTab?.textContent || '';
        const countMatch = tabText.match(/\\d+/);
        return {
          pianoRollSelected: prTab?.getAttribute('aria-selected') === 'true',
          pianoRollTabText: tabText,
          finalNoteCount: countMatch ? Number(countMatch[0]) : 0,
          hasCandidateBadge: bodyText.includes('候補 1') && bodyText.includes('候補 4'),
          hasErrorBanner: bodyText.includes('エラー:'),
          hasStaleUploadNotice: bodyText.includes('uploaded_test.mid を読み込みました'),
        };
      })()
    `);
    const pianoRollGeneratedShot = await capturePng(win, '05_pianoroll_generated_foreground.png');

    // 5B. [R4-HIGH-01] Verify `compass-app://studio` custom-scheme IPC trust & pop-out handlers + authenticated `/hardware_info` 200
    const customSchemeIpcCheck = await win.webContents.executeJavaScript(`
      (async () => {
        const cfg = await window.compassDesktop.requestApiConfig();
        await window.compassDesktop.openPianoRollWindow();
        window.open('compass-app://studio/index.html?tab=pianoroll&popout=1', '_blank');
        const authedRes = await fetch(cfg.apiBaseUrl + '/hardware_info', {
          headers: { 'X-Compass-Handshake': cfg.handshakeToken },
        });
        return {
          apiBaseUrlLen: (cfg.apiBaseUrl || '').length,
          handshakeTokenLen: (cfg.handshakeToken || '').length,
          authedHardwareStatus: authedRes.status,
        };
      })()
    `);
    await sleep(60);

    // 5C. [R4-MED-01] Verify CSS cascade under real Chromium `prefers-reduced-motion: reduce` (zero `!important`)
    win.webContents.debugger.attach('1.3');
    await win.webContents.debugger.sendCommand('Emulation.setEmulatedMedia', {
      features: [{ name: 'prefers-reduced-motion', value: 'reduce' }],
    });
    const reducedMotionCascadeCheck = await win.webContents.executeJavaScript(`
      (() => {
        const mediaMatchesReduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        const probeHost = document.createElement('div');
        const spinEl = document.createElement('div');
        spinEl.className = 'animate-spin motion-reduce:animate-none';
        const genericSpinEl = document.createElement('div');
        genericSpinEl.className = 'animate-spin';
        const durEl = document.createElement('div');
        durEl.className = 'transition-all duration-300 ease-out';
        const btnEl = document.createElement('button');
        btnEl.className = 'transition-colors';
        probeHost.append(spinEl, genericSpinEl, durEl, btnEl);
        document.body.appendChild(probeHost);

        const parseSec = (raw) => {
          if (!raw) return 0;
          const s = String(raw).trim();
          if (s.endsWith('ms')) return parseFloat(s) / 1000;
          if (s.endsWith('s')) return parseFloat(s);
          return parseFloat(s) || 0;
        };
        const spinCs = window.getComputedStyle(spinEl);
        const genSpinCs = window.getComputedStyle(genericSpinEl);
        const durCs = window.getComputedStyle(durEl);
        const btnCs = window.getComputedStyle(btnEl);
        const res = {
          mediaMatchesReduce,
          spinAnimDurationSec: parseSec(spinCs.animationDuration),
          genericSpinAnimDurationSec: parseSec(genSpinCs.animationDuration),
          genericSpinIterCount: genSpinCs.animationIterationCount,
          dur300TransitionSec: parseSec(durCs.transitionDuration),
          btnTransitionSec: parseSec(btnCs.transitionDuration),
        };
        probeHost.remove();
        return res;
      })()
    `);
    await win.webContents.debugger.sendCommand('Emulation.setEmulatedMedia', {
      features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }],
    });
    win.webContents.debugger.detach();

    // 5D. [R6B-MED-05] Verify Piano Roll canvas & bottom Velocity Lane never inflate/clip off-viewport on rack toggle or height shrink
    const measureCanvasBounds = () =>
      win.webContents.executeJavaScript(`
        (() => {
          const c = document.querySelector('[data-testid="pianoroll-canvas"]');
          const r = c ? c.getBoundingClientRect() : null;
          return {
            innerH: window.innerHeight,
            docScrollH: document.documentElement.scrollHeight,
            rectH: r ? Math.round(r.height) : 0,
            rectBottom: r ? Math.round(r.bottom) : 0,
            velocityClippedOffViewport: r ? r.bottom > window.innerHeight + 2 : true,
          };
        })()
      `);
    const prRestoredBounds = await measureCanvasBounds();
    const compassPianoRollUiCheck = await win.webContents.executeJavaScript(`
      (() => {
        const btns = Array.from(document.querySelectorAll('button'));
        const hasRedundantSelectBtn = btns.some((b) => (b.textContent || '').trim() === '選択');
        const hasRedundantPanBtn = btns.some((b) => (b.textContent || '').trim() === 'パン');
        const hasPencilToggle = Boolean(document.querySelector('button[aria-label="ペン入力"]'));
        const hasRangeSelectBtn = Boolean(document.querySelector('button[aria-label="対象小節内のノートを選択"]'));
        const hasTripletBtn = btns.some((b) => (b.textContent || '').trim() === '3連');
        const sampleBtn = document.querySelector('[data-testid="load-sample-arrangement-button"]');
        const hasLabeledSampleBtn = Boolean(sampleBtn && (sampleBtn.textContent || '').includes('サンプル'));
        const hasTelemetryBar = Boolean(document.querySelector('[data-testid="pianoroll-telemetry-bar"]'));
        const canvas = document.querySelector('[data-testid="pianoroll-canvas"]');
        const canvasParentBg = canvas && canvas.parentElement
          ? window.getComputedStyle(canvas.parentElement).backgroundColor
          : '';
        let velocityLaneTranslucent = false;
        let playheadRenderedInGrid = false;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          const dpr = canvas.width / Math.max(1, canvas.clientWidth);
          // Sample empty right-hand Velocity Lane background pixel (far from note stems)
          const velSampleX = Math.min(canvas.width - 10, Math.round((canvas.clientWidth - 24) * dpr));
          const velSampleY = Math.min(canvas.height - 12, Math.round((canvas.clientHeight - 24) * dpr));
          const velPx = ctx.getImageData(velSampleX, velSampleY, 1, 1).data;
          velocityLaneTranslucent = velPx[3] > 0 && velPx[3] < 210;

          // Scan horizontal band at y = 96px inside Note Grid around x = 68..80px (KEYBOARD_WIDTH=72 playhead at tick 0)
          // or anywhere in grid if playhead moved, looking for bright #F4F4F5 playhead stroke
          const scanY = Math.round(96 * dpr);
          const rowData = ctx.getImageData(Math.round(70 * dpr), scanY, Math.max(8, Math.round(600 * dpr)), 1).data;
          for (let i = 0; i < rowData.length; i += 4) {
            if (rowData[i] >= 220 && rowData[i + 1] >= 220 && rowData[i + 2] >= 220 && rowData[i + 3] >= 200) {
              playheadRenderedInGrid = true;
              break;
            }
          }
        }
        const rackBtn = document.querySelector('[data-testid="toggle-track-rack-button"]');
        const toolbarEl = rackBtn ? rackBtn.closest('div.border-b') : null;
        const toolbarHeight1440 = toolbarEl ? Math.round(toolbarEl.getBoundingClientRect().height) : 999;
        if (rackBtn) rackBtn.click();
        return {
          hasRedundantSelectBtn,
          hasRedundantPanBtn,
          hasPencilToggle,
          hasRangeSelectBtn,
          hasTripletBtn,
          hasLabeledSampleBtn,
          hasTelemetryBar,
          toolbarHeight1440,
          canvasContainerTransparent: canvasParentBg === 'rgba(0, 0, 0, 0)',
          velocityLaneTranslucent,
          playheadRenderedInGrid,
          rackToggleFound: Boolean(rackBtn),
        };
      })()
    `);
    await sleep(120);
    const prAfterRackOffBounds = await measureCanvasBounds();
    win.setContentSize(1440, 650);
    await sleep(160);
    const prShrunk650Bounds = await measureCanvasBounds();
    win.setContentSize(1440, 900);
    await sleep(120);
    const velocityLaneBoundsCheck = {
      prRestoredBounds,
      prAfterRackOffBounds,
      prShrunk650Bounds,
    };

    // 5D2. [R7-HIGH-01] [R7-MED-01] [R7-MED-04] Verify 375x667 & 667x375 with BOTH Track Rack open AND 4-candidate switcher strip rendered
    await win.webContents.executeJavaScript(`
      (() => {
        const rackBtn = document.querySelector('[data-testid="toggle-track-rack-button"]');
        if (rackBtn && rackBtn.getAttribute('aria-pressed') !== 'true') {
          rackBtn.click();
        }
      })()
    `);
    await sleep(80);
    await win.webContents.executeJavaScript(`
      (() => {
        const addBtn = document.querySelector('button[aria-label="トラックを追加"]');
        if (addBtn) addBtn.click();
      })()
    `);
    await sleep(80);

    const measureCrowdedMobileCanvasAndTooltips = () =>
      win.webContents.executeJavaScript(`
        (() => {
          const rackBtn = document.querySelector('[data-testid="toggle-track-rack-button"]');
          if (rackBtn && rackBtn.getAttribute('aria-pressed') !== 'true') {
            rackBtn.click();
          }
          const aside = document.querySelector('[aria-label="トラックラック"]');
          const canvas = document.querySelector('[data-testid="pianoroll-canvas"]');
          const cRect = canvas ? canvas.getBoundingClientRect() : null;
          const dpr = Math.min(window.devicePixelRatio || 1, 2);
          const backingCssH = canvas ? canvas.height / dpr : 0;
          const aspectDeltaPx = cRect ? Math.abs(backingCssH - cRect.height) : 999;
          const has4Candidates =
            (document.body.innerText || '').includes('候補 1') &&
            (document.body.innerText || '').includes('候補 4');

          // Check vertical tooltip flip inside scrollable <aside> (Track 1 Trash2 near top vs bottom) + Escape dismissal
          let firstTrashFlippedFromTop = false;
          let tooltipEscapeDismissed = false;
          if (aside) {
            aside.scrollTop = 0;
            const trashBtns = Array.from(aside.querySelectorAll('button[data-tooltip="トラックを削除"]'));
            const firstTrash = trashBtns[0] || null;
            if (firstTrash) {
              firstTrash.dispatchEvent(new PointerEvent('pointerover', { bubbles: true }));
              const asideRect = aside.getBoundingClientRect();
              const trRect = firstTrash.getBoundingClientRect();
              const posAttr = firstTrash.getAttribute('data-tooltip-pos');
              const tipTop = posAttr === 'top' ? trRect.top - 32 : trRect.bottom + 6;
              firstTrashFlippedFromTop = tipTop >= asideRect.top - 2;
              window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
              tooltipEscapeDismissed = firstTrash.getAttribute('data-tooltip-dismissed') === 'true';
              firstTrash.dispatchEvent(new PointerEvent('pointerout', { bubbles: true }));
            }
          }

          return {
            innerW: window.innerWidth,
            innerH: window.innerHeight,
            docScrollW: document.documentElement.scrollWidth,
            hasHorizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
            rackOpen: Boolean(aside && aside.getBoundingClientRect().height >= 80),
            has4Candidates,
            canvasHeight: cRect ? Math.round(cRect.height) : 0,
            backingCssH: Math.round(backingCssH),
            aspectDeltaPx: Math.round(aspectDeltaPx * 10) / 10,
            firstTrashFlippedFromTop,
            tooltipEscapeDismissed,
          };
        })()
      `);

    win.setMinimumSize(320, 320);
    win.setContentSize(375, 667);
    await sleep(160);
    const crowdedPortrait375x667 = await measureCrowdedMobileCanvasAndTooltips();
    win.setContentSize(667, 375);
    await sleep(160);
    const crowdedLandscape667x375 = await measureCrowdedMobileCanvasAndTooltips();
    win.setMinimumSize(375, 620);
    win.setContentSize(1440, 900);
    await sleep(120);
    const tooltipHitTestAndBridgeCheck = await win.webContents.executeJavaScript(`
      (() => {
        const aside = document.querySelector('[aria-label="トラックラック"]');
        const canvas = document.querySelector('[data-testid="pianoroll-canvas"]');
        const pencilBtn = document.querySelector('button[aria-label="ペン入力"]');
        const rangeBtn = document.querySelector('button[aria-label="3連符グリッド切替"]');
        if (!aside || !canvas || !pencilBtn || !rangeBtn) {
          return { ok: false, reason: 'missing-elements' };
        }
        aside.scrollTop = 0;
        const trashBtns = Array.from(aside.querySelectorAll('button[data-tooltip="トラックを削除"]'));
        const soloBtns = Array.from(aside.querySelectorAll('button[data-tooltip="ソロ"]'));
        const secondTrash = trashBtns[1] || trashBtns[0];
        const targetSolo = soloBtns[1] || soloBtns[0];
        if (!secondTrash || !targetSolo) {
          return { ok: false, reason: 'missing-track-buttons' };
        }

        secondTrash.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, pointerType: 'mouse' }));

        // Simulate bridging upward across Track 2 card gap, then hit-testing Row-1 Solo button & generated ::after
        const trRect = secondTrash.getBoundingClientRect();
        secondTrash.dispatchEvent(
          new PointerEvent('pointerout', {
            bubbles: true,
            pointerType: 'mouse',
            relatedTarget: aside,
            clientX: trRect.left + trRect.width * 0.5,
            clientY: trRect.top - 4,
          }),
        );
        const afterStyle = window.getComputedStyle(secondTrash, '::after');
        const beforeStyle = window.getComputedStyle(secondTrash, '::before');
        const tooltipPointerEventsNone =
          secondTrash.getAttribute('data-tooltip-bridge') === 'true' &&
          afterStyle.pointerEvents === 'none';
        const noBeforeHoverTrap = beforeStyle.content === 'none' || beforeStyle.pointerEvents === 'none';
        const soloRect = targetSolo.getBoundingClientRect();
        const hitAtSoloBottom = document.elementFromPoint(
          soloRect.left + soloRect.width * 0.5,
          soloRect.bottom - 4,
        );
        const soloHitUnoccluded = Boolean(hitAtSoloBottom && targetSolo.contains(hitAtSoloBottom));

        // Moving onto Solo button clears Trash2 bridge immediately
        targetSolo.dispatchEvent(
          new PointerEvent('pointerover', {
            bubbles: true,
            pointerType: 'mouse',
            clientX: soloRect.left + soloRect.width * 0.5,
            clientY: soloRect.bottom - 4,
          }),
        );
        const trashBridgeClearedOnNeighbor = secondTrash.getAttribute('data-tooltip-bridge') === null;
        targetSolo.dispatchEvent(new PointerEvent('pointerout', { bubbles: true, pointerType: 'mouse' }));

        // Verify toolbar button -> canvas tooltip corridor bridge (WCAG 1.4.13 Hoverable + unoccluded canvas ruler)
        const pRect = pencilBtn.getBoundingClientRect();
        const tipProbeX = pRect.left + pRect.width * 0.5;
        const tipProbeY = pRect.bottom + 12;
        pencilBtn.dispatchEvent(
          new PointerEvent('pointerover', {
            bubbles: true,
            pointerType: 'mouse',
            clientX: tipProbeX,
            clientY: pRect.top + pRect.height * 0.5,
          }),
        );
        pencilBtn.dispatchEvent(
          new PointerEvent('pointerout', {
            bubbles: true,
            pointerType: 'mouse',
            relatedTarget: canvas,
            clientX: tipProbeX,
            clientY: tipProbeY,
          }),
        );
        const bridgeActiveOverCanvas = pencilBtn.getAttribute('data-tooltip-bridge') === 'true';
        const hitUnderToolbarTooltip = document.elementFromPoint(tipProbeX, tipProbeY);
        const cRect = canvas.getBoundingClientRect();
        const hitAtCanvasRuler = document.elementFromPoint(cRect.left + 140, cRect.top + 8);
        const canvasRulerUnoccluded =
          Boolean(hitUnderToolbarTooltip && !pencilBtn.contains(hitUnderToolbarTooltip)) &&
          hitAtCanvasRuler === canvas;

        // Moving onto neighbor toolbar button clears bridge immediately
        rangeBtn.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, pointerType: 'mouse' }));
        const bridgeClearedOnToolbarNeighbor = pencilBtn.getAttribute('data-tooltip-bridge') === null;
        rangeBtn.dispatchEvent(new PointerEvent('pointerout', { bubbles: true, pointerType: 'mouse' }));

        // [R9-TOOLTIP-01]: Verify Main Split Workspace stacking containment (z-index 5 < 50),
        // focus-restore -> hover mutual exclusion, and [inert] button[data-tooltip]::after { display: none }
        const splitWorkspace = aside.parentElement;
        const splitWorkspaceZIndex5 = Boolean(
          splitWorkspace && window.getComputedStyle(splitWorkspace).zIndex === '5',
        );
        const firstTrackBtn = aside.querySelector('button[data-track-select-id]');
        let focusRestoreDismissedOnHover = false;
        if (firstTrackBtn instanceof HTMLButtonElement) {
          window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
          firstTrackBtn.focus();
          targetSolo.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, pointerType: 'mouse' }));
          focusRestoreDismissedOnHover =
            !firstTrackBtn.matches(':focus-visible') ||
            firstTrackBtn.getAttribute('data-tooltip-dismissed') === 'true';
          targetSolo.dispatchEvent(new PointerEvent('pointerout', { bubbles: true, pointerType: 'mouse' }));
          firstTrackBtn.blur();
        }
        targetSolo.setAttribute('data-tooltip-bridge', 'true');
        aside.setAttribute('inert', '');
        const inertSuppressesTooltipAfter =
          window.getComputedStyle(targetSolo, '::after').display === 'none';
        aside.removeAttribute('inert');
        targetSolo.removeAttribute('data-tooltip-bridge');

        return {
          ok: true,
          tooltipPointerEventsNone,
          noBeforeHoverTrap,
          soloHitUnoccluded,
          trashBridgeClearedOnNeighbor,
          bridgeActiveOverCanvas,
          canvasRulerUnoccluded,
          bridgeClearedOnToolbarNeighbor,
          splitWorkspaceZIndex5,
          focusRestoreDismissedOnHover,
          inertSuppressesTooltipAfter,
        };
      })()
    `);
    const mobileCrowdedCanvasCheck = {
      crowdedPortrait375x667,
      crowdedLandscape667x375,
      tooltipHitTestAndBridgeCheck,
    };

    // 5E. [FC-MED-04] Verify Pop-Out Piano Roll (`?tab=pianoroll&popout=1`) hides self-popout button & duplicate tablist
    await win.loadURL(COMPASS_APP_ENTRY_URL + '?tab=pianoroll&popout=1');
    for (let i = 0; i < 40; i += 1) {
      const ready = await win.webContents.executeJavaScript(
        `Boolean(document.querySelector('[data-testid="popout-pianoroll-badge"]'))`,
      );
      if (ready) break;
      await sleep(100);
    }
    win.setContentSize(1280, 800);
    await sleep(140);
    const popoutToolbarHeight1280 = await win.webContents.executeJavaScript(`
      (() => {
        const rackBtn = document.querySelector('[data-testid="toggle-track-rack-button"]');
        const toolbarEl = rackBtn ? rackBtn.closest('div.border-b') : null;
        return toolbarEl ? Math.round(toolbarEl.getBoundingClientRect().height) : 999;
      })()
    `);
    win.setContentSize(1360, 880);
    await sleep(140);
    const mainDefaultToolbarHeight1360 = await win.webContents.executeJavaScript(`
      (() => {
        const rackBtn = document.querySelector('[data-testid="toggle-track-rack-button"]');
        const toolbarEl = rackBtn ? rackBtn.closest('div.border-b') : null;
        return toolbarEl ? Math.round(toolbarEl.getBoundingClientRect().height) : 999;
      })()
    `);
    win.setContentSize(375, 812);
    await sleep(140);
    const mobileTooltipBoundsCheck = await win.webContents.executeJavaScript(`
      (() => {
        const docW = document.documentElement.clientWidth;
        const docH = document.documentElement.clientHeight;
        const clipped = [];
        for (const btn of Array.from(document.querySelectorAll('button[data-tooltip]:not(:disabled), button[data-tooltip][data-tooltip-when-disabled="true"]:disabled'))) {
          const cs = window.getComputedStyle(btn);
          if (cs.display === 'none' || cs.visibility === 'hidden') continue;
          const r = btn.getBoundingClientRect();
          if (r.width <= 0 || r.height <= 0) continue;
          const evtTarget = btn.disabled && btn.parentElement ? btn.parentElement : btn;
          const cx = r.left + r.width * 0.5;
          const cy = r.top + r.height * 0.5;
          evtTarget.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, clientX: cx, clientY: cy, buttons: 0 }));
          evtTarget.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: cx, clientY: cy, buttons: 0 }));
          const align = btn.getAttribute('data-tooltip-align') || 'center';
          const pos = btn.getAttribute('data-tooltip-pos') || 'bottom';
          const tip = (btn.getAttribute('data-tooltip') || '').trim();
          let estPx = 22;
          for (let i = 0; i < tip.length; i += 1) {
            estPx += tip.charCodeAt(i) > 0x7f ? 12.5 : 7.5;
          }
          const inAside = Boolean(btn.closest('aside'));
          const maxCap = inAside
            ? Math.min(224, Math.max(48, docW - 32))
            : Math.min(340, Math.max(48, docW - 16));
          const w = Math.min(maxCap, Math.ceil(estPx));
          const left = align === 'start' ? r.left : align === 'end' ? r.right - w : r.left + r.width * 0.5 - w * 0.5;
          const right = left + w;
          const top = pos === 'top' ? r.top - 30 : r.bottom + 6;
          const bottom = top + 24;
          if (left < -1 || right > docW + 1 || top < -1 || bottom > docH + 1) {
            clipped.push({ tip, align, pos, left: Math.round(left), right: Math.round(right), top: Math.round(top), bottom: Math.round(bottom), docW, docH });
          }
          evtTarget.dispatchEvent(new PointerEvent('pointerout', { bubbles: true, clientX: -10, clientY: -10, buttons: 0 }));
          evtTarget.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: -10, clientY: -10, buttons: 0 }));
        }
        return { docW, docH, clipped };
      })()
    `);
    win.setContentSize(1440, 900);
    await sleep(100);
    const popoutWindowUiCheck = await win.webContents.executeJavaScript(`
      (() => {
        const hasPopoutBadge = Boolean(document.querySelector('[data-testid="popout-pianoroll-badge"]'));
        const hasTablist = Boolean(document.querySelector('[role="tablist"]'));
        const hasSelfPopoutButton = Boolean(document.querySelector('button[aria-label="別ウィンドウでピアノロールを開く"]'));
        return { hasPopoutBadge, hasTablist, hasSelfPopoutButton };
      })()
    `);
    popoutWindowUiCheck.popoutToolbarHeight1280 = popoutToolbarHeight1280;
    popoutWindowUiCheck.mainDefaultToolbarHeight1360 = mainDefaultToolbarHeight1360;
    popoutWindowUiCheck.mobileTooltipClippedCount = mobileTooltipBoundsCheck.clipped.length;
    popoutWindowUiCheck.mobileTooltipClipped = mobileTooltipBoundsCheck.clipped;

    // 5F. [Round-10 Critic] Verify disabled tooltip full contrast + pointermove lifecycle ([R10-HIGH-01]),
    // SYNC_ACTIVE_CONTROLLER pop-out standby card ([R10-HIGH-02]), ControlPanel preview/CoT tooltips ([R10-MED-01]),
    // Secondary Strip tooltip z-index elevation ([R10-MED-02]), and programmatic tab focus retention ([R10-MED-03])
    const syncControllerIpcValid =
      isValidWorkspaceSyncPayload({
        messageId: 'e2e_sync_ctrl_1',
        type: 'SYNC_ACTIVE_CONTROLLER',
        controllerId: 'cubase-midi-remote',
        senderId: 'e2e_peer',
      }) === true &&
      isValidWorkspaceSyncPayload({
        messageId: 'e2e_sync_ctrl_bad',
        type: 'SYNC_ACTIVE_CONTROLLER',
        controllerId: 'unknown-daw',
        senderId: 'e2e_peer',
      }) === false;

    const popoutRound10Check = await win.webContents.executeJavaScript(`
      (async () => {
        const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
        // 1. Trigger SYNC_ACTIVE_CONTROLLER -> Pop-Out Standby Card mounts with 0 form inputs
        const bc = new BroadcastChannel('compass_ai_workspace_v1');
        bc.postMessage({
          messageId: 'e2e_bc_ctrl_cubase',
          type: 'SYNC_ACTIVE_CONTROLLER',
          controllerId: 'cubase-midi-remote',
          senderId: 'e2e_main_sender',
        });
        for (let i = 0; i < 20; i += 1) {
          if (document.querySelector('[data-testid="popout-external-daw-standby"]')) break;
          await waitMs(40);
        }
        const standby = document.querySelector('[data-testid="popout-external-daw-standby"]');
        const standbyMounted = Boolean(standby);
        const standbyZeroFormControls = standby ? standby.querySelectorAll('select, input').length === 0 : false;
        const standbyLaunchShortcut =
          standby?.querySelector('[data-testid="launch-compass-ai-button"]')?.getAttribute('aria-keyshortcuts') === 'Shift+G';
        const switchDemoBtn = document.querySelector('[data-testid="popout-switch-to-demo-button"]');
        if (switchDemoBtn instanceof HTMLButtonElement) {
          switchDemoBtn.focus();
          switchDemoBtn.click();
        }
        for (let i = 0; i < 20; i += 1) {
          if (document.querySelector('[data-testid="pianoroll-canvas"]')) break;
          await waitMs(40);
        }
        const switchedBackToCanvas =
          !document.querySelector('[data-testid="popout-external-daw-standby"]') &&
          Boolean(document.querySelector('[data-testid="pianoroll-canvas"]'));
        const focusRetainedAfterSwitch = Boolean(
          document.activeElement && document.activeElement !== document.body,
        );
        bc.close();

        // 2. Upload 1-track MIDI to mount [data-tooltip-strip] and verify z-index 30 elevation on hover & 10 on Escape
        const trackEvents = [
          0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
          0x00, 0xc0, 0x00,
          0x00, 0x90, 60, 100, 0x83, 0x60, 0x80, 60, 0,
          0x00, 0xff, 0x2f, 0x00
        ];
        const trackLen = trackEvents.length;
        const raw = new Uint8Array([
          0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
          0x00, 0x00, 0x00, 0x01, 0x01, 0xe0,
          0x4d, 0x54, 0x72, 0x6b,
          (trackLen >>> 24) & 0xff,
          (trackLen >>> 16) & 0xff,
          (trackLen >>> 8) & 0xff,
          trackLen & 0xff,
          ...trackEvents
        ]);
        const file = new File([raw], 'r10_strip.mid', { type: 'audio/midi' });
        const dt = new DataTransfer();
        dt.items.add(file);
        const fInput = document.querySelector('[data-testid="midi-file-input"]');
        fInput.files = dt.files;
        fInput.dispatchEvent(new Event('change', { bubbles: true }));
        for (let i = 0; i < 20; i += 1) {
          if (document.querySelector('[data-tooltip-strip]')) break;
          await waitMs(40);
        }
        const strip = document.querySelector('[data-tooltip-strip]');
        const closeNoticeBtn = strip ? strip.querySelector('button[data-tooltip="通知を閉じる"]') : null;
        let stripZBefore = '';
        let stripZActive = '';
        let stripZAfterEscape = '';
        if (strip && closeNoticeBtn instanceof HTMLButtonElement) {
          stripZBefore = window.getComputedStyle(strip).zIndex;
          const nr = closeNoticeBtn.getBoundingClientRect();
          closeNoticeBtn.dispatchEvent(
            new PointerEvent('pointerover', {
              bubbles: true,
              pointerType: 'mouse',
              clientX: nr.left + nr.width * 0.5,
              clientY: nr.top + nr.height * 0.5,
              buttons: 0,
            }),
          );
          stripZActive = window.getComputedStyle(strip).zIndex;
          window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
          stripZAfterEscape = window.getComputedStyle(strip).zIndex;
          closeNoticeBtn.dispatchEvent(
            new PointerEvent('pointerout', { bubbles: true, pointerType: 'mouse', clientX: 0, clientY: 0, buttons: 0 }),
          );
        }

        // 3. Fill up to 64 tracks -> verify disabled + button opacity === '1', ::after opacity === '1' & zIndex === '60',
        // pointermove hover -> bridge -> Escape -> exit + re-hover recovery
        const addBtn = document.querySelector('button[aria-label="トラックを追加"]');
        let guard = 0;
        while (addBtn instanceof HTMLButtonElement && !addBtn.disabled && guard < 70) {
          addBtn.click();
          guard += 1;
        }
        await waitMs(40);
        let disabledBtnFullOpacity = false;
        let disabledAfterFullOpacity = false;
        let disabledAfterZIndex60 = false;
        let disabledBridgeActivated = false;
        let disabledEscapeDismissed = false;
        let disabledRehoverRecovered = false;
        if (addBtn instanceof HTMLButtonElement && addBtn.disabled && addBtn.parentElement) {
          const parentDiv = addBtn.parentElement;
          const ar = addBtn.getBoundingClientRect();
          const cx = ar.left + ar.width * 0.5;
          const cy = ar.top + ar.height * 0.5;
          disabledBtnFullOpacity = window.getComputedStyle(addBtn).opacity === '1';
          parentDiv.dispatchEvent(
            new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse', clientX: cx, clientY: cy, buttons: 0 }),
          );
          const afterCs = window.getComputedStyle(addBtn, '::after');
          disabledAfterFullOpacity =
            addBtn.getAttribute('data-tooltip-hovered') === 'true' &&
            afterCs.opacity === '1' &&
            afterCs.content !== 'none';
          disabledAfterZIndex60 = afterCs.zIndex === '60';

          // Move across 6px gap below + button onto tooltip corridor
          parentDiv.dispatchEvent(
            new PointerEvent('pointermove', {
              bubbles: true,
              pointerType: 'mouse',
              clientX: cx,
              clientY: ar.bottom + 4,
              buttons: 0,
            }),
          );
          disabledBridgeActivated =
            addBtn.getAttribute('data-tooltip-bridge') === 'true' &&
            window.getComputedStyle(addBtn, '::after').content !== 'none';

          // Escape dismisses bridged disabled tooltip
          window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
          disabledEscapeDismissed =
            addBtn.getAttribute('data-tooltip-dismissed') === 'true' &&
            window.getComputedStyle(addBtn, '::after').content === 'none';

          // Re-enter button, exit into parent div left of button, then re-hover -> recovers cleanly!
          parentDiv.dispatchEvent(
            new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse', clientX: cx, clientY: cy, buttons: 0 }),
          );
          parentDiv.dispatchEvent(
            new PointerEvent('pointermove', {
              bubbles: true,
              pointerType: 'mouse',
              clientX: Math.max(4, ar.left - 36),
              clientY: cy,
              buttons: 0,
            }),
          );
          parentDiv.dispatchEvent(
            new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse', clientX: cx, clientY: cy, buttons: 0 }),
          );
          disabledRehoverRecovered =
            addBtn.getAttribute('data-tooltip-dismissed') === null &&
            addBtn.getAttribute('data-tooltip-hovered') === 'true' &&
            window.getComputedStyle(addBtn, '::after').content !== 'none';
          parentDiv.dispatchEvent(
            new PointerEvent('pointermove', {
              bubbles: true,
              pointerType: 'mouse',
              clientX: Math.max(4, ar.left - 36),
              clientY: cy,
              buttons: 0,
            }),
          );
        }

        return {
          standbyMounted,
          standbyZeroFormControls,
          standbyLaunchShortcut,
          switchedBackToCanvas,
          focusRetainedAfterSwitch,
          stripZBefore,
          stripZActive,
          stripZAfterEscape,
          disabledBtnFullOpacity,
          disabledAfterFullOpacity,
          disabledAfterZIndex60,
          disabledBridgeActivated,
          disabledEscapeDismissed,
          disabledRehoverRecovered,
        };
      })()
    `);

    await win.loadURL(COMPASS_APP_ENTRY_URL);
    for (let i = 0; i < 60; i += 1) {
      const ready = await win.webContents.executeJavaScript(
        `Boolean(document.querySelector('[data-testid="generate-button"]'))`,
      );
      if (ready) break;
      await sleep(100);
    }
    const mainRound10Check = await win.webContents.executeJavaScript(`
      (async () => {
        const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
        const previewBtn = document.querySelector('button[aria-label="サウンド確認"]');
        const cotBtn = Array.from(document.querySelectorAll('button')).find((b) =>
          (b.textContent || '').includes('構造推論 自動省略'),
        );
        const previewTooltipIdle = previewBtn ? previewBtn.getAttribute('data-tooltip') : '';
        const cotDisabledInMeta = Boolean(
          cotBtn &&
            cotBtn.disabled &&
            window.getComputedStyle(cotBtn).opacity === '1' &&
            cotBtn.getAttribute('data-tooltip') === '白紙からの生成 (meta) では構造推論は自動省略されます',
        );

        const dawSelect = document.querySelector('#compass-daw-select');
        if (dawSelect instanceof HTMLSelectElement) {
          const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
          setter?.call(dawSelect, 'cubase-midi-remote');
          dawSelect.dispatchEvent(new Event('change', { bubbles: true }));
        }
        await waitMs(80);
        const popoutHeaderBtn = document.querySelector('button[aria-label="別ウィンドウでピアノロールを開く"]');
        const previewDisabledOnExternal = Boolean(
          previewBtn &&
            previewBtn.disabled &&
            window.getComputedStyle(previewBtn).opacity === '1' &&
            previewBtn.getAttribute('data-tooltip') === '外部DAW接続中はホストDAW側で再生を操作します',
        );
        const popoutHeaderFullOpacityWhenDisabled = Boolean(
          popoutHeaderBtn &&
            popoutHeaderBtn.disabled &&
            window.getComputedStyle(popoutHeaderBtn).opacity === '1',
        );

        document.querySelector('[data-testid="tab-pianoroll"]')?.click();
        await waitMs(60);
        const extLaunchBtn = document.querySelector('[data-testid="launch-compass-ai-button"]');
        const extLaunchShortcut = extLaunchBtn ? extLaunchBtn.getAttribute('aria-keyshortcuts') === 'Shift+G' : false;
        if (extLaunchBtn instanceof HTMLButtonElement) {
          extLaunchBtn.focus();
          extLaunchBtn.click();
        }
        await waitMs(80);
        const activeIdAfterProgrammaticSwitch = document.activeElement ? document.activeElement.id : '';

        return {
          previewTooltipIdle,
          cotDisabledInMeta,
          previewDisabledOnExternal,
          popoutHeaderFullOpacityWhenDisabled,
          extLaunchShortcut,
          activeIdAfterProgrammaticSwitch,
        };
      })()
    `);
    const round10UiAndSyncCheck = {
      syncControllerIpcValid,
      popoutRound10Check,
      mainRound10Check,
    };

    // 5c. Round-11 Desktop Zero-Scroll Parameter Console Fit (`1440x900`, `1360x880`, `1280x800`) + Self-Unmount Focus & Reload Sync
    await win.webContents.executeJavaScript(`
      (() => {
        const dawSelect = document.querySelector('#compass-daw-select');
        if (dawSelect instanceof HTMLSelectElement) {
          const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
          setter?.call(dawSelect, 'demo-pianoroll');
          dawSelect.dispatchEvent(new Event('change', { bubbles: true }));
        }
        document.querySelector('[data-testid="tab-compass-ai"]')?.click();
      })()
    `);
    await sleep(100);

    const measureControlPanelVerticalFit = async (w, h, expandSft = false) => {
      win.setContentSize(w, h);
      await sleep(160);
      return win.webContents.executeJavaScript(`
        (async () => {
          const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
          window.scrollTo(0, 0);
          const sftBtn = Array.from(document.querySelectorAll('button')).find((b) =>
            (b.textContent || '').includes('音楽構造・ジャンル・タスクの詳細指定'),
          );
          if (sftBtn instanceof HTMLButtonElement) {
            const expanded = sftBtn.getAttribute('aria-expanded') === 'true';
            if (expanded !== ${expandSft ? 'true' : 'false'}) {
              sftBtn.click();
              await waitMs(80);
            }
          }
          const innerW = window.innerWidth;
          const innerH = window.innerHeight;
          const scrollH = document.documentElement.scrollHeight;
          const scrollW = document.documentElement.scrollWidth;
          const genreTray = document.querySelector('[data-testid="genre-chips-scroll-tray"]');
          const genreTrayHeight = genreTray ? Math.round(genreTray.getBoundingClientRect().height) : 0;
          const genreTrayZeroNegMargin = genreTray ? !genreTray.className.includes('-m-1') : true;
          return {
            innerW,
            innerH,
            scrollH,
            scrollW,
            genreTrayHeight,
            genreTrayZeroNegMargin,
            fitsWithoutVerticalScroll: scrollH <= innerH,
            fitsWithoutHorizontalScroll: scrollW <= innerW,
          };
        })()
      `);
    };

    const fit1440Default = await measureControlPanelVerticalFit(1440, 900, false);
    const fit1440Expanded = await measureControlPanelVerticalFit(1440, 900, true);
    const fit1360Default = await measureControlPanelVerticalFit(1360, 880, false);
    const fit1360Expanded = await measureControlPanelVerticalFit(1360, 880, true);
    const desktopControlExpandedShot = await capturePng(
      win,
      '02b_control_panel_expanded_desktop.png',
    );
    await win.webContents.executeJavaScript(`
      (() => {
        const modelSel = document.querySelector('#compass-model-select');
        if (modelSel instanceof HTMLSelectElement) {
          const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
          setter?.call(modelSel, 'MORTM4.5D-80M');
          modelSel.dispatchEvent(new Event('change', { bubbles: true }));
        }
      })()
    `);
    await sleep(80);
    const fit1360ExpandedFoundation = await measureControlPanelVerticalFit(1360, 880, true);
    await win.webContents.executeJavaScript(`
      (() => {
        const modelSel = document.querySelector('#compass-model-select');
        if (modelSel instanceof HTMLSelectElement) {
          const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
          setter?.call(modelSel, 'MORTM4.5D-80M-SFT-Gen');
          modelSel.dispatchEvent(new Event('change', { bubbles: true }));
        }
      })()
    `);
    await sleep(80);
    const fit1280Default = await measureControlPanelVerticalFit(1280, 800, false);
    const fit1280Expanded = await measureControlPanelVerticalFit(1280, 800, true);
    await win.webContents.executeJavaScript(`
      (() => {
        const modelSel = document.querySelector('#compass-model-select');
        if (modelSel instanceof HTMLSelectElement) {
          const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
          setter?.call(modelSel, 'MORTM4.5D-80M');
          modelSel.dispatchEvent(new Event('change', { bubbles: true }));
        }
      })()
    `);
    await sleep(80);
    const fit1280ExpandedFoundation = await measureControlPanelVerticalFit(1280, 800, true);
    await win.webContents.executeJavaScript(`
      (() => {
        const modelSel = document.querySelector('#compass-model-select');
        if (modelSel instanceof HTMLSelectElement) {
          const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
          setter?.call(modelSel, 'MORTM4.5D-80M-SFT-Gen');
          modelSel.dispatchEvent(new Event('change', { bubbles: true }));
        }
      })()
    `);
    await sleep(80);
    win.setContentSize(1440, 900);
    await sleep(120);

    const round14HeaderTablistCheck = await win.webContents.executeJavaScript(`
      (async () => {
        const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
        const header = document.querySelector('header');
        const navRoleTablistAbsent = header ? header.querySelector('nav[role="tablist"]') === null : false;
        const tablist = header ? header.querySelector('div[role="tablist"]') : null;
        const divRoleTablistPresent = Boolean(tablist);
        const focusables = header
          ? Array.from(header.querySelectorAll('button:not(:disabled), a[href], select:not(:disabled), input:not(:disabled)'))
              .filter((el) => el.getClientRects().length > 0 && el.getAttribute('tabindex') !== '-1')
          : [];
        let headerDomMatchesVisualX = focusables.length >= 2;
        for (let i = 1; i < focusables.length; i += 1) {
          if (focusables[i].getBoundingClientRect().left < focusables[i - 1].getBoundingClientRect().left - 1) {
            headerDomMatchesVisualX = false;
            break;
          }
        }
        const aiTab = header ? header.querySelector('[data-testid="tab-compass-ai"]') : null;
        const prTab = header ? header.querySelector('[data-testid="tab-pianoroll"]') : null;
        let tablistArrowSwitchOk = false;
        let rovingTabIndexOk = false;
        if (tablist && aiTab instanceof HTMLButtonElement && prTab instanceof HTMLButtonElement) {
          const initialRoving = aiTab.tabIndex === 0 && prTab.tabIndex === -1;
          aiTab.focus();
          tablist.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', code: 'ArrowRight', bubbles: true, cancelable: true }));
          await waitMs(40);
          const movedRight = document.activeElement === prTab && prTab.getAttribute('aria-selected') === 'true';
          const rightRoving = aiTab.tabIndex === -1 && prTab.tabIndex === 0;
          tablist.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', code: 'ArrowLeft', bubbles: true, cancelable: true }));
          await waitMs(40);
          const movedLeft = document.activeElement === aiTab && aiTab.getAttribute('aria-selected') === 'true';
          const leftRoving = aiTab.tabIndex === 0 && prTab.tabIndex === -1;
          tablistArrowSwitchOk = movedRight && movedLeft;
          rovingTabIndexOk = initialRoving && rightRoving && leftRoving;
        }
        const sftBtn = Array.from(document.querySelectorAll('button')).find((b) =>
          (b.textContent || '').includes('音楽構造・ジャンル・タスクの詳細指定'),
        );
        if (sftBtn instanceof HTMLButtonElement && sftBtn.getAttribute('aria-expanded') === 'true') {
          sftBtn.click();
          await waitMs(50);
        }
        const sftPanel = document.getElementById('sft-structure-options-panel');
        const sftAriaControlsMountedWhenCollapsed = Boolean(sftPanel && sftPanel.hidden === true);
        return {
          navRoleTablistAbsent,
          divRoleTablistPresent,
          headerDomMatchesVisualX,
          tablistArrowSwitchOk,
          rovingTabIndexOk,
          sftAriaControlsMountedWhenCollapsed,
        };
      })()
    `);

    // `[R12-MED-01]`, `[R12-LOW-01]` & `[R15-HIGH-01]`: Verify zero ellipsis/clipping on ALL disabled-reason tooltips + overflow:visible on every button[data-tooltip]
    const disabledTooltipNonTruncationCheck = await win.webContents.executeJavaScript(`
      (async () => {
        const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
        const checkBtnTooltipFits = (btn) => {
          if (!(btn instanceof HTMLButtonElement)) return { ok: false, reason: 'missing' };
          const btnSt = window.getComputedStyle(btn);
          const overflowVisible = btnSt.overflowX === 'visible' && btnSt.overflowY === 'visible';
          const r = btn.getBoundingClientRect();
          const cx = r.left + r.width * 0.5;
          const cy = r.top + r.height * 0.5;
          document.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: cx, clientY: cy, buttons: 0 }));
          const st = window.getComputedStyle(btn, '::after');
          const boxW = parseFloat(st.width);
          const maxW = parseFloat(st.maxWidth);
          document.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: 2, clientY: 2, buttons: 0 }));
          return {
            ok: overflowVisible && Number.isFinite(boxW) && Number.isFinite(maxW) && boxW > 40 && boxW < maxW - 1,
            overflowVisible,
            boxW: Math.round(boxW * 10) / 10,
            maxW: Math.round(maxW * 10) / 10,
          };
        };
        const checkAllTooltipButtonsOverflowVisible = () =>
          Array.from(document.querySelectorAll('button[data-tooltip]')).every((b) => {
            const s = window.getComputedStyle(b);
            return s.overflowX === 'visible' && s.overflowY === 'visible';
          });

        // Switch to Piano Roll to test Undo, Redo, and MidiTrackImportModal confirm disabled tooltips
        document.querySelector('[data-testid="tab-pianoroll"]')?.click();
        await waitMs(80);
        const pianoRollOverflowVisible = checkAllTooltipButtonsOverflowVisible();

        // Draw 1 note if needed so we can test both disabled Redo and disabled Undo cleanly
        const sampleBtn = document.querySelector('[data-testid="load-sample-arrangement-button"]');
        if (sampleBtn instanceof HTMLButtonElement) {
          sampleBtn.click();
          await waitMs(60);
          if (document.querySelector('[role="dialog"]')) {
            const ok = document.querySelector('[data-testid="confirm-midi-import-button"]');
            if (ok instanceof HTMLButtonElement) ok.click();
            await waitMs(60);
          }
          // Open MidiTrackImportModal again, uncheck all tracks -> confirm-midi-import-button becomes :disabled
          sampleBtn.click();
          await waitMs(80);
        }
        let modalConfirmCheck = { ok: false, reason: 'modal-not-opened' };
        let modalOverflowVisible = false;
        const openDialog = document.querySelector('[role="dialog"]');
        if (openDialog) {
          const cbs = Array.from(openDialog.querySelectorAll('input[type="checkbox"]'));
          for (const cb of cbs) {
            if (cb instanceof HTMLInputElement && cb.checked) {
              cb.click();
            }
          }
          await waitMs(40);
          modalOverflowVisible = checkAllTooltipButtonsOverflowVisible();
          const confirmBtn = openDialog.querySelector('[data-testid="confirm-midi-import-button"]');
          if (confirmBtn instanceof HTMLButtonElement && confirmBtn.disabled) {
            modalConfirmCheck = {
              ...checkBtnTooltipFits(confirmBtn),
              posTop: confirmBtn.getAttribute('data-tooltip-pos') === 'top',
            };
          }
          const closeModalBtn = openDialog.querySelector('button[aria-label="閉じる"]');
          if (closeModalBtn instanceof HTMLButtonElement) closeModalBtn.click();
          await waitMs(60);
        }

        const redoBtn = document.querySelector('button[aria-label="やり直す"]');
        const undoBtn = document.querySelector('button[aria-label="元に戻す"]');
        const redoCheck =
          redoBtn instanceof HTMLButtonElement && redoBtn.disabled
            ? checkBtnTooltipFits(redoBtn)
            : { ok: false, reason: 'redo-not-disabled' };
        for (let guard = 0; guard < 16; guard += 1) {
          if (!(undoBtn instanceof HTMLButtonElement) || undoBtn.disabled) break;
          undoBtn.click();
          await waitMs(15);
        }
        const undoCheck =
          undoBtn instanceof HTMLButtonElement && undoBtn.disabled
            ? checkBtnTooltipFits(undoBtn)
            : { ok: false, reason: 'undo-not-disabled' };

        // Ensure blank workspace so CoT button is in meta disabled state
        window.dispatchEvent(new KeyboardEvent('keydown', { key: 'a', code: 'KeyA', ctrlKey: true, bubbles: true }));
        window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Delete', code: 'Delete', bubbles: true }));
        await waitMs(50);
        document.querySelector('[data-testid="tab-compass-ai"]')?.click();
        await waitMs(80);
        const controlPanelOverflowVisible = checkAllTooltipButtonsOverflowVisible();

        const cotBtn = Array.from(document.querySelectorAll('button')).find((b) =>
          (b.textContent || '').includes('構造推論 自動省略'),
        );
        const cotCheck = checkBtnTooltipFits(cotBtn);

        const rescanBtn = document.querySelector('button[aria-label="モデルフォルダを再スキャン"]');
        let rescanCheck = { ok: false, reason: 'missing-rescan' };
        if (rescanBtn instanceof HTMLButtonElement) {
          const prevDisabled = rescanBtn.disabled;
          const prevTip = rescanBtn.getAttribute('data-tooltip');
          rescanBtn.disabled = true;
          rescanBtn.setAttribute('data-tooltip', 'モデルフォルダを再スキャン中です');
          rescanCheck = checkBtnTooltipFits(rescanBtn);
          rescanBtn.disabled = prevDisabled;
          if (prevTip !== null) rescanBtn.setAttribute('data-tooltip', prevTip);
        }

        // Switch to External DAW (cubase-midi-remote) -> check preview & popout disabled tooltips
        const dawSelect = document.querySelector('#compass-daw-select');
        if (dawSelect instanceof HTMLSelectElement) {
          const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
          setter?.call(dawSelect, 'cubase-midi-remote');
          dawSelect.dispatchEvent(new Event('change', { bubbles: true }));
        }
        await waitMs(100);
        const previewBtn = document.querySelector('button[aria-label="サウンド確認"]');
        const popoutBtn = document.querySelector('button[aria-label="別ウィンドウでピアノロールを開く"]');
        const previewCheck = checkBtnTooltipFits(previewBtn);
        const popoutCheck = checkBtnTooltipFits(popoutBtn);

        if (dawSelect instanceof HTMLSelectElement) {
          const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
          setter?.call(dawSelect, 'demo-pianoroll');
          dawSelect.dispatchEvent(new Event('change', { bubbles: true }));
        }
        await waitMs(100);
        return {
          cotCheck,
          rescanCheck,
          previewCheck,
          popoutCheck,
          undoCheck,
          redoCheck,
          modalConfirmCheck,
          allTooltipBtnsOverflowVisible:
            pianoRollOverflowVisible && modalOverflowVisible && controlPanelOverflowVisible,
          allUntruncated:
            cotCheck.ok &&
            rescanCheck.ok &&
            previewCheck.ok &&
            popoutCheck.ok &&
            undoCheck.ok &&
            redoCheck.ok &&
            modalConfirmCheck.ok &&
            modalConfirmCheck.posTop === true &&
            pianoRollOverflowVisible &&
            modalOverflowVisible &&
            controlPanelOverflowVisible,
        };
      })()
    `);

    // Verify `[R11-MED-03]` & `[R12-HIGH-01]` focus modality, Blink mouse-click `Undo` + `Space` playback, and `63 -> 64` sticky Track-Rack header
    const round11FocusCheck = await win.webContents.executeJavaScript(`
      (async () => {
        const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
        const sftBtn = Array.from(document.querySelectorAll('button')).find((b) =>
          (b.textContent || '').includes('音楽構造・ジャンル・タスクの詳細指定'),
        );
        if (sftBtn instanceof HTMLButtonElement && sftBtn.getAttribute('aria-expanded') !== 'true') {
          sftBtn.click();
          await waitMs(60);
        }
        const genreInput = document.querySelector('#genre-filter-input');
        const genrePlaceholderFits =
          genreInput instanceof HTMLInputElement &&
          genreInput.placeholder === 'ジャンルを検索 (例: jazz, ambient)';

        const jazzBtn = Array.from(document.querySelectorAll('button')).find(
          (b) => (b.textContent || '').trim() === 'jazz',
        );
        jazzBtn?.click();
        await waitMs(50);
        const clearBtn = Array.from(document.querySelectorAll('button')).find((b) =>
          (b.textContent || '').includes('選択をクリア'),
        );
        clearBtn?.focus();
        clearBtn?.click();
        await waitMs(50);
        const focusAfterGenreClear = document.activeElement ? document.activeElement.id : '';

        document.querySelector('[data-testid="tab-pianoroll"]')?.click();
        await waitMs(80);
        const addTrkBtn = document.querySelector('button[aria-label="トラックを追加"]');
        addTrkBtn?.click();
        await waitMs(60);
        const trashBtns = document.querySelectorAll('button[data-tooltip="トラックを削除"]');
        const lastTrash = trashBtns[trashBtns.length - 1];
        if (lastTrash instanceof HTMLButtonElement) {
          lastTrash.focus();
          lastTrash.click();
        }
        await waitMs(60);
        const activeAfterTrackDelete = document.activeElement;
        const focusAfterTrackDeleteValid = Boolean(
          activeAfterTrackDelete &&
            activeAfterTrackDelete !== document.body &&
            activeAfterTrackDelete.hasAttribute('data-track-select-id'),
        );

        // Draw 2 notes via Pencil so Undo stays enabled after 1st click, and return undoBtn center for real Blink mouse click
        const pencilBtn = document.querySelector('button[aria-label="ペン入力"]');
        if (pencilBtn instanceof HTMLButtonElement && pencilBtn.getAttribute('aria-pressed') !== 'true') {
          pencilBtn.click();
          await waitMs(40);
        }
        const canvas = document.querySelector('[data-testid="pianoroll-canvas"]');
        if (canvas instanceof HTMLCanvasElement) {
          const cr = canvas.getBoundingClientRect();
          const px1 = cr.left + 220;
          const py1 = cr.top + 180;
          canvas.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: px1, clientY: py1, button: 0, pointerId: 1 }));
          window.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, clientX: px1, clientY: py1, button: 0, pointerId: 1 }));
          await waitMs(40);
          const px2 = cr.left + 280;
          const py2 = cr.top + 160;
          canvas.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: px2, clientY: py2, button: 0, pointerId: 1 }));
          window.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, clientX: px2, clientY: py2, button: 0, pointerId: 1 }));
          await waitMs(40);
        }
        const undoBtn = document.querySelector('button[aria-label="元に戻す"]');
        const ur = undoBtn instanceof HTMLButtonElement ? undoBtn.getBoundingClientRect() : null;
        return {
          genrePlaceholderFits,
          focusAfterGenreClear,
          focusAfterTrackDeleteValid,
          undoReady: undoBtn instanceof HTMLButtonElement && !undoBtn.disabled,
          undoCenter: ur ? { x: Math.round(ur.left + ur.width * 0.5), y: Math.round(ur.top + ur.height * 0.5) } : null,
        };
      })()
    `);

    if (round11FocusCheck.undoCenter) {
      win.webContents.sendInputEvent({
        type: 'mouseDown',
        x: round11FocusCheck.undoCenter.x,
        y: round11FocusCheck.undoCenter.y,
        button: 'left',
        clickCount: 1,
      });
      win.webContents.sendInputEvent({
        type: 'mouseUp',
        x: round11FocusCheck.undoCenter.x,
        y: round11FocusCheck.undoCenter.y,
        button: 'left',
        clickCount: 1,
      });
      await sleep(60);
      win.webContents.sendInputEvent({ type: 'keyDown', keyCode: 'Space' });
      win.webContents.sendInputEvent({ type: 'keyUp', keyCode: 'Space' });
      await sleep(80);
    }

    const round12EnabledUndoCheck = await win.webContents.executeJavaScript(`
      (async () => {
        const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
        const undoBtn = document.querySelector('button[aria-label="元に戻す"]');
        const redoBtn = document.querySelector('button[aria-label="やり直す"]');
        const pauseBtn = document.querySelector('button[aria-label="一時停止"]');
        const spaceStartedPlayback = Boolean(pauseBtn);
        const undoStayedEnabledAndFocused = Boolean(
          undoBtn instanceof HTMLButtonElement &&
            !undoBtn.disabled &&
            document.activeElement === undoBtn,
        );
        const spaceDidNotTriggerRedo = Boolean(redoBtn instanceof HTMLButtonElement && !redoBtn.disabled);
        pauseBtn?.click();
        await waitMs(40);
        const prTab = document.getElementById('workspace-tab-pianoroll');
        if (prTab instanceof HTMLElement) {
          prTab.focus();
        }
        return {
          spaceStartedPlayback,
          undoStayedEnabledAndFocused,
          spaceDidNotTriggerRedo,
        };
      })()
    `);

    win.webContents.sendInputEvent({ type: 'keyDown', keyCode: 'Space' });
    win.webContents.sendInputEvent({ type: 'keyUp', keyCode: 'Space' });
    await sleep(80);

    const round12PointerAndStickyCheck = await win.webContents.executeJavaScript(`
      (async () => {
        const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
        const pauseAfterTabSpace = document.querySelector('button[aria-label="一時停止"]');
        const activeTabSpaceStartedPlayback = Boolean(pauseAfterTabSpace);
        pauseAfterTabSpace?.click();
        await waitMs(40);

        // Add tracks up to 63 (yielding so React commits each batch), then focus + and activate 64th track
        let addBtn = document.querySelector('button[aria-label="トラックを追加"]');
        for (let guard = 0; guard < 70; guard += 1) {
          const count = document.querySelectorAll('button[data-track-select-id]').length;
          if (count >= 63) break;
          addBtn = document.querySelector('button[aria-label="トラックを追加"]');
          addBtn?.click();
          await waitMs(4);
        }
        await waitMs(40);
        const aside = document.querySelector('aside[aria-label="トラックラック"]');
        if (aside instanceof HTMLElement) {
          aside.scrollTop = 0;
        }
        addBtn = document.querySelector('button[aria-label="トラックを追加"]');
        if (addBtn instanceof HTMLButtonElement) {
          addBtn.focus();
          addBtn.click();
        }
        await waitMs(60);
        const stickyHdr = document.querySelector('aside [data-tooltip-sticky-header="true"]');
        const asideRect = aside instanceof HTMLElement ? aside.getBoundingClientRect() : null;
        const hdrRect = stickyHdr instanceof HTMLElement ? stickyHdr.getBoundingClientRect() : null;
        const stickyHeaderStayedVisible = Boolean(
          aside instanceof HTMLElement &&
            aside.scrollTop === 0 &&
            asideRect &&
            hdrRect &&
            Math.abs(hdrRect.top - asideRect.top) <= 2 &&
            addBtn instanceof HTMLButtonElement &&
            addBtn.disabled,
        );
        const focusAfter64thTrackValid = Boolean(
          document.activeElement &&
            document.activeElement !== document.body &&
            document.activeElement.hasAttribute('data-track-select-id'),
        );
        // Scroll Track 1 partially behind sticky header, open inline rename via double-click, press Escape, and verify scroll clearance
        let stickyScrollClearanceOk = false;
        const firstTrackBtn = document.querySelector('button[data-track-select-id]');
        if (aside instanceof HTMLElement && firstTrackBtn instanceof HTMLButtonElement) {
          aside.scrollTop = 28;
          await waitMs(20);
          firstTrackBtn.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
          await waitMs(40);
          const renameInput = aside.querySelector('input[type="text"]');
          if (renameInput instanceof HTMLInputElement) {
            renameInput.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', code: 'Escape', bubbles: true, cancelable: true }));
            await waitMs(40);
          }
          const restoredTrackBtn = document.querySelector('button[data-track-select-id]');
          const curHdr = document.querySelector('aside [data-tooltip-sticky-header="true"]');
          if (restoredTrackBtn instanceof HTMLButtonElement && curHdr instanceof HTMLElement) {
            stickyScrollClearanceOk =
              document.activeElement === restoredTrackBtn &&
              restoredTrackBtn.getBoundingClientRect().top >= curHdr.getBoundingClientRect().bottom - 2;
          }
        }
        return {
          activeTabSpaceStartedPlayback,
          stickyHeaderStayedVisible,
          focusAfter64thTrackValid,
          stickyScrollClearanceOk,
        };
      })()
    `);

    // `[R14-HIGH-01]` Real-Blink pointer click on sample button -> confirm modal -> Space starts playback
    const round14SampleCenter = await win.webContents.executeJavaScript(`
      (async () => {
        const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
        const sampleBtn = document.querySelector('[data-testid="load-sample-arrangement-button"]');
        if (!(sampleBtn instanceof HTMLButtonElement)) return null;
        // Ensure notes exist first so clicking sampleBtn opens MidiTrackImportModal
        sampleBtn.click();
        await waitMs(60);
        if (document.querySelector('[role="dialog"]')) {
          const ok = document.querySelector('[data-testid="confirm-midi-import-button"]');
          if (ok instanceof HTMLButtonElement) ok.click();
          await waitMs(60);
        }
        const sr = sampleBtn.getBoundingClientRect();
        return { x: Math.round(sr.left + sr.width * 0.5), y: Math.round(sr.top + sr.height * 0.5) };
      })()
    `);
    let round14ModalConfirmCenter = null;
    if (round14SampleCenter) {
      win.webContents.sendInputEvent({
        type: 'mouseDown',
        x: round14SampleCenter.x,
        y: round14SampleCenter.y,
        button: 'left',
        clickCount: 1,
      });
      win.webContents.sendInputEvent({
        type: 'mouseUp',
        x: round14SampleCenter.x,
        y: round14SampleCenter.y,
        button: 'left',
        clickCount: 1,
      });
      await sleep(80);
      round14ModalConfirmCenter = await win.webContents.executeJavaScript(`
        (() => {
          const ok = document.querySelector('[data-testid="confirm-midi-import-button"]');
          if (!(ok instanceof HTMLButtonElement)) return null;
          const r = ok.getBoundingClientRect();
          return { x: Math.round(r.left + r.width * 0.5), y: Math.round(r.top + r.height * 0.5) };
        })()
      `);
    }
    if (round14ModalConfirmCenter) {
      win.webContents.sendInputEvent({
        type: 'mouseDown',
        x: round14ModalConfirmCenter.x,
        y: round14ModalConfirmCenter.y,
        button: 'left',
        clickCount: 1,
      });
      win.webContents.sendInputEvent({
        type: 'mouseUp',
        x: round14ModalConfirmCenter.x,
        y: round14ModalConfirmCenter.y,
        button: 'left',
        clickCount: 1,
      });
      await sleep(80);
      win.webContents.sendInputEvent({ type: 'keyDown', keyCode: 'Space' });
      win.webContents.sendInputEvent({ type: 'keyUp', keyCode: 'Space' });
      await sleep(80);
    }
    const round14ModalPointerSpaceCheck = await win.webContents.executeJavaScript(`
      (async () => {
        const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
        const sampleBtn = document.querySelector('[data-testid="load-sample-arrangement-button"]');
        const pauseBtn = document.querySelector('button[aria-label="一時停止"]');
        const modalStayedClosed = document.querySelector('[role="dialog"]') === null;
        const focusRestoredToSampleBtn = Boolean(sampleBtn && document.activeElement === sampleBtn);
        const spaceStartedPlaybackAfterModalClose = Boolean(pauseBtn);
        pauseBtn?.click();
        await waitMs(40);
        return {
          modalStayedClosed,
          focusRestoredToSampleBtn,
          spaceStartedPlaybackAfterModalClose,
        };
      })()
    `);

    // Round-16 Critic Physical Runtime Check (`[R16-MED-01]`, `[R16-MED-02]`, `[R16-LOW-01]`)
    const [preR16W, preR16H] = win.getContentSize();
    win.setMinimumSize(320, 320);
    win.setContentSize(320, 360);
    await sleep(160);
    const round16CriticCheck = await win.webContents.executeJavaScript(`
      (async () => {
        const waitMs = (ms) => new Promise((r) => setTimeout(r, ms));
        // 1. [R16-MED-02] Mouse side-button (button: 3) + Enter reset on canvasHadRecentPointerRef
        const prTab = document.querySelector('[data-testid="tab-pianoroll"]');
        if (prTab && prTab.getAttribute('aria-selected') !== 'true') {
          prTab.click();
          await waitMs(80);
        }
        const canvas = document.querySelector('[data-testid="pianoroll-canvas"]');
        const pencilBtn = document.querySelector('button[aria-label="ペン入力"]');
        window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', code: 'Tab', bubbles: true }));
        pencilBtn?.focus();
        if (canvas) {
          canvas.dispatchEvent(new PointerEvent('pointerdown', {
            bubbles: true,
            cancelable: true,
            pointerId: 501,
            pointerType: 'mouse',
            button: 3,
            buttons: 8,
            clientX: 120,
            clientY: 160,
          }));
        }
        const sideBtnEv = new KeyboardEvent('keydown', {
          key: ' ',
          code: 'Space',
          bubbles: true,
          cancelable: true,
        });
        pencilBtn?.dispatchEvent(sideBtnEv);
        const sideBtnDidNotHijackSpace = sideBtnEv.defaultPrevented === false;

        if (canvas) {
          canvas.dispatchEvent(new PointerEvent('pointerdown', {
            bubbles: true,
            cancelable: true,
            pointerId: 502,
            pointerType: 'mouse',
            button: 2,
            buttons: 2,
            clientX: 120,
            clientY: 160,
          }));
        }
        pencilBtn?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
        const enterResetEv = new KeyboardEvent('keydown', {
          key: ' ',
          code: 'Space',
          bubbles: true,
          cancelable: true,
        });
        pencilBtn?.dispatchEvent(enterResetEv);
        const enterResetCanvasPointerRef = enterResetEv.defaultPrevented === false;

        // 2. [R16-MED-01] Pointer-modality session across peer BroadcastChannel SYNC_ACTIVE_CONTROLLER
        if (canvas) {
          if (document.activeElement instanceof HTMLElement && document.activeElement !== document.body) {
            document.activeElement.blur();
          }
          canvas.dispatchEvent(new PointerEvent('pointerdown', {
            bubbles: true,
            cancelable: true,
            pointerId: 503,
            pointerType: 'mouse',
            button: 0,
            buttons: 1,
            clientX: 140,
            clientY: 160,
          }));
          window.dispatchEvent(new PointerEvent('pointerup', {
            bubbles: true,
            cancelable: true,
            pointerId: 503,
            pointerType: 'mouse',
            button: 0,
            buttons: 0,
            clientX: 140,
            clientY: 160,
          }));
        }
        const bc = new BroadcastChannel('compass_ai_workspace_v1');
        bc.postMessage({
          messageId: 'e2e_r16_cubase',
          type: 'SYNC_ACTIVE_CONTROLLER',
          controllerId: 'cubase-midi-remote',
          senderId: 'e2e_r16_peer',
        });
        for (let i = 0; i < 25; i += 1) {
          if (document.getElementById('external-daw-track-select')) break;
          await waitMs(40);
        }
        const standbyActiveIsBody = document.activeElement === document.body;

        // 3. [R16-LOW-01] & [R17-LOW-01] 320px #external-daw-track-select non-overflow, ChevronDown alignment & sole native label
        const extSel = document.getElementById('external-daw-track-select');
        const extSelNoRedundantAriaLabel = Boolean(
          extSel &&
          extSel.getAttribute('aria-label') === null &&
          document.querySelector('label[for="external-daw-track-select"]')
        );
        const extWrap = extSel ? extSel.parentElement : null;
        const extChev = extWrap ? extWrap.querySelector('svg') : null;
        const selRect = extSel ? extSel.getBoundingClientRect() : null;
        const wrapRect = extWrap ? extWrap.getBoundingClientRect() : null;
        const chevRect = extChev ? extChev.getBoundingClientRect() : null;
        const innerW320 = window.innerWidth;
        const docScrollW320 = document.documentElement.scrollWidth;
        const extTrackSelectFits320 = Boolean(
          selRect &&
          wrapRect &&
          chevRect &&
          innerW320 === 320 &&
          selRect.width >= 44 &&
          selRect.right <= wrapRect.right + 0.5 &&
          wrapRect.right <= innerW320 + 0.5 &&
          chevRect.right <= wrapRect.right + 0.5 &&
          chevRect.left >= selRect.right - 32.5 &&
          docScrollW320 <= 320
        );

        // Switch back to demo-pianoroll via peer BroadcastChannel in pointer session
        bc.postMessage({
          messageId: 'e2e_r16_demo',
          type: 'SYNC_ACTIVE_CONTROLLER',
          controllerId: 'demo-pianoroll',
          senderId: 'e2e_r16_peer',
        });
        for (let i = 0; i < 25; i += 1) {
          if (document.querySelector('[data-testid="pianoroll-canvas"]')) break;
          await waitMs(40);
        }
        bc.close();
        const launchBtnAfterPeerDemo = document.querySelector('[data-testid="launch-compass-ai-button"]');
        const peerDemoFocusedLaunchBtn = Boolean(
          launchBtnAfterPeerDemo && document.activeElement === launchBtnAfterPeerDemo
        );
        const peerSpaceEv = new KeyboardEvent('keydown', {
          key: ' ',
          code: 'Space',
          bubbles: true,
          cancelable: true,
        });
        launchBtnAfterPeerDemo?.dispatchEvent(peerSpaceEv);
        await waitMs(50);
        const peerSpaceStartedPlayback = peerSpaceEv.defaultPrevented === true;
        const pauseAfterR16 = document.querySelector('button[aria-label="一時停止"]');
        pauseAfterR16?.click();
        await waitMs(40);

        // 4. [R17-MED-02] & [R17-MED-03] Click #pianoroll-bpm + Enter keeps pointer modality + controller switch resets drafts/rename
        const bpmInp = document.getElementById('pianoroll-bpm');
        if (bpmInp) {
          bpmInp.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'mouse', button: 0 }));
          bpmInp.focus();
          bpmInp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
        }
        const bcR17 = new BroadcastChannel('compass_ai_workspace_v1');
        bcR17.postMessage({
          messageId: 'e2e_r17_cubase_after_bpm_enter',
          type: 'SYNC_ACTIVE_CONTROLLER',
          controllerId: 'cubase-midi-remote',
          senderId: 'e2e_r17_peer',
        });
        for (let i = 0; i < 25; i += 1) {
          if (document.getElementById('external-daw-track-select')) break;
          await waitMs(40);
        }
        const bpmEnterKeptPointerStandby = document.activeElement === document.body;
        bcR17.postMessage({
          messageId: 'e2e_r17_demo_after_bpm_enter',
          type: 'SYNC_ACTIVE_CONTROLLER',
          controllerId: 'demo-pianoroll',
          senderId: 'e2e_r17_peer',
        });
        for (let i = 0; i < 25; i += 1) {
          if (document.querySelector('[data-testid="pianoroll-canvas"]')) break;
          await waitMs(40);
        }
        bcR17.close();
        const pauseCleanup = document.querySelector('button[aria-label="一時停止"]');
        pauseCleanup?.click();
        await waitMs(30);

        return {
          sideBtnDidNotHijackSpace,
          enterResetCanvasPointerRef,
          standbyActiveIsBody,
          extSelNoRedundantAriaLabel,
          extTrackSelectFits320,
          innerW320,
          docScrollW320,
          selWidth: selRect ? Math.round(selRect.width * 10) / 10 : null,
          wrapWidth: wrapRect ? Math.round(wrapRect.width * 10) / 10 : null,
          peerDemoFocusedLaunchBtn,
          peerSpaceStartedPlayback,
          bpmEnterKeptPointerStandby,
        };
      })()
    `);
    win.setMinimumSize(375, 620);
    win.setContentSize(preR16W, preR16H);
    await sleep(120);

    const round11UiAndFitCheck = {
      fit1440Default,
      fit1440Expanded,
      fit1360Default,
      fit1360Expanded,
      fit1360ExpandedFoundation,
      fit1280Default,
      fit1280Expanded,
      fit1280ExpandedFoundation,
      round14HeaderTablistCheck,
      round14ModalPointerSpaceCheck,
      round16CriticCheck,
      disabledTooltipNonTruncationCheck,
      round11FocusCheck,
      round12PointerAndStickyCheck: {
        ...round12EnabledUndoCheck,
        ...round12PointerAndStickyCheck,
      },
    };

    win.destroy();
    killPythonTree();
    await sleep(250);

    // 6. Verify Spawned Python Process Tree is Dead (Zero Zombie Check)
    let zombieAlive = false;
    if (spawnedPid) {
      try {
        process.kill(spawnedPid, 0);
        zombieAlive = true;
      } catch {
        zombieAlive = false;
      }
    }

    const report = {
      timestamp: new Date().toISOString(),
      splashCheck,
      audits: [
        auditDesktopControl,
        auditMobileControl,
        auditDesktopPianoRoll,
        auditMobilePianoRoll,
        auditOpenMidiModal,
      ],
      mobileCanvasCheck,
      mobileModalBoundsCheck,
      wheelAndAuthCheck,
      consoleErrors,
      keyboardIsolationCheck,
      multiTrackModalResult,
      midiUploadResult,
      afterShiftG,
      afterLaunchButton,
      sawGeneratingState,
      autoSwitchedToPianoRoll,
      finalState,
      customSchemeIpcCheck: {
        ...customSchemeIpcCheck,
        popoutOpenCount,
        workspaceSyncValidatedCount,
      },
      reducedMotionCascadeCheck,
      velocityLaneBoundsCheck,
      mobileCrowdedCanvasCheck,
      compassPianoRollUiCheck,
      popoutWindowUiCheck,
      round10UiAndSyncCheck,
      round11UiAndFitCheck,
      zombieAlive,
      screenshots: {
        splashShot,
        desktopControlShot,
        desktopControlExpandedShot,
        mobileControlShot,
        pianoRollUploadedShot,
        pianoRollGeneratedShot,
      },
    };

    const reportPath = path.join(OUTPUT_DIR, 'e2e_audit_report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf8');
    console.log(JSON.stringify(report, null, 2));

    const allAuditsClean = report.audits.every(
      (a) =>
        a.rootColorScheme === 'dark' &&
        !a.hasHorizontalOverflow &&
        a.slopGlyphFound === null &&
        a.auroraBackdropPresent === true &&
        a.auroraPointerEventsNone === true &&
        a.auroraGradientLayerCount >= 3 &&
        a.nonAuroraGradientCount === 0 &&
        a.redundantTargetSectionAbsent === true &&
        a.backdropBlurCount === 0 &&
        a.neonShadowCount === 0 &&
        a.nativeTitleCount === 0 &&
        a.undersizedControls.length === 0 &&
        a.unroundedControls.length === 0 &&
        a.unlabeledControls.length === 0 &&
        a.iconMissingTooltipControls.length === 0 &&
        a.lowContrastControls.length === 0 &&
        a.solidPrimaryButtons.length <= 1,
    );

    const passed =
      splashCheck.origin === 'compass-app://studio' &&
      splashCheck.hasCompassTitle &&
      splashCheck.hasLogo &&
      splashCheck.hasAuroraBackdrop &&
      splashCheck.hasLoading &&
      allAuditsClean &&
      compassPianoRollUiCheck.hasRedundantSelectBtn === false &&
      compassPianoRollUiCheck.hasRedundantPanBtn === false &&
      compassPianoRollUiCheck.hasPencilToggle === true &&
      compassPianoRollUiCheck.hasRangeSelectBtn === true &&
      compassPianoRollUiCheck.hasTripletBtn === true &&
      compassPianoRollUiCheck.hasLabeledSampleBtn === true &&
      compassPianoRollUiCheck.hasTelemetryBar === true &&
      compassPianoRollUiCheck.toolbarHeight1440 <= 62 &&
      compassPianoRollUiCheck.canvasContainerTransparent === true &&
      compassPianoRollUiCheck.velocityLaneTranslucent === true &&
      compassPianoRollUiCheck.playheadRenderedInGrid === true &&
      compassPianoRollUiCheck.rackToggleFound === true &&
      popoutWindowUiCheck.hasPopoutBadge === true &&
      popoutWindowUiCheck.hasTablist === false &&
      popoutWindowUiCheck.hasSelfPopoutButton === false &&
      popoutWindowUiCheck.popoutToolbarHeight1280 <= 62 &&
      popoutWindowUiCheck.mainDefaultToolbarHeight1360 <= 62 &&
      popoutWindowUiCheck.mobileTooltipClippedCount === 0 &&
      mobileCanvasCheck.trackRackAutoCollapsed &&
      mobileCanvasCheck.canvasHeight >= 240 &&
      mobileCanvasCheck.canvasBottomFitsViewport &&
      mobileCanvasCheck.telemetryBarVisibleOnMobile &&
      mobileModalBoundsCheck.opened &&
      mobileModalBoundsCheck.capHintMounted &&
      mobileModalBoundsCheck.headerInertWhileModalOpen &&
      mobileModalBoundsCheck.closeBtnReceivesHitPoint &&
      mobileModalBoundsCheck.docScrollWidth <= 375 &&
      mobileModalBoundsCheck.closeFitsInsideDialog &&
      mobileModalBoundsCheck.cardFitsVertically &&
      mobileModalBoundsCheck.visibleTrackAreaAtScroll0Px >= 88 &&
      mobileModalBoundsCheck.lastRowVisibleAtBottom &&
      mobileModalBoundsCheck.popoutMinModal375x521.opened &&
      mobileModalBoundsCheck.popoutMinModal375x521.capHintMounted &&
      mobileModalBoundsCheck.popoutMinModal375x521.headerInertWhileModalOpen &&
      mobileModalBoundsCheck.popoutMinModal375x521.closeBtnReceivesHitPoint &&
      mobileModalBoundsCheck.popoutMinModal375x521.closeFitsInsideDialog &&
      mobileModalBoundsCheck.popoutMinModal375x521.cardFitsVertically &&
      mobileModalBoundsCheck.popoutMinModal375x521.visibleTrackAreaAtScroll0Px >= 88 &&
      mobileModalBoundsCheck.popoutMinModal375x521.lastRowVisibleAtBottom &&
      mobileModalBoundsCheck.landscapeCanvas667x375.fitsViewport &&
      mobileModalBoundsCheck.landscapeModal667x375.opened &&
      mobileModalBoundsCheck.landscapeModal667x375.capHintMounted &&
      mobileModalBoundsCheck.landscapeModal667x375.headerInertWhileModalOpen &&
      mobileModalBoundsCheck.landscapeModal667x375.closeBtnReceivesHitPoint &&
      mobileModalBoundsCheck.landscapeModal667x375.closeFitsInsideDialog &&
      mobileModalBoundsCheck.landscapeModal667x375.cardFitsVertically &&
      mobileModalBoundsCheck.landscapeModal667x375.visibleTrackAreaAtScroll0Px >= 88 &&
      mobileModalBoundsCheck.landscapeModal667x375.lastRowVisibleAtBottom &&
      mobileModalBoundsCheck.compactModal360x375.opened &&
      mobileModalBoundsCheck.compactModal360x375.capHintMounted &&
      mobileModalBoundsCheck.compactModal360x375.headerInertWhileModalOpen &&
      mobileModalBoundsCheck.compactModal360x375.closeBtnReceivesHitPoint &&
      mobileModalBoundsCheck.compactModal360x375.closeFitsInsideDialog &&
      mobileModalBoundsCheck.compactModal360x375.cardFitsVertically &&
      mobileModalBoundsCheck.compactModal360x375.visibleTrackAreaAtScroll0Px >= 88 &&
      mobileModalBoundsCheck.compactModal360x375.lastRowVisibleAtBottom &&
      mobileModalBoundsCheck.compactScrollEscape320x360.nonTrappedScrollZone &&
      mobileModalBoundsCheck.compactModal320x360.opened &&
      mobileModalBoundsCheck.compactModal320x360.capHintMounted &&
      mobileModalBoundsCheck.compactModal320x360.headerInertWhileModalOpen &&
      mobileModalBoundsCheck.compactModal320x360.closeBtnReceivesHitPoint &&
      mobileModalBoundsCheck.compactModal320x360.docScrollWidth <= 320 &&
      mobileModalBoundsCheck.compactModal320x360.closeFitsInsideDialog &&
      mobileModalBoundsCheck.compactModal320x360.cardFitsVertically &&
      mobileModalBoundsCheck.compactModal320x360.visibleTrackAreaAtScroll0Px >= 88 &&
      mobileModalBoundsCheck.compactModal320x360.lastRowVisibleAtBottom &&
      mobileModalBoundsCheck.narrowModal320x501.opened &&
      mobileModalBoundsCheck.narrowModal320x501.capHintMounted &&
      mobileModalBoundsCheck.narrowModal320x501.headerInertWhileModalOpen &&
      mobileModalBoundsCheck.narrowModal320x501.closeBtnReceivesHitPoint &&
      mobileModalBoundsCheck.narrowModal320x501.docScrollWidth <= 320 &&
      mobileModalBoundsCheck.narrowModal320x501.closeFitsInsideDialog &&
      mobileModalBoundsCheck.narrowModal320x501.cardFitsVertically &&
      mobileModalBoundsCheck.narrowModal320x501.visibleTrackAreaAtScroll0Px >= 88 &&
      mobileModalBoundsCheck.narrowModal320x501.lastRowVisibleAtBottom &&
      wheelAndAuthCheck.wheelDefaultPrevented &&
      wheelAndAuthCheck.unauthStatus === 403 &&
      wheelAndAuthCheck.focusedTagBeforeShiftG === 'BUTTON' &&
      consoleErrors.length === 0 &&
      keyboardIsolationCheck.spaceNotPrevented &&
      keyboardIsolationCheck.notesSurvived &&
      multiTrackModalResult.modalOpened &&
      multiTrackModalResult.headerAriaHidden === true &&
      multiTrackModalResult.focusInsideModal === true &&
      multiTrackModalResult.modalClosed &&
      multiTrackModalResult.headerRestored === true &&
      multiTrackModalResult.focusRestoredToUploadBtn === true &&
      multiTrackModalResult.sampleModalTitle === 'サンプルアレンジの読み込み設定' &&
      multiTrackModalResult.sampleFocusInside === true &&
      multiTrackModalResult.focusRestoredToSampleBtn === true &&
      multiTrackModalResult.tabText.includes('4') &&
      midiUploadResult.tabText.includes('4') &&
      midiUploadResult.singleTrackLoaded === true &&
      midiUploadResult.hasPositiveUploadNotice === true &&
      midiUploadResult.hasErrorBanner === false &&
      afterShiftG &&
      afterLaunchButton &&
      sawGeneratingState &&
      autoSwitchedToPianoRoll &&
      finalState.pianoRollSelected &&
      finalState.hasCandidateBadge &&
      finalState.hasErrorBanner === false &&
      finalState.hasStaleUploadNotice === false &&
      finalState.finalNoteCount > 4 &&
      customSchemeIpcCheck.handshakeTokenLen === 64 &&
      customSchemeIpcCheck.authedHardwareStatus === 200 &&
      popoutOpenCount >= 2 &&
      workspaceSyncValidatedCount >= 1 &&
      reducedMotionCascadeCheck.mediaMatchesReduce === true &&
      reducedMotionCascadeCheck.spinAnimDurationSec <= 0.001 &&
      reducedMotionCascadeCheck.genericSpinAnimDurationSec <= 0.001 &&
      reducedMotionCascadeCheck.dur300TransitionSec <= 0.001 &&
      reducedMotionCascadeCheck.btnTransitionSec <= 0.001 &&
      velocityLaneBoundsCheck.prRestoredBounds.velocityClippedOffViewport === false &&
      velocityLaneBoundsCheck.prAfterRackOffBounds.velocityClippedOffViewport === false &&
      velocityLaneBoundsCheck.prShrunk650Bounds.velocityClippedOffViewport === false &&
      mobileCrowdedCanvasCheck.crowdedPortrait375x667.innerH === 667 &&
      mobileCrowdedCanvasCheck.crowdedPortrait375x667.hasHorizontalOverflow === false &&
      mobileCrowdedCanvasCheck.crowdedPortrait375x667.rackOpen === true &&
      mobileCrowdedCanvasCheck.crowdedPortrait375x667.has4Candidates === true &&
      mobileCrowdedCanvasCheck.crowdedPortrait375x667.canvasHeight >= 240 &&
      mobileCrowdedCanvasCheck.crowdedPortrait375x667.aspectDeltaPx <= 2 &&
      mobileCrowdedCanvasCheck.crowdedPortrait375x667.firstTrashFlippedFromTop === true &&
      mobileCrowdedCanvasCheck.crowdedPortrait375x667.tooltipEscapeDismissed === true &&
      mobileCrowdedCanvasCheck.crowdedLandscape667x375.innerH === 375 &&
      mobileCrowdedCanvasCheck.crowdedLandscape667x375.hasHorizontalOverflow === false &&
      mobileCrowdedCanvasCheck.crowdedLandscape667x375.rackOpen === true &&
      mobileCrowdedCanvasCheck.crowdedLandscape667x375.has4Candidates === true &&
      mobileCrowdedCanvasCheck.crowdedLandscape667x375.canvasHeight >= 176 &&
      mobileCrowdedCanvasCheck.crowdedLandscape667x375.aspectDeltaPx <= 2 &&
      mobileCrowdedCanvasCheck.crowdedLandscape667x375.firstTrashFlippedFromTop === true &&
      mobileCrowdedCanvasCheck.crowdedLandscape667x375.tooltipEscapeDismissed === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.ok === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.tooltipPointerEventsNone === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.noBeforeHoverTrap === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.soloHitUnoccluded === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.trashBridgeClearedOnNeighbor === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.bridgeActiveOverCanvas === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.canvasRulerUnoccluded === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.bridgeClearedOnToolbarNeighbor === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.splitWorkspaceZIndex5 === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.focusRestoreDismissedOnHover === true &&
      mobileCrowdedCanvasCheck.tooltipHitTestAndBridgeCheck.inertSuppressesTooltipAfter === true &&
      round10UiAndSyncCheck.syncControllerIpcValid === true &&
      round10UiAndSyncCheck.popoutRound10Check.standbyMounted === true &&
      round10UiAndSyncCheck.popoutRound10Check.standbyZeroFormControls === true &&
      round10UiAndSyncCheck.popoutRound10Check.standbyLaunchShortcut === true &&
      round10UiAndSyncCheck.popoutRound10Check.switchedBackToCanvas === true &&
      round10UiAndSyncCheck.popoutRound10Check.focusRetainedAfterSwitch === true &&
      round10UiAndSyncCheck.popoutRound10Check.stripZBefore === '10' &&
      round10UiAndSyncCheck.popoutRound10Check.stripZActive === '30' &&
      round10UiAndSyncCheck.popoutRound10Check.stripZAfterEscape === '10' &&
      round10UiAndSyncCheck.popoutRound10Check.disabledBtnFullOpacity === true &&
      round10UiAndSyncCheck.popoutRound10Check.disabledAfterFullOpacity === true &&
      round10UiAndSyncCheck.popoutRound10Check.disabledAfterZIndex60 === true &&
      round10UiAndSyncCheck.popoutRound10Check.disabledBridgeActivated === true &&
      round10UiAndSyncCheck.popoutRound10Check.disabledEscapeDismissed === true &&
      round10UiAndSyncCheck.popoutRound10Check.disabledRehoverRecovered === true &&
      round10UiAndSyncCheck.mainRound10Check.previewTooltipIdle === 'サウンド確認' &&
      round10UiAndSyncCheck.mainRound10Check.cotDisabledInMeta === true &&
      round10UiAndSyncCheck.mainRound10Check.previewDisabledOnExternal === true &&
      round10UiAndSyncCheck.mainRound10Check.popoutHeaderFullOpacityWhenDisabled === true &&
      round10UiAndSyncCheck.mainRound10Check.extLaunchShortcut === true &&
      round10UiAndSyncCheck.mainRound10Check.activeIdAfterProgrammaticSwitch === 'workspace-tab-compass-ai' &&
      round11UiAndFitCheck.fit1440Default.fitsWithoutVerticalScroll === true &&
      round11UiAndFitCheck.fit1440Default.fitsWithoutHorizontalScroll === true &&
      round11UiAndFitCheck.fit1440Expanded.fitsWithoutVerticalScroll === true &&
      round11UiAndFitCheck.fit1440Expanded.genreTrayHeight >= 96 &&
      round11UiAndFitCheck.fit1440Expanded.genreTrayZeroNegMargin === true &&
      round11UiAndFitCheck.fit1360Default.fitsWithoutVerticalScroll === true &&
      round11UiAndFitCheck.fit1360Default.fitsWithoutHorizontalScroll === true &&
      round11UiAndFitCheck.fit1360Expanded.fitsWithoutVerticalScroll === true &&
      round11UiAndFitCheck.fit1360Expanded.genreTrayHeight >= 96 &&
      round11UiAndFitCheck.fit1360ExpandedFoundation.fitsWithoutVerticalScroll === true &&
      round11UiAndFitCheck.fit1280Default.fitsWithoutVerticalScroll === true &&
      round11UiAndFitCheck.fit1280Default.fitsWithoutHorizontalScroll === true &&
      round11UiAndFitCheck.fit1280Expanded.fitsWithoutVerticalScroll === true &&
      round11UiAndFitCheck.fit1280Expanded.fitsWithoutHorizontalScroll === true &&
      round11UiAndFitCheck.fit1280ExpandedFoundation.fitsWithoutVerticalScroll === true &&
      round11UiAndFitCheck.fit1280ExpandedFoundation.fitsWithoutHorizontalScroll === true &&
      round11UiAndFitCheck.round14HeaderTablistCheck.navRoleTablistAbsent === true &&
      round11UiAndFitCheck.round14HeaderTablistCheck.divRoleTablistPresent === true &&
      round11UiAndFitCheck.round14HeaderTablistCheck.headerDomMatchesVisualX === true &&
      round11UiAndFitCheck.round14HeaderTablistCheck.tablistArrowSwitchOk === true &&
      round11UiAndFitCheck.round14HeaderTablistCheck.rovingTabIndexOk === true &&
      round11UiAndFitCheck.round14HeaderTablistCheck.sftAriaControlsMountedWhenCollapsed === true &&
      round11UiAndFitCheck.round14ModalPointerSpaceCheck.modalStayedClosed === true &&
      round11UiAndFitCheck.round14ModalPointerSpaceCheck.focusRestoredToSampleBtn === true &&
      round11UiAndFitCheck.round14ModalPointerSpaceCheck.spaceStartedPlaybackAfterModalClose === true &&
      round11UiAndFitCheck.round16CriticCheck.sideBtnDidNotHijackSpace === true &&
      round11UiAndFitCheck.round16CriticCheck.enterResetCanvasPointerRef === true &&
      round11UiAndFitCheck.round16CriticCheck.standbyActiveIsBody === true &&
      round11UiAndFitCheck.round16CriticCheck.extSelNoRedundantAriaLabel === true &&
      round11UiAndFitCheck.round16CriticCheck.extTrackSelectFits320 === true &&
      round11UiAndFitCheck.round16CriticCheck.peerDemoFocusedLaunchBtn === true &&
      round11UiAndFitCheck.round16CriticCheck.peerSpaceStartedPlayback === true &&
      round11UiAndFitCheck.round16CriticCheck.bpmEnterKeptPointerStandby === true &&
      round11UiAndFitCheck.disabledTooltipNonTruncationCheck.allUntruncated === true &&
      round11UiAndFitCheck.round11FocusCheck.genrePlaceholderFits === true &&
      round11UiAndFitCheck.round11FocusCheck.focusAfterGenreClear === 'genre-filter-input' &&
      round11UiAndFitCheck.round11FocusCheck.focusAfterTrackDeleteValid === true &&
      round11UiAndFitCheck.round12PointerAndStickyCheck.spaceStartedPlayback === true &&
      round11UiAndFitCheck.round12PointerAndStickyCheck.undoStayedEnabledAndFocused === true &&
      round11UiAndFitCheck.round12PointerAndStickyCheck.spaceDidNotTriggerRedo === true &&
      round11UiAndFitCheck.round12PointerAndStickyCheck.activeTabSpaceStartedPlayback === true &&
      round11UiAndFitCheck.round12PointerAndStickyCheck.stickyHeaderStayedVisible === true &&
      round11UiAndFitCheck.round12PointerAndStickyCheck.focusAfter64thTrackValid === true &&
      round11UiAndFitCheck.round12PointerAndStickyCheck.stickyScrollClearanceOk === true &&
      !zombieAlive;

    cleanupTempModels();
    app.exit(passed ? 0 : 1);
  } catch (err) {
    console.error('[E2E AUDIT ERROR]', err);
    killPythonTree();
    cleanupTempModels();
    app.exit(1);
  }
});
