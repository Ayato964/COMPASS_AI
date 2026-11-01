/**
 * COMPASS AI Chromium Desktop Shell (`electron/main.cjs`).
 * - Serves production assets over privileged `compass-app://studio` origin (eliminating CORS `null` origins)
 * - Opens Splash Screen immediately (<20ms) and asynchronously probes/launches the local Python MORTM 4.5 engine (`main.py`)
 * - Streams buffered & live boot logs to the Splash Screen
 * - Manages foreground window elevation (`Shift+G` -> Control Panel, Generation Done -> Piano Roll)
 * - Guarantees zero zombie child processes via `COMPASS_PARENT_PID` kernel watchdog + `taskkill /T /F`
 */

const { app, BrowserWindow, ipcMain, protocol, session } = require('electron');
const { execFile, spawn, spawnSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const https = require('https');
const net = require('net');
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
const MAX_HEALTH_BODY_BYTES = 8192;

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

let mainWindow = null;
let mainWindowLoaded = false;
let pianoRollWindow = null;
let pythonProc = null;
let spawnedByElectron = false;
let isShuttingDown = false;
let isQuittingDuplicate = false;
function normalizeExplicitApiUrl(rawUrl) {
  const trimmed = String(rawUrl || '').trim();
  if (!trimmed) return null;
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return null;
    }
    parsed.username = '';
    parsed.password = '';
    parsed.search = '';
    parsed.hash = '';
    parsed.pathname = parsed.pathname.replace(/\/+$/, '');
    const defaultPort = parsed.protocol === 'https:' ? 443 : 80;
    const port = parsed.port ? Number(parsed.port) || defaultPort : defaultPort;
    const normalized = parsed.toString().replace(/\/+$/, '');
    return { url: normalized, port, protocol: parsed.protocol };
  } catch {
    return null;
  }
}

const EXPLICIT_API_URL = (process.env.COMPASS_API_URL || '').trim();
const EXPLICIT_HANDSHAKE_TOKEN = (
  process.env.COMPASS_HANDSHAKE_TOKEN ||
  process.env.COMPASS_API_HANDSHAKE_TOKEN ||
  ''
).trim();
const LOCAL_HANDSHAKE_TOKEN = crypto.randomBytes(32).toString('hex');
const PARSED_EXPLICIT_API = normalizeExplicitApiUrl(EXPLICIT_API_URL);
let usingExplicitBackend = Boolean(PARSED_EXPLICIT_API);
let resolvedApiBaseUrl = PARSED_EXPLICIT_API
  ? PARSED_EXPLICIT_API.url
  : 'http://127.0.0.1:8000';
let backendAuthenticated = false;
let backendVerified = false;
let lazyAuthProbePromise = null;
const pendingBootLogs = [];

function activeHandshakeToken() {
  return usingExplicitBackend ? EXPLICIT_HANDSHAKE_TOKEN : LOCAL_HANDSHAKE_TOKEN;
}

function currentRendererApiConfig() {
  return {
    apiBaseUrl: resolvedApiBaseUrl,
    handshakeToken: backendAuthenticated ? activeHandshakeToken() : '',
    backendReady: backendVerified,
    hasPeerWindows: Boolean(
      mainWindow &&
        !mainWindow.isDestroyed() &&
        pianoRollWindow &&
        !pianoRollWindow.isDestroyed(),
    ),
  };
}

function broadcastApiConfig() {
  const payload = currentRendererApiConfig();
  for (const win of [mainWindow, pianoRollWindow]) {
    if (win && !win.isDestroyed()) {
      win.webContents.send('compass:api-config', payload);
    }
  }
}

async function ensureBackendAuthenticatedLazy() {
  if (backendVerified || isShuttingDown) {
    return backendVerified;
  }
  if (lazyAuthProbePromise) {
    return lazyAuthProbePromise;
  }
  lazyAuthProbePromise = (async () => {
    const probeUrl = resolvedApiBaseUrl;
    const probeExplicit = usingExplicitBackend;
    const expected = probeExplicit ? EXPLICIT_HANDSHAKE_TOKEN : LOCAL_HANDSHAKE_TOKEN;
    try {
      const probe = await probeCompassHealth(probeUrl, {
        timeoutMs: 900,
        expectedHandshake: expected,
      });
      if (!isShuttingDown && probeUrl === resolvedApiBaseUrl && probeExplicit === usingExplicitBackend) {
        if (probe.validEngine && (probe.authenticated || (probeExplicit && !probe.authRequired))) {
          const nextAuth = Boolean(probe.authRequired && probe.authenticated && expected);
          const changed = !backendVerified || backendAuthenticated !== nextAuth;
          backendAuthenticated = nextAuth;
          backendVerified = true;
          if (changed) {
            broadcastApiConfig();
            emitBootLog(
              probeExplicit
                ? `指定された推論サーバー (${probeUrl}) との接続を確認しました。`
                : 'ローカル推論エンジンとのセキュアハンドシェイクを確認しました。',
            );
          }
          return backendVerified;
        }
        const hadAuth = backendAuthenticated;
        backendAuthenticated = false;
        backendVerified = false;
        if (hadAuth) {
          broadcastApiConfig();
        }
      }
    } catch {
      if (!isShuttingDown && probeUrl === resolvedApiBaseUrl && probeExplicit === usingExplicitBackend) {
        const hadAuth = backendAuthenticated;
        backendAuthenticated = false;
        backendVerified = false;
        if (hadAuth) {
          broadcastApiConfig();
        }
      }
    } finally {
      lazyAuthProbePromise = null;
    }
    return backendVerified;
  })();
  return lazyAuthProbePromise;
}

// Enforce single instance lock unless running isolated automated tests ([HIGH-02.1])
if (!process.env.COMPASS_ELECTRON_MULTI_INSTANCE) {
  const gotLock = app.requestSingleInstanceLock();
  if (!gotLock) {
    isQuittingDuplicate = true;
    app.quit();
  } else {
    app.on('second-instance', () => {
      elevateWindowToForeground(mainWindow, 'compass-ai');
    });
  }
}

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
      // Reject NUL bytes and NTFS Alternate Data Streams (`::$DATA`) ([LOW-03])
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
      let bodyBytes = fs.readFileSync(realTarget);
      if (ext === '.html' && usingExplicitBackend && PARSED_EXPLICIT_API) {
        try {
          const explicitUrl = new URL(PARSED_EXPLICIT_API.url);
          // W3C CSP3 / Chromium `network::csp::ParseHost` excludes bracketed IPv6 literals and
          // underscore (`_`) hostnames from `host-source` (`[R4-LOW-02]`); use `scheme-source`
          // (`http:` / `https:`) for IPv6 or underscore hostnames and exact origin otherwise.
          const cspConnectToken =
            explicitUrl.hostname.startsWith('[') || explicitUrl.hostname.includes('_')
              ? explicitUrl.protocol
              : explicitUrl.origin;
          if (/^https?:(\/\/[a-zA-Z0-9.\-:]+)?$/.test(cspConnectToken)) {
            const htmlStr = bodyBytes
              .toString('utf8')
              .replace(
                "connect-src 'self' compass-app:",
                () => `connect-src 'self' compass-app: ${cspConnectToken}`,
              );
            bodyBytes = Buffer.from(htmlStr, 'utf8');
          }
        } catch {
          // Keep default loopback-only CSP
        }
      }
      return new Response(bodyBytes, {
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

/**
 * Probes `GET /health` using a per-request random nonce (`X-Compass-Challenge`) and verifies
 * `HMAC-SHA256(expectedHandshake, challenge)` in constant time (`[R2-HIGH-03]`).
 * Returns `{ reachable, validEngine, authenticated, authRequired }`.
 */
function probeCompassHealth(baseUrl, options = {}) {
  const timeoutMs = options.timeoutMs ?? 900;
  const expectedHandshake = options.expectedHandshake ?? '';
  const challenge = expectedHandshake ? crypto.randomBytes(16).toString('hex') : '';

  return new Promise((resolve) => {
    let settled = false;
    let req = null;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(wallClockTimer);
      resolve(result);
    };

    const wallClockTimer = setTimeout(() => {
      if (req) req.destroy();
      finish({ reachable: false, validEngine: false, authenticated: false, authRequired: false });
    }, timeoutMs);

    try {
      const rawBase = String(baseUrl || '').trim();
      if (!rawBase) {
        finish({ reachable: false, validEngine: false, authenticated: false, authRequired: false });
        return;
      }
      const url = new URL(rawBase);
      if (url.protocol !== 'http:' && url.protocol !== 'https:') {
        finish({ reachable: false, validEngine: false, authenticated: false, authRequired: false });
        return;
      }
      url.pathname = `${url.pathname.replace(/\/+$/, '')}/health`;
      url.search = '';
      url.hash = '';
      const transport = url.protocol === 'https:' ? https : http;
      req = transport.get(
        url,
        {
          timeout: timeoutMs,
          headers: challenge ? { 'X-Compass-Challenge': challenge } : {},
        },
        (res) => {
          let totalBytes = 0;
          const chunks = [];
          res.on('data', (chunk) => {
            totalBytes += chunk.length;
            if (totalBytes > MAX_HEALTH_BODY_BYTES) {
              req.destroy();
              finish({ reachable: true, validEngine: false, authenticated: false, authRequired: false });
              return;
            }
            chunks.push(chunk);
          });
          res.on('end', () => {
            if (res.statusCode !== 200) {
              finish({ reachable: true, validEngine: false, authenticated: false, authRequired: false });
              return;
            }
            try {
              const json = JSON.parse(Buffer.concat(chunks).toString('utf8'));
              const validEngine =
                Boolean(json) &&
                json.status === 'ok' &&
                json.engine === 'compass-mortm-4.5';
              if (!validEngine) {
                finish({ reachable: true, validEngine: false, authenticated: false, authRequired: false });
                return;
              }
              const authRequired = Boolean(json.auth_required);
              if (!expectedHandshake) {
                finish({
                  reachable: true,
                  validEngine: true,
                  authenticated: !authRequired,
                  authRequired,
                });
                return;
              }
              const expectedHmac = crypto
                .createHmac('sha256', expectedHandshake)
                .update(challenge, 'utf8')
                .digest('hex');
              const actualHmac = typeof json.handshake_hmac === 'string' ? json.handshake_hmac : '';
              const hmacOk =
                actualHmac.length === expectedHmac.length &&
                crypto.timingSafeEqual(
                  Buffer.from(actualHmac, 'utf8'),
                  Buffer.from(expectedHmac, 'utf8'),
                );
              finish({
                reachable: true,
                validEngine: true,
                authenticated: hmacOk,
                authRequired,
              });
            } catch {
              finish({ reachable: true, validEngine: false, authenticated: false, authRequired: false });
            }
          });
        },
      );
      req.on('error', () =>
        finish({ reachable: false, validEngine: false, authenticated: false, authRequired: false }),
      );
      req.on('timeout', () => {
        req.destroy();
        finish({ reachable: false, validEngine: false, authenticated: false, authRequired: false });
      });
    } catch {
      finish({ reachable: false, validEngine: false, authenticated: false, authRequired: false });
    }
  });
}

function probeTcpConnect(host, port, timeoutMs = 45) {
  return new Promise((resolve) => {
    const sock = new net.Socket();
    let done = false;
    const finish = (inUse) => {
      if (done) return;
      done = true;
      sock.destroy();
      resolve(inUse);
    };
    sock.setTimeout(timeoutMs);
    sock.once('connect', () => finish(true));
    sock.once('timeout', () => finish(false));
    sock.once('error', (err) => {
      // ECONNREFUSED / ENETUNREACH / EADDRNOTAVAIL => nothing listening on this address
      const errCode = err && 'code' in err ? err.code : '';
      if (errCode === 'EACCES' || errCode === 'EPERM') {
        finish(true);
      } else {
        finish(false);
      }
    });
    try {
      sock.connect(port, host);
    } catch {
      finish(false);
    }
  });
}

function probeBindHost(host, port) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.unref();
    srv.once('error', () => resolve(false));
    srv.listen({ port, host, exclusive: true }, () => {
      srv.close(() => resolve(true));
    });
  });
}

function probeBindIpv6Loopback(port) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.unref();
    srv.once('error', (err) => {
      const code = err && 'code' in err ? err.code : '';
      if (code === 'EAFNOSUPPORT' || code === 'EADDRNOTAVAIL' || code === 'EINVAL') {
        resolve(true);
      } else {
        resolve(false);
      }
    });
    srv.listen({ port, host: '::1', exclusive: true }, () => {
      srv.close(() => resolve(true));
    });
  });
}

async function isPortCompletelyFree(port) {
  // Detect wildcard/loopback listeners across IPv4 (`0.0.0.0`, `127.0.0.1`) and IPv6 (`[::]`, `::1`) ([HIGH-03.1], [R2-LOW-03])
  if (await probeTcpConnect('127.0.0.1', port)) return false;
  if (await probeTcpConnect('::1', port)) return false;
  if (!(await probeBindHost('0.0.0.0', port))) return false;
  if (!(await probeBindHost('127.0.0.1', port))) return false;
  if (!(await probeBindIpv6Loopback(port))) return false;
  return true;
}

const CHROMIUM_RESTRICTED_HIGH_PORTS = new Set([
  1719, 1720, 1723, 2049, 3659, 4045, 5060, 5061, 6000, 6566,
  6665, 6666, 6667, 6668, 6669, 6697, 10080,
]);

function isSafeRendererPort(port) {
  return (
    Number.isInteger(port) &&
    port >= 1024 &&
    port <= 65535 &&
    !CHROMIUM_RESTRICTED_HIGH_PORTS.has(port)
  );
}

async function findAvailablePort(preferredPort = 8000, maxAttempts = 25) {
  const basePort = isSafeRendererPort(preferredPort) ? preferredPort : 8000;
  for (let i = 0; i < maxAttempts; i += 1) {
    const candidate = basePort + i;
    if (isSafeRendererPort(candidate) && (await isPortCompletelyFree(candidate))) {
      return candidate;
    }
  }
  return new Promise((resolve, reject) => {
    const fallback = net.createServer();
    fallback.unref();
    fallback.once('error', reject);
    fallback.listen(0, '127.0.0.1', () => {
      const addr = fallback.address();
      const port = typeof addr === 'object' && addr ? addr.port : preferredPort;
      fallback.close(() => resolve(port));
    });
  });
}

function isPythonRuntimeUsableAsync(pyExe, timeoutMs = 15000) {
  return new Promise((resolve) => {
    try {
      // Fast <100ms spec discovery avoids double-importing PyTorch C++ DLLs on cold/Defender boots ([R2-MED-04])
      const probeExpr =
        "import importlib.util, sys; sys.exit(0 if (importlib.util.find_spec('torch') and importlib.util.find_spec('fastapi')) else 1)";
      const child = execFile(
        pyExe,
        ['-c', probeExpr],
        {
          windowsHide: true,
          timeout: timeoutMs,
        },
        (err) => {
          resolve(!err);
        },
      );
      child.on('error', () => resolve(false));
    } catch {
      resolve(false);
    }
  });
}

async function resolvePythonExecutableAsync() {
  const fileCandidates = [];
  if (process.env.COMPASS_PYTHON) {
    fileCandidates.push(process.env.COMPASS_PYTHON);
  }
  if (process.platform === 'win32') {
    fileCandidates.push(path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe'));
  } else {
    fileCandidates.push(path.join(REPO_ROOT, '.venv', 'bin', 'python'));
    fileCandidates.push(path.join(REPO_ROOT, '.venv', 'bin', 'python3'));
  }

  for (const candidate of fileCandidates) {
    if (candidate && fs.existsSync(candidate) && (await isPythonRuntimeUsableAsync(candidate))) {
      return candidate;
    }
  }

  const pathFallbacks = process.platform === 'win32' ? ['python'] : ['python3', 'python'];
  for (const cmd of pathFallbacks) {
    if (await isPythonRuntimeUsableAsync(cmd)) {
      return cmd;
    }
  }
  return null;
}

function emitBootLog(line) {
  const sanitized = String(line || '').trim().slice(0, 160);
  if (!sanitized) return;

  if (mainWindow && !mainWindow.isDestroyed() && mainWindowLoaded) {
    mainWindow.webContents.send('compass:boot-log', sanitized);
  } else {
    if (pendingBootLogs.length >= 40) {
      pendingBootLogs.shift();
    }
    pendingBootLogs.push(sanitized);
  }
}

function flushPendingBootLogs() {
  if (!mainWindow || mainWindow.isDestroyed() || !mainWindowLoaded) return;
  while (pendingBootLogs.length > 0) {
    const line = pendingBootLogs.shift();
    mainWindow.webContents.send('compass:boot-log', line);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function terminateChildProcessByPid(pid) {
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
    // Process already exited
  }
}

async function startPythonBackendInBackground(initialPort) {
  if (isShuttingDown) {
    return;
  }
  const skipLocalSpawn = process.env.COMPASS_SKIP_BACKEND_SPAWN === '1';

  let activePort = initialPort;

  if (usingExplicitBackend) {
    const explicitTargetUrl = resolvedApiBaseUrl;
    const explicitProbe = await probeCompassHealth(explicitTargetUrl, {
      timeoutMs: 900,
      expectedHandshake: EXPLICIT_HANDSHAKE_TOKEN,
    });
    if (
      explicitProbe.validEngine &&
      (explicitProbe.authenticated || (!explicitProbe.authRequired && !EXPLICIT_HANDSHAKE_TOKEN))
    ) {
      backendAuthenticated = Boolean(explicitProbe.authenticated && EXPLICIT_HANDSHAKE_TOKEN);
      backendVerified = true;
      broadcastApiConfig();
      emitBootLog(`指定された推論サーバー (${explicitTargetUrl}) に接続しました。`);
      return;
    }
    if (skipLocalSpawn) {
      backendAuthenticated = false;
      backendVerified = false;
      broadcastApiConfig();
      return;
    }
    usingExplicitBackend = false;
    backendAuthenticated = false;
    backendVerified = false;
    // Explicit URL is either down or occupied by an untrusted/different listener -> ensure safe free port ([R2-HIGH-03], [R6C-LOW-01], [R6D-MED-01], [R6D-MED-02])
    if (!isSafeRendererPort(activePort) || !(await isPortCompletelyFree(activePort))) {
      activePort = await findAvailablePort(8000);
    }
    resolvedApiBaseUrl = `http://127.0.0.1:${activePort}`;
    broadcastApiConfig();
  } else if (skipLocalSpawn) {
    return;
  }

  const mainScript = path.join(REPO_ROOT, 'main.py');
  if (!fs.existsSync(mainScript)) {
    emitBootLog('main.py が見つかりません — UI単体モードで起動します。');
    return;
  }

  emitBootLog('ローカル Python / PyTorch 推論ランタイムを検証しています...');
  const pyExe = await resolvePythonExecutableAsync();
  if (isShuttingDown) return;

  if (!pyExe) {
    emitBootLog('Python推論環境 (.venv) が未検出のため、UI単体デモモードで起動します。');
    return;
  }

  for (let spawnRound = 0; spawnRound < 3; spawnRound += 1) {
    if (isShuttingDown) return;
    usingExplicitBackend = false;
    backendAuthenticated = false;
    backendVerified = false;

    // Re-verify port exclusivity immediately prior to spawn to close TOCTOU window ([R2-HIGH-03])
    if (!isSafeRendererPort(activePort) || !(await isPortCompletelyFree(activePort))) {
      activePort = await findAvailablePort(activePort + 1);
      resolvedApiBaseUrl = `http://127.0.0.1:${activePort}`;
      broadcastApiConfig();
    } else if (resolvedApiBaseUrl !== `http://127.0.0.1:${activePort}`) {
      resolvedApiBaseUrl = `http://127.0.0.1:${activePort}`;
      broadcastApiConfig();
    } else {
      broadcastApiConfig();
    }
    if (isShuttingDown) return;

    const childEnv = {
      ...process.env,
      COMPASS_HOST: '127.0.0.1',
      COMPASS_PORT: String(activePort),
      COMPASS_PARENT_PID: String(process.pid),
      COMPASS_HANDSHAKE_TOKEN: LOCAL_HANDSHAKE_TOKEN,
      PYTHONIOENCODING: 'utf-8',
    };

    let spawnedChild = null;
    try {
      const child = spawn(pyExe, [mainScript], {
        cwd: REPO_ROOT,
        env: childEnv,
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });
      spawnedChild = child;
      pythonProc = child;
      spawnedByElectron = true;

      child.on('error', (err) => {
        if (pythonProc === child) {
          pythonProc = null;
          spawnedByElectron = false;
          backendAuthenticated = false;
          backendVerified = false;
          broadcastApiConfig();
        }
        const errCode = err && 'code' in err ? err.code : 'ERR';
        emitBootLog(`Python起動エラー (${errCode || 'ERR'}) — UI単体モードへ切り替えます。`);
      });

      const forwardStream = (chunk) => {
        const lines = chunk
          .toString('utf8')
          .split(/\r?\n/)
          .map((s) => s.trim())
          .filter(Boolean);
        for (const l of lines) {
          emitBootLog(l.slice(0, 140));
        }
      };

      child.stdout?.on('data', forwardStream);
      child.stderr?.on('data', forwardStream);
      child.on('exit', () => {
        if (pythonProc === child) {
          pythonProc = null;
          spawnedByElectron = false;
          backendAuthenticated = false;
          backendVerified = false;
          broadcastApiConfig();
        }
      });
    } catch {
      pythonProc = null;
      spawnedByElectron = false;
      backendAuthenticated = false;
      backendVerified = false;
      broadcastApiConfig();
      emitBootLog('推論エンジンの起動に失敗しました — UI単体モードで継続します。');
      return;
    }

    // Verify cryptographic HMAC handshake across slow cold PyTorch DLL imports (up to 90s wall-clock)
    // and detect non-HTTP TCP port squatters (`EADDRINUSE`) (`[R3-HIGH-01]`, `[R6B-MED-01]`)
    const rawDeadline = Number(process.env.COMPASS_STARTUP_PROBE_DEADLINE_MS);
    const startupDeadlineMs =
      Number.isFinite(rawDeadline) && rawDeadline > 0
        ? Math.max(1_500, rawDeadline)
        : 90_000;
    const pollStartedAt = Date.now();
    let spoofDetected = false;
    let portCollisionDetected = false;
    try {
      while (!isShuttingDown && Date.now() - pollStartedAt < startupDeadlineMs) {
        await sleep(300);
        if (isShuttingDown) return;

        const probe = await probeCompassHealth(resolvedApiBaseUrl, {
          timeoutMs: 1200,
          expectedHandshake: LOCAL_HANDSHAKE_TOKEN,
        });
        if (probe.authenticated) {
          backendAuthenticated = true;
          backendVerified = true;
          broadcastApiConfig();
          emitBootLog('ローカル推論エンジンとのセキュアハンドシェイクを確認しました。');
          return;
        }
        if (probe.reachable && !probe.authenticated) {
          spoofDetected = true;
          break;
        }
        if (!spawnedChild || pythonProc !== spawnedChild) {
          if (!isShuttingDown && !(await isPortCompletelyFree(activePort))) {
            portCollisionDetected = true;
          }
          break;
        }
      }
    } catch {
      // Non-fatal health poll error
    }

    if (!backendAuthenticated && !spoofDetected && !portCollisionDetected && !isShuttingDown) {
      if (
        (!spawnedChild || pythonProc !== spawnedChild) &&
        !(await isPortCompletelyFree(activePort))
      ) {
        portCollisionDetected = true;
      }
    }

    if (spoofDetected || portCollisionDetected) {
      const oldPid = spawnedChild ? spawnedChild.pid : null;
      if (pythonProc === spawnedChild) {
        pythonProc = null;
        spawnedByElectron = false;
      }
      terminateChildProcessByPid(oldPid);
      activePort = await findAvailablePort(activePort + 1);
      resolvedApiBaseUrl = `http://127.0.0.1:${activePort}`;
      broadcastApiConfig();
      emitBootLog(`ポート競合を検出したため空きポート (${activePort}) へ再割り当てしています...`);
      continue;
    }
    return;
  }
}

function terminatePythonBackend() {
  isShuttingDown = true;
  if (!pythonProc || !spawnedByElectron) {
    return;
  }
  const pid = pythonProc.pid;
  pythonProc = null;
  spawnedByElectron = false;
  terminateChildProcessByPid(pid);
}

function elevateWindowToForeground(win, tabToFocus) {
  if (!win || win.isDestroyed()) return;
  if (win.isMinimized()) {
    win.restore();
  }
  win.show();
  win.setAlwaysOnTop(true);
  win.focus();
  setTimeout(() => {
    if (!win.isDestroyed()) {
      win.setAlwaysOnTop(false);
    }
  }, 120);

  if (tabToFocus) {
    win.webContents.send('compass:focus-tab', tabToFocus);
  }
}

function hardenWebContents(win) {
  const allowedOrigin = getAllowedAppOrigin();

  win.webContents.setWindowOpenHandler(({ url }) => {
    try {
      const parsed = new URL(url);
      if (
        extractCanonicalOrigin(url) === allowedOrigin &&
        parsed.searchParams.get('tab') === 'pianoroll'
      ) {
        openDedicatedPianoRollWindow();
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
  win.webContents.on('will-attach-webview', (event) => {
    event.preventDefault();
  });
}

function loadAppEntry(win, query) {
  const base = process.env.COMPASS_VITE_DEV_URL || `${COMPASS_APP_ORIGIN}/index.html`;
  const u = new URL(base);
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      u.searchParams.set(k, v);
    }
  }
  void win.loadURL(u.toString());
}

function createMainWindow() {
  mainWindowLoaded = false;
  mainWindow = new BrowserWindow({
    width: 1360,
    height: 880,
    minWidth: 375,
    minHeight: 620,
    useContentSize: true,
    title: 'COMPASS AI',
    backgroundColor: '#09090B',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      additionalArguments: [`--compass-api-url=${resolvedApiBaseUrl}`],
    },
  });

  hardenWebContents(mainWindow);
  mainWindow.webContents.once('did-finish-load', () => {
    mainWindowLoaded = true;
    broadcastApiConfig();
    flushPendingBootLogs();
  });
  loadAppEntry(mainWindow);

  mainWindow.on('closed', () => {
    mainWindow = null;
    mainWindowLoaded = false;
    if (pianoRollWindow && !pianoRollWindow.isDestroyed()) {
      pianoRollWindow.close();
    }
  });
}

function openDedicatedPianoRollWindow() {
  if (pianoRollWindow && !pianoRollWindow.isDestroyed()) {
    elevateWindowToForeground(pianoRollWindow, 'pianoroll');
    return;
  }

  pianoRollWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 375,
    minHeight: 560,
    useContentSize: true,
    title: 'COMPASS AI — ピアノロール',
    backgroundColor: '#09090B',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      additionalArguments: [`--compass-api-url=${resolvedApiBaseUrl}`],
    },
  });

  hardenWebContents(pianoRollWindow);
  broadcastApiConfig();
  pianoRollWindow.webContents.once('did-finish-load', () => {
    broadcastApiConfig();
  });
  loadAppEntry(pianoRollWindow, { tab: 'pianoroll', popout: '1' });

  pianoRollWindow.on('closed', () => {
    pianoRollWindow = null;
    broadcastApiConfig();
  });
}

ipcMain.handle('compass:get-api-config', async (event, forceRefresh) => {
  if (!isTrustedIpcSender(event)) {
    return {
      apiBaseUrl: '',
      handshakeToken: '',
      backendReady: false,
      hasPeerWindows: false,
    };
  }
  if (forceRefresh === true) {
    if (lazyAuthProbePromise) {
      try {
        await lazyAuthProbePromise;
      } catch {
        // Ignore prior probe error before forcing fresh probe
      }
    }
    backendVerified = false;
  }
  if (!backendVerified && !isShuttingDown) {
    await ensureBackendAuthenticatedLazy();
  }
  return currentRendererApiConfig();
});

ipcMain.handle('compass:bring-to-front', (event, targetTab) => {
  if (!isTrustedIpcSender(event)) return;
  if (targetTab === 'pianoroll' && pianoRollWindow && !pianoRollWindow.isDestroyed()) {
    elevateWindowToForeground(pianoRollWindow, 'pianoroll');
    return;
  }
  elevateWindowToForeground(
    mainWindow,
    targetTab === 'pianoroll' ? 'pianoroll' : 'compass-ai',
  );
});

ipcMain.handle('compass:open-pianoroll-window', (event) => {
  if (!isTrustedIpcSender(event)) return;
  openDedicatedPianoRollWindow();
});

ipcMain.on('compass:workspace-sync', (event, msg) => {
  if (!isTrustedIpcSender(event)) return;
  const targets = [mainWindow, pianoRollWindow].filter(
    (win) => win && !win.isDestroyed() && win.webContents.id !== event.sender.id,
  );
  if (targets.length === 0 || !isValidWorkspaceSyncPayload(msg)) return;
  for (const win of targets) {
    win.webContents.send('compass:workspace-sync', msg);
  }
});

app
  .whenReady()
  .then(async () => {
    if (isQuittingDuplicate) return;

    session.defaultSession.setPermissionRequestHandler((_wc, _permission, callback) => {
      callback(false);
    });
    registerCompassAppProtocol();

    let selectedPort = 8000;
    if (!PARSED_EXPLICIT_API) {
      usingExplicitBackend = false;
      selectedPort = await findAvailablePort(8000);
      resolvedApiBaseUrl = `http://127.0.0.1:${selectedPort}`;
    } else {
      selectedPort = PARSED_EXPLICIT_API.port;
      resolvedApiBaseUrl = PARSED_EXPLICIT_API.url;
    }

    if (isShuttingDown || isQuittingDuplicate) return;

    // Open the main window IMMEDIATELY so the Splash Screen renders without waiting for Python probe
    createMainWindow();
    void startPythonBackendInBackground(selectedPort);
  })
  .catch((err) => {
    console.error('[COMPASS AI STARTUP ERROR]', err);
    terminatePythonBackend();
    app.exit(1);
  });

app.on('window-all-closed', () => {
  terminatePythonBackend();
  app.quit();
});

app.on('before-quit', () => {
  terminatePythonBackend();
});

app.on('will-quit', () => {
  terminatePythonBackend();
});

process.on('SIGINT', () => {
  terminatePythonBackend();
  process.exit(0);
});

process.on('SIGTERM', () => {
  terminatePythonBackend();
  process.exit(0);
});

process.on('uncaughtException', (err) => {
  console.error('[COMPASS AI UNCAUGHT EXCEPTION]', err);
  terminatePythonBackend();
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  console.error('[COMPASS AI UNHANDLED REJECTION]', reason);
  terminatePythonBackend();
  process.exit(1);
});
