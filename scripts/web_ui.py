#!/usr/bin/env python3
"""
PicoCode Local Coding Web Studio (Zero-Dependency Python Server).

Runs at http://127.0.0.1:7870 (no port conflict with PicoVid on 7860) and manages
an OpenAI-compatible llama-server on http://127.0.0.1:8080/v1.

Features:
  - Live Dedicated & Shared VRAM + System RAM telemetry bars
  - Model manager (Inkling 2 Mini, Qwen3-Coder-30B-A3B, Qwen2.5-Coder-7B) + 1-click download
  - Token-by-token streaming coding chat with reasoning fold & fenced code-block Copy buttons
  - Screenshot / UI image attach for Inkling 2 Mini Vision
  - 1-click VS Code (Continue / Cline / Roo Code) & Python OpenAI SDK configuration panel

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import download_models  # noqa: E402
import probe_hardware  # noqa: E402
import serve  # noqa: E402


class ServerManager:
    """Thread-safe lifecycle manager for llama-server and background model downloads."""

    def __init__(self, api_host: str = "127.0.0.1", api_port: int = 8080) -> None:
        self.api_host = api_host
        self.api_port = api_port
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.state: str = "stopped"  # stopped | starting | ready | error
        self.active_plan: dict[str, Any] | None = None
        self.error_msg: str = ""
        self.logs: list[str] = []
        self.dl_proc: subprocess.Popen[str] | None = None
        self.dl_profile: str | None = None
        self.dl_logs: list[str] = []
        self.hw_base: dict[str, Any] = probe_hardware.probe_hardware(REPO_ROOT / "models")

    def _append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > 120:
                self.logs = self.logs[-120:]

    def stop(self) -> None:
        with self.lock:
            proc = self.proc
            self.proc = None
            self.state = "stopped"
            self.active_plan = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                proc.kill()

    def start(
        self,
        model_id: str,
        *,
        ctx_size: int | None = None,
        flash_attn: str = "auto",
        use_draft: bool = True,
        use_vision: bool = True,
    ) -> tuple[bool, str]:
        self.stop()
        reg = serve.load_registry()
        hw = probe_hardware.probe_hardware(REPO_ROOT / "models")
        self.hw_base = hw
        exe = serve.find_llama_server(hw.get("backend", "vulkan"))
        if exe is None:
            return False, "llama-server.exe bulunamadi! Once .\\setup.cmd calistirin."
        try:
            cmd, plan = serve.build_server_command(
                model_id,
                hw,
                reg,
                server_exe=exe,
                host=self.api_host,
                port=self.api_port,
                ctx_size=ctx_size,
                flash_attn=flash_attn,
                use_draft=use_draft,
                use_vision=use_vision,
            )
        except Exception as exc:
            return False, str(exc)

        with self.lock:
            self.logs = [f"[PicoCode] Baslatiliyor: {plan['label']}", f"[PicoCode] Komut: {' '.join(cmd)}"]
            self.state = "starting"
            self.active_plan = plan
            self.error_msg = ""

        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        with self.lock:
            self.proc = proc

        def _watch() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._append_log(line.rstrip())
            code = proc.wait()
            with self.lock:
                if self.proc is proc:
                    if self.state != "stopped":
                        self.state = "error"
                        self.error_msg = f"llama-server cikis yapti (kod={code})"
                    self.proc = None

        def _health() -> None:
            ok = serve.wait_for_health(self.api_host, self.api_port, proc, timeout_s=240.0)
            with self.lock:
                if self.proc is proc:
                    if ok:
                        self.state = "ready"
                        self.logs.append(f"[PicoCode] Sunucu HAZIR -> http://{self.api_host}:{self.api_port}/v1")
                    elif self.state == "starting":
                        self.state = "error"
                        self.error_msg = "Sunucu saglik kontrolunu gecemedi. Loglara bakin."

        threading.Thread(target=_watch, daemon=True).start()
        threading.Thread(target=_health, daemon=True).start()
        return True, "Baslatiliyor..."

    def start_download(self, profile: str) -> tuple[bool, str]:
        with self.lock:
            if self.dl_proc is not None and self.dl_proc.poll() is None:
                return False, f"Zaten indiriliyor: {self.dl_profile}"
            self.dl_profile = profile
            self.dl_logs = [f"[PicoCode] Indirme basladi: {profile}"]

        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        cmd = [sys.executable, "-u", str(SCRIPT_DIR / "download_models.py"), "--install-engine", "auto", "--profile", profile]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        with self.lock:
            self.dl_proc = proc

        def _read_dl() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                clean = line.strip()
                if clean:
                    with self.lock:
                        self.dl_logs.append(clean)
                        if len(self.dl_logs) > 60:
                            self.dl_logs = self.dl_logs[-60:]
            proc.wait()
            with self.lock:
                self.dl_logs.append(f"[PicoCode] Indirme tamamlandi (kod={proc.returncode})")
                self.dl_proc = None

        threading.Thread(target=_read_dl, daemon=True).start()
        return True, f"Indirme baslatildi: {profile}"

    def get_status(self) -> dict[str, Any]:
        ded_m, shr_m = probe_hardware.probe_live_vram_mib()
        ram_tot, ram_avail = probe_hardware.probe_ram_gib()
        with self.lock:
            if self.proc is not None and self.proc.poll() is not None and self.state in ("starting", "ready"):
                self.state = "error"
                self.error_msg = f"llama-server durdu (kod={self.proc.returncode})"
            dl_running = self.dl_proc is not None and self.dl_proc.poll() is None
            return {
                "state": self.state,
                "error": self.error_msg,
                "api_base": f"http://{self.api_host}:{self.api_port}/v1",
                "plan": self.active_plan,
                "logs": self.logs[-30:],
                "download": {
                    "running": dl_running,
                    "profile": self.dl_profile,
                    "logs": self.dl_logs[-12:],
                },
                "hardware": {
                    "gpu_name": self.hw_base.get("gpu_name", "GPU"),
                    "backend": self.hw_base.get("backend", "vulkan"),
                    "vram_total_gib": self.hw_base.get("vram_total_gib", 6.0),
                    "vram_used_ded_gib": round(ded_m / 1024.0, 2) if ded_m is not None else None,
                    "vram_used_shr_gib": round(shr_m / 1024.0, 2) if shr_m is not None else None,
                    "ram_total_gib": ram_tot,
                    "ram_used_gib": round(max(0.0, (ram_tot or 0) - (ram_avail or 0)), 2) if ram_tot and ram_avail else None,
                    "engine_ready": serve.find_llama_server(self.hw_base.get("backend", "vulkan")) is not None,
                },
            }

    def get_models(self) -> list[dict[str, Any]]:
        reg = serve.load_registry()
        files_map = reg["files"]
        profile_for_model = {
            "inkling2-mini": "inkling2-mini",
            "qwen3-coder-30b": "qwen3-coder",
            "qwen25-coder-7b": "starter-7b",
        }
        items = []
        for mid, mcfg in reg["models"].items():
            primary_item = files_map[mcfg["model_file"]]
            ok, msg, rem = download_models.check_file(primary_item)
            if not ok and mcfg.get("compact_file"):
                ok_c, msg_c, rem_c = download_models.check_file(files_map[mcfg["compact_file"]])
                if ok_c:
                    ok, msg, rem = ok_c, msg_c, rem_c
            has_draft = bool(mcfg.get("draft_file") and download_models.check_file(files_map[mcfg["draft_file"]])[0])
            has_vision = bool(mcfg.get("mmproj_file") and download_models.check_file(files_map[mcfg["mmproj_file"]])[0])
            items.append({
                "id": mid,
                "label": mcfg["label"],
                "arch": mcfg["arch"],
                "installed": ok,
                "status_text": msg,
                "remaining_bytes": rem,
                "size_gib": round(primary_item["expected_bytes"] / (1024**3), 2),
                "has_draft": has_draft,
                "has_vision": has_vision,
                "download_profile": profile_for_model.get(mid, "inkling2-mini"),
                "defaults": mcfg.get("defaults", {}),
            })
        return items


HTML_PAGE = r"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PicoCode Studio — Yerel Kodlama Yapay Zekasi</title>
<style>
  :root {
    --bg: #0b0f17; --panel: #121826; --panel2: #182033; --border: #26314a;
    --text: #e6edf7; --muted: #8ea0bd; --accent: #38bdf8; --accent2: #818cf8;
    --ok: #22c55e; --warn: #f59e0b; --err: #ef4444; --code-bg: #090d14;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Segoe UI", system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 10px 18px; background: var(--panel); border-bottom: 1px solid var(--border); gap: 14px; flex-wrap: wrap; }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 16px; letter-spacing: 0.2px; }
  .pill { font-size: 11px; padding: 2px 8px; border-radius: 999px; background: rgba(56,189,248,0.14); color: var(--accent); border: 1px solid rgba(56,189,248,0.35); }
  .pill.ok { background: rgba(34,197,94,0.14); color: var(--ok); border-color: rgba(34,197,94,0.35); }
  .pill.warn { background: rgba(245,158,11,0.14); color: var(--warn); border-color: rgba(245,158,11,0.35); }
  .pill.err { background: rgba(239,68,68,0.14); color: var(--err); border-color: rgba(239,68,68,0.35); }
  .metrics { display: flex; gap: 14px; align-items: center; font-size: 12px; color: var(--muted); flex-wrap: wrap; }
  .meter { display: flex; flex-direction: column; gap: 3px; min-width: 130px; }
  .meter-bar { height: 6px; border-radius: 4px; background: #1e293b; overflow: hidden; }
  .meter-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); width: 0%; transition: width 0.35s; }
  .meter-fill.warn { background: var(--warn); }
  .meter-fill.err { background: var(--err); }
  .layout { flex: 1; display: grid; grid-template-columns: 320px 1fr; min-height: 0; }
  aside { background: var(--panel); border-right: 1px solid var(--border); padding: 14px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
  .card { background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
  .card h3 { margin: 0 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); display: flex; justify-content: space-between; align-items: center; }
  label { display: block; font-size: 12px; color: var(--muted); margin: 8px 0 4px; }
  select, input[type="number"], textarea { width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 7px; padding: 7px 9px; font-size: 13px; font-family: inherit; }
  .row { display: flex; gap: 8px; align-items: center; }
  .chk { display: flex; align-items: center; gap: 7px; font-size: 12px; margin-top: 8px; color: var(--text); cursor: pointer; }
  button { border: 1px solid var(--border); background: #1d2740; color: var(--text); padding: 7px 12px; border-radius: 7px; font-size: 12.5px; font-weight: 600; cursor: pointer; transition: 0.15s; }
  button:hover { border-color: var(--accent); background: #23304f; }
  button.primary { background: linear-gradient(135deg, #0284c7, #4f46e5); border-color: transparent; color: #fff; }
  button.primary:hover { filter: brightness(1.1); }
  button.danger { background: rgba(239,68,68,0.16); border-color: rgba(239,68,68,0.45); color: #fca5a5; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .model-row { padding: 8px; border-radius: 7px; background: rgba(11,15,23,0.65); border: 1px solid var(--border); margin-bottom: 7px; font-size: 12px; }
  .model-row.active { border-color: var(--accent); }
  .model-top { display: flex; justify-content: space-between; align-items: center; gap: 6px; margin-bottom: 4px; font-weight: 600; }
  .logbox { font-family: Consolas, monospace; font-size: 11px; background: var(--code-bg); border: 1px solid var(--border); border-radius: 7px; padding: 8px; max-height: 140px; overflow-y: auto; white-space: pre-wrap; color: #9fb3d1; }
  main { display: flex; flex-direction: column; min-height: 0; }
  .tabs { display: flex; gap: 6px; padding: 8px 16px; border-bottom: 1px solid var(--border); background: var(--panel); }
  .tab { background: transparent; border-color: transparent; color: var(--muted); }
  .tab.active { background: var(--panel2); border-color: var(--border); color: var(--accent); }
  .chat-wrap { flex: 1; overflow-y: auto; padding: 18px 22px; display: flex; flex-direction: column; gap: 14px; }
  .msg { max-width: 920px; width: 100%; border-radius: 10px; padding: 12px 15px; line-height: 1.55; font-size: 14px; }
  .msg.user { align-self: flex-end; background: #1b2a47; border: 1px solid #2d436e; max-width: 760px; }
  .msg.assistant { align-self: flex-start; background: var(--panel); border: 1px solid var(--border); }
  .msg-meta { font-size: 11px; color: var(--muted); margin-bottom: 6px; display: flex; justify-content: space-between; }
  details.think { background: rgba(129,140,248,0.08); border: 1px solid rgba(129,140,248,0.28); border-radius: 7px; padding: 6px 10px; margin-bottom: 10px; font-size: 12.5px; color: #b7c3e0; }
  details.think summary { cursor: pointer; color: var(--accent2); font-weight: 600; }
  .code-wrap { position: relative; margin: 10px 0; border-radius: 8px; overflow: hidden; border: 1px solid #26314a; background: var(--code-bg); }
  .code-hdr { display: flex; justify-content: space-between; align-items: center; padding: 5px 10px; background: #101726; font-size: 11px; color: var(--muted); border-bottom: 1px solid #1e293b; }
  .code-wrap pre { margin: 0; padding: 11px; overflow-x: auto; font-family: "Cascadia Code", Consolas, monospace; font-size: 12.8px; color: #dbeafe; }
  .composer { border-top: 1px solid var(--border); background: var(--panel); padding: 12px 18px; display: flex; flex-direction: column; gap: 8px; }
  .presets { display: flex; gap: 6px; flex-wrap: wrap; }
  .presets button { font-size: 11.5px; padding: 4px 9px; background: var(--bg); }
  .composer-row { display: flex; gap: 10px; align-items: flex-end; }
  .composer textarea { min-height: 54px; max-height: 180px; resize: vertical; }
  .ide-pane { flex: 1; overflow-y: auto; padding: 20px; display: none; max-width: 960px; }
  .img-preview { max-height: 90px; border-radius: 6px; border: 1px solid var(--border); }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } aside { max-height: 36vh; } }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>PicoCode Studio</span>
    <span class="pill">v0.1.0</span>
    <span id="srvBadge" class="pill warn">Kapali</span>
  </div>
  <div class="metrics">
    <span id="gpuName">GPU algilaniyor...</span>
    <div class="meter">
      <div id="vramLbl">VRAM: --</div>
      <div class="meter-bar"><div id="vramBar" class="meter-fill"></div></div>
    </div>
    <div class="meter">
      <div id="ramLbl">RAM: --</div>
      <div class="meter-bar"><div id="ramBar" class="meter-fill"></div></div>
    </div>
    <span id="tpsBadge" class="pill">-- tok/s</span>
  </div>
</header>

<div class="layout">
  <aside>
    <div class="card">
      <h3>Sunucu Kontrolu</h3>
      <label for="modelSel">Kodlama Modeli</label>
      <select id="modelSel"></select>
      <div class="row">
        <div style="flex:1">
          <label for="ctxInp">Baglam (Context)</label>
          <input id="ctxInp" type="number" step="2048" min="2048" max="65536" value="16384" />
        </div>
        <div style="flex:1">
          <label for="faSel">Flash Attn</label>
          <select id="faSel">
            <option value="auto" selected>Auto</option>
            <option value="on">Acik (on)</option>
            <option value="off">Kapali (off)</option>
          </select>
        </div>
      </div>
      <label class="chk"><input id="draftChk" type="checkbox" checked /> DSpark Hizlandirici (Draft)</label>
      <label class="chk"><input id="visionChk" type="checkbox" checked /> Vision (Ekran Goruntusu Okuma)</label>
      <div class="row" style="margin-top:12px;">
        <button id="startBtn" class="primary" style="flex:1" onclick="startServer()">Modeli Baslat</button>
        <button id="stopBtn" class="danger" onclick="stopServer()">Durdur</button>
      </div>
    </div>

    <div class="card">
      <h3>Kurulu Modeller <button style="padding:2px 7px;font-size:11px" onclick="refreshModels()">Yenile</button></h3>
      <div id="modelsList">Yukleniyor...</div>
      <div id="dlBox" style="display:none;margin-top:8px;">
        <div class="pill warn" id="dlStatus">Indiriliyor...</div>
        <div class="logbox" id="dlLogs" style="margin-top:6px;"></div>
      </div>
    </div>

    <div class="card">
      <h3>Sunucu Loglari</h3>
      <div id="srvLogs" class="logbox">Sunucu baslatilmadi.</div>
    </div>
  </aside>

  <main>
    <div class="tabs">
      <button class="tab active" id="tabChatBtn" onclick="switchTab('chat')">Kodlama Sohbeti</button>
      <button class="tab" id="tabIdeBtn" onclick="switchTab('ide')">VS Code / IDE Baglantisi</button>
      <button style="margin-left:auto" onclick="clearChat()">Sohbeti Temizle</button>
    </div>

    <div id="chatPane" style="flex:1;display:flex;flex-direction:column;min-height:0;">
      <div id="chatLog" class="chat-wrap">
        <div class="msg assistant">
          <div class="msg-meta"><span>PicoCode</span><span>Hazir</span></div>
          Sol panelden <b>Inkling 2 Mini</b> (veya kurulu bir kodlama modelini) baslatin. Ardindan burada dogrudan kod yazabilir, hata ekran goruntusu yapistirabilir veya <b>VS Code (Continue / Cline / Roo Code)</b> eklentilerini <code>http://127.0.0.1:8080/v1</code> adresine baglayabilirsiniz.
        </div>
      </div>

      <div class="composer">
        <div class="presets">
          <button onclick="usePreset('Bu kodda olasi buglari, race conditionlari ve performans darboğazlarini bulup duzeltilmis halini ver:\n\n')">Bug Bul</button>
          <button onclick="usePreset('Asagidaki kodu temiz, tip guvenli ve uretim kalitesinde refactor et:\n\n')">Refactor Et</button>
          <button onclick="usePreset('Bu fonksiyon icin uc durumlari (edge cases) kapsayan pytest birim testleri yaz:\n\n')">Test Yaz</button>
          <button onclick="usePreset('Bu hata mesajini adim adim Turkce acikla ve kalici cozumu goster:\n\n')">Hata Coz</button>
          <label style="margin:0;cursor:pointer;">
            <span class="pill" style="display:inline-block;padding:4px 9px;">+ Ekran Goruntusu (Vision)</span>
            <input id="imgInput" type="file" accept="image/*" style="display:none" onchange="onImageSelect(event)" />
          </label>
          <span id="imgBadge" style="display:none;align-items:center;gap:6px;">
            <img id="imgThumb" class="img-preview" />
            <button onclick="clearImage()" style="padding:2px 6px;font-size:11px">Kaldir</button>
          </span>
        </div>
        <div class="composer-row">
          <textarea id="promptBox" placeholder="Kod sorunuzu yazin (Ctrl+Enter ile gonderin)..." onkeydown="if(event.ctrlKey && event.key==='Enter'){sendChat();}"></textarea>
          <button id="sendBtn" class="primary" style="height:54px;min-width:104px" onclick="sendChat()">Gonder</button>
        </div>
      </div>
    </div>

    <div id="idePane" class="ide-pane">
      <h2>VS Code, Cline, Roo Code & OpenCode Entegrasyonu</h2>
      <p style="color:var(--muted)">PicoCode arkaplanda standart OpenAI uyumlu API (<code>http://127.0.0.1:8080/v1</code>) sunar. Bulut anahtari gerektirmez; API Key alanina <code>picocode</code> yazmaniz yeterlidir.</p>

      <div class="code-wrap">
        <div class="code-hdr"><span>Cline / Roo Code (VS Code Eklentisi)</span><button onclick="copyText('cfgCline')">Kopyala</button></div>
        <pre id="cfgCline">API Provider : OpenAI Compatible
Base URL     : http://127.0.0.1:8080/v1
API Key      : picocode
Model ID     : inkling2-mini</pre>
      </div>

      <div class="code-wrap">
        <div class="code-hdr"><span>Continue.dev (~/.continue/config.yaml)</span><button onclick="copyText('cfgContinue')">Kopyala</button></div>
        <pre id="cfgContinue">models:
  - name: PicoCode Inkling 2 Mini (Local)
    provider: openai
    model: inkling2-mini
    apiBase: http://127.0.0.1:8080/v1
    apiKey: picocode</pre>
      </div>

      <div class="code-wrap">
        <div class="code-hdr"><span>Python OpenAI SDK</span><button onclick="copyText('cfgPy')">Kopyala</button></div>
        <pre id="cfgPy">from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="picocode")
resp = client.chat.completions.create(
    model="inkling2-mini",
    messages=[{"role": "user", "content": "FastAPI async health endpoint yaz."}],
)
print(resp.choices[0].message.content)</pre>
      </div>
    </div>
  </main>
</div>

<script>
let historyMsgs = [];
let attachedImageDataUrl = null;
let currentModels = [];
let serverState = "stopped";

function switchTab(tab) {
  document.getElementById('chatPane').style.display = tab === 'chat' ? 'flex' : 'none';
  document.getElementById('idePane').style.display = tab === 'ide' ? 'block' : 'none';
  document.getElementById('tabChatBtn').classList.toggle('active', tab === 'chat');
  document.getElementById('tabIdeBtn').classList.toggle('active', tab === 'ide');
}

function copyText(id) {
  navigator.clipboard.writeText(document.getElementById(id).innerText);
}

function usePreset(text) {
  const box = document.getElementById('promptBox');
  box.value = text + box.value;
  box.focus();
}

function onImageSelect(ev) {
  const f = ev.target.files[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    attachedImageDataUrl = r.result;
    document.getElementById('imgThumb').src = attachedImageDataUrl;
    document.getElementById('imgBadge').style.display = 'inline-flex';
  };
  r.readAsDataURL(f);
}

function clearImage() {
  attachedImageDataUrl = null;
  document.getElementById('imgInput').value = '';
  document.getElementById('imgBadge').style.display = 'none';
}

function clearChat() {
  historyMsgs = [];
  document.getElementById('chatLog').innerHTML = '';
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderMarkdown(text) {
  let thinkHtml = '';
  const thinkMatch = text.match(/<think>([\s\S]*?)(<\/think>|$)/);
  if (thinkMatch) {
    thinkHtml = `<details class="think"><summary>Dusunme Sureci (Reasoning)</summary><div style="white-space:pre-wrap;margin-top:6px">${escHtml(thinkMatch[1].trim())}</div></details>`;
    text = text.replace(/<think>[\s\S]*?(<\/think>|$)/, '').trim();
  }
  const parts = text.split(/```(\w*)\n?([\s\S]*?)```/g);
  let out = thinkHtml;
  for (let i = 0; i < parts.length; i += 3) {
    const prose = escHtml(parts[i] || '').replace(/`([^`]+)`/g, '<code style="background:#0f172a;padding:1px 5px;border-radius:4px">$1</code>').replace(/\n/g, '<br>');
    out += prose;
    if (i + 2 < parts.length) {
      const lang = parts[i + 1] || 'code';
      const code = parts[i + 2] || '';
      const cid = 'cb_' + Math.random().toString(36).slice(2, 9);
      out += `<div class="code-wrap"><div class="code-hdr"><span>${escHtml(lang)}</span><button onclick="copyText('${cid}')">Kopyala</button></div><pre id="${cid}">${escHtml(code)}</pre></div>`;
    }
  }
  return out;
}

async function refreshModels() {
  const res = await fetch('/api/models');
  currentModels = await res.json();
  const sel = document.getElementById('modelSel');
  const prev = sel.value;
  sel.innerHTML = '';
  const list = document.getElementById('modelsList');
  list.innerHTML = '';
  for (const m of currentModels) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `${m.id} (${m.size_gib} GiB)${m.installed ? '' : ' [Indirilmedi]'}`;
    sel.appendChild(opt);

    const row = document.createElement('div');
    row.className = 'model-row';
    const badges = [
      m.installed ? '<span class="pill ok">Hazir</span>' : '<span class="pill warn">Eksik</span>',
      m.has_draft ? '<span class="pill">DSpark</span>' : '',
      m.has_vision ? '<span class="pill">Vision</span>' : '',
    ].join(' ');
    row.innerHTML = `
      <div class="model-top"><span>${escHtml(m.id)}</span><span>${badges}</span></div>
      <div style="color:var(--muted);margin-bottom:5px">${escHtml(m.label)}</div>
      <div class="row" style="justify-content:space-between">
        <span style="color:var(--muted)">${m.size_gib} GiB</span>
        ${m.installed ? `<button style="padding:3px 8px;font-size:11px" onclick="selectAndStart('${m.id}')">Sec & Baslat</button>`
                      : `<button style="padding:3px 8px;font-size:11px" onclick="downloadProfile('${m.download_profile}')">Indir</button>`}
      </div>`;
    list.appendChild(row);
  }
  if (prev && currentModels.some(x => x.id === prev)) sel.value = prev;
  else {
    const firstReady = currentModels.find(x => x.installed);
    if (firstReady) sel.value = firstReady.id;
  }
}

function selectAndStart(id) {
  document.getElementById('modelSel').value = id;
  startServer();
}

async function downloadProfile(profile) {
  const res = await fetch('/api/download', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({profile})
  });
  const j = await res.json();
  if (!j.ok) alert(j.message);
  pollStatus();
}

async function startServer() {
  const body = {
    model: document.getElementById('modelSel').value,
    ctx_size: parseInt(document.getElementById('ctxInp').value || '16384', 10),
    flash_attn: document.getElementById('faSel').value,
    use_draft: document.getElementById('draftChk').checked,
    use_vision: document.getElementById('visionChk').checked,
  };
  const res = await fetch('/api/start', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body),
  });
  const j = await res.json();
  if (!j.ok) alert(j.message);
  pollStatus();
}

async function stopServer() {
  await fetch('/api/stop', {method: 'POST'});
  pollStatus();
}

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const st = await res.json();
    serverState = st.state;
    const badge = document.getElementById('srvBadge');
    if (st.state === 'ready') { badge.className = 'pill ok'; badge.textContent = 'HAZIR · ' + (st.plan?.model_id || ''); }
    else if (st.state === 'starting') { badge.className = 'pill warn'; badge.textContent = 'Yukleniyor...'; }
    else if (st.state === 'error') { badge.className = 'pill err'; badge.textContent = 'Hata'; }
    else { badge.className = 'pill'; badge.textContent = 'Kapali'; }

    const hw = st.hardware || {};
    document.getElementById('gpuName').textContent = `${hw.gpu_name || 'GPU'} (${(hw.backend||'').toUpperCase()})`;
    if (hw.vram_used_ded_gib != null && hw.vram_total_gib) {
      const pct = Math.min(100, (hw.vram_used_ded_gib / hw.vram_total_gib) * 100);
      const shr = hw.vram_used_shr_gib != null ? ` · Sh:${hw.vram_used_shr_gib.toFixed(1)}` : '';
      document.getElementById('vramLbl').textContent = `VRAM: ${hw.vram_used_ded_gib.toFixed(2)}/${hw.vram_total_gib.toFixed(1)}G${shr}`;
      const bar = document.getElementById('vramBar');
      bar.style.width = pct.toFixed(0) + '%';
      bar.className = 'meter-fill' + (pct > 85 ? ' err' : (pct > 70 ? ' warn' : ''));
    }
    if (hw.ram_used_gib != null && hw.ram_total_gib) {
      const pct = Math.min(100, (hw.ram_used_gib / hw.ram_total_gib) * 100);
      document.getElementById('ramLbl').textContent = `RAM: ${hw.ram_used_gib.toFixed(1)}/${hw.ram_total_gib.toFixed(1)}G`;
      document.getElementById('ramBar').style.width = pct.toFixed(0) + '%';
    }
    if (st.logs && st.logs.length) {
      const box = document.getElementById('srvLogs');
      box.textContent = st.logs.join('\n');
      box.scrollTop = box.scrollHeight;
    }
    const dl = st.download || {};
    const dlBox = document.getElementById('dlBox');
    if (dl.running || (dl.logs && dl.logs.length)) {
      dlBox.style.display = 'block';
      document.getElementById('dlStatus').textContent = dl.running ? `Indiriliyor: ${dl.profile}` : `Son indirme: ${dl.profile}`;
      document.getElementById('dlLogs').textContent = (dl.logs || []).join('\n');
      if (!dl.running) refreshModels();
    }
  } catch (e) {}
}

async function sendChat() {
  const box = document.getElementById('promptBox');
  const text = box.value.trim();
  if (!text && !attachedImageDataUrl) return;
  if (serverState !== 'ready') {
    alert('Once sol panelden "Modeli Baslat" tusuna basin ve durumun HAZIR olmasini bekleyin.');
    return;
  }
  box.value = '';
  const chatLog = document.getElementById('chatLog');

  let userContent = text;
  let userPreview = escHtml(text).replace(/\n/g, '<br>');
  if (attachedImageDataUrl) {
    userContent = [
      {type: 'text', text: text || 'Bu ekran goruntusunu analiz et ve kod cozumunu ver.'},
      {type: 'image_url', image_url: {url: attachedImageDataUrl}},
    ];
    userPreview += `<br><img src="${attachedImageDataUrl}" class="img-preview" style="margin-top:6px"/>`;
    clearImage();
  }
  historyMsgs.push({role: 'user', content: userContent});

  const uDiv = document.createElement('div');
  uDiv.className = 'msg user';
  uDiv.innerHTML = `<div class="msg-meta"><span>Sen</span></div><div>${userPreview}</div>`;
  chatLog.appendChild(uDiv);

  const aDiv = document.createElement('div');
  aDiv.className = 'msg assistant';
  aDiv.innerHTML = `<div class="msg-meta"><span>PicoCode</span><span class="stat">Uretiliyor...</span></div><div class="body">...</div>`;
  chatLog.appendChild(aDiv);
  chatLog.scrollTop = chatLog.scrollHeight;

  const sendBtn = document.getElementById('sendBtn');
  sendBtn.disabled = true;
  let fullText = '';
  let fullReason = '';
  let tokCount = 0;
  const t0 = performance.now();

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model: document.getElementById('modelSel').value,
        messages: [
          {role: 'system', content: 'You are PicoCode, an expert senior software engineer running locally. Give concise, accurate, production-grade code and clear explanations.'},
          ...historyMsgs
        ],
        stream: true,
        temperature: 0.4,
      }),
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const ln of lines) {
        if (!ln.startsWith('data:')) continue;
        const payload = ln.slice(5).trim();
        if (!payload || payload === '[DONE]') continue;
        try {
          const ev = JSON.parse(payload);
          const delta = ev.choices?.[0]?.delta || {};
          if (delta.reasoning_content) { fullReason += delta.reasoning_content; tokCount++; }
          if (delta.content) { fullText += delta.content; tokCount++; }
          if (ev.timings?.predicted_per_second) {
            document.getElementById('tpsBadge').textContent = ev.timings.predicted_per_second.toFixed(1) + ' tok/s';
          } else {
            const elapsed = (performance.now() - t0) / 1000;
            if (elapsed > 0.3 && tokCount > 0) {
              document.getElementById('tpsBadge').textContent = (tokCount / elapsed).toFixed(1) + ' tok/s';
            }
          }
          const combined = (fullReason ? `<think>${fullReason}</think>\n` : '') + fullText;
          aDiv.querySelector('.body').innerHTML = renderMarkdown(combined);
          chatLog.scrollTop = chatLog.scrollHeight;
        } catch (e) {}
      }
    }
    const totalS = ((performance.now() - t0) / 1000).toFixed(1);
    aDiv.querySelector('.stat').textContent = `${tokCount} token · ${totalS}s`;
    historyMsgs.push({role: 'assistant', content: fullText});
  } catch (err) {
    aDiv.querySelector('.body').textContent = 'Hata: ' + err;
  } finally {
    sendBtn.disabled = false;
  }
}

refreshModels();
pollStatus();
setInterval(pollStatus, 2500);
</script>
</body>
</html>
"""


def make_handler(mgr: ServerManager) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _send_json(self, data: Any, status: int = 200) -> None:
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                raw = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if self.path == "/api/status":
                self._send_json(mgr.get_status())
                return
            if self.path == "/api/models":
                self._send_json(mgr.get_models())
                return
            self._send_json({"error": "Not found"}, status=404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                body = {}

            if self.path == "/api/start":
                ok, msg = mgr.start(
                    str(body.get("model") or "inkling2-mini"),
                    ctx_size=body.get("ctx_size"),
                    flash_attn=str(body.get("flash_attn") or "auto"),
                    use_draft=bool(body.get("use_draft", True)),
                    use_vision=bool(body.get("use_vision", True)),
                )
                self._send_json({"ok": ok, "message": msg}, status=200 if ok else 400)
                return

            if self.path == "/api/stop":
                mgr.stop()
                self._send_json({"ok": True})
                return

            if self.path == "/api/download":
                ok, msg = mgr.start_download(str(body.get("profile") or "inkling2-mini"))
                self._send_json({"ok": ok, "message": msg}, status=200 if ok else 400)
                return

            if self.path == "/api/chat":
                # Stream SSE directly from llama-server /v1/chat/completions
                target = f"http://{mgr.api_host}:{mgr.api_port}/v1/chat/completions"
                req = urllib.request.Request(
                    target,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        while True:
                            chunk = resp.readline()
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=502)
                return

            self._send_json({"error": "Not found"}, status=404)

    return Handler


def main() -> int:
    serve.configure_utf8()
    parser = argparse.ArgumentParser(description="PicoCode Web Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--api-port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--auto-start", default=None, help="Automatically start model ID on launch")
    args = parser.parse_args()

    mgr = ServerManager(api_host=args.host, api_port=args.api_port)
    if args.auto_start:
        mgr.start(args.auto_start)

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(mgr))
    url = f"http://{args.host}:{args.port}"
    print(f"============================================================")
    print(f"  PicoCode Web Studio -> {url}")
    print(f"  OpenAI API Hedefi   -> http://{args.host}:{args.api_port}/v1")
    print(f"  Kapatmak icin Ctrl+C")
    print(f"============================================================")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
