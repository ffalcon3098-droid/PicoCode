#!/usr/bin/env python3
"""
PicoCode v0.4.0 — Claude-Code-Style Local Terminal Coding Agent CLI.

Runs an autonomous 16-step local coding agent directly in your terminal backed by
PicoCode's OpenAI-compatible `llama-server` (Qwen3-Coder-30B-A3B up to 30 tok/s,
Inkling 2 Mini, or Qwen2.5-Coder-7B).

Features:
  - 24-bit TrueColor Claude-Code / OpenCode-style TUI (`scripts/tui.py`):
      * Rounded 2-column welcome header with gradient PICOCODE logo
      * Interactive `❯` prompt with live `/` slash-command & `@file` dropdown autocomplete,
        persistent history, and bottom status toolbar (mode · model · git · ctx%)
      * Animated braille spinner during startup & thinking + live syntax-highlighted Markdown
      * Clean `● Tool(args)` / `⎿ summary` tree & colorized Diff Cards
      * Interactive `[e] Çalıştır · [h] Hep izin ver · [Enter] İptal` command approval card
  - 9 safe workspace tools: list_files, read_file, search_code, repo_map (AST outline),
    git_status, git_diff, edit_file, write_file, run_command
  - Path-traversal guard, ambiguous-edit rejection, instant Python/JSON syntax check,
    destructive shell command blocker, `@file` context injection, and `/undo` rollback
  - Ctrl+C cleanly cancels only the active turn without quitting the session

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tui import TerminalUI, init_windows_console  # noqa: E402

VERSION = "0.4.0"

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "models", "engine", "build", "dist", ".pytest_cache", ".mypy_cache",
}
TEXT_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".json", ".md",
    ".txt", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".cmd", ".bat", ".ps1",
    ".sh", ".sql", ".csv", ".xml", ".c", ".cpp", ".h", ".hpp", ".rs", ".go",
}
CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx"}

_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r", re.I),
    re.compile(r"\bremove-item\b[^\n]*-recurse[^\n]*-force|\bremove-item\b[^\n]*-force[^\n]*-recurse", re.I),
    re.compile(r"\b(del|erase|rmdir|rd)\b[^\n]*(/[sqf]\s*){2,}", re.I),
    re.compile(r"\bgit\s+reset\s+--hard\b|\bgit\s+clean\s+-[a-z]*f|\bgit\s+push\s+[^\n]*--force\b", re.I),
    re.compile(r"\b(invoke-expression|iex)\b|\|\s*(sh|bash|powershell|pwsh)\b", re.I),
    re.compile(r"\b(format-volume|diskpart|shutdown|stop-computer|reg\s+delete)\b", re.I),
]


def is_dangerous_command(cmd: str) -> str | None:
    for pat in _DANGEROUS_PATTERNS:
        if pat.search(cmd or ""):
            return f"Blocked destructive command matching `{pat.pattern[:48]}`."
    return None


def check_syntax_quick(rel_path: str, content: str) -> list[str]:
    ext = pathlib.Path(rel_path).suffix.lower()
    if ext == ".py":
        try:
            ast.parse(content, filename=rel_path)
            return []
        except SyntaxError as exc:
            return [f"Line {exc.lineno or 1}: {exc.msg}"]
    if ext == ".json":
        try:
            json.loads(content)
            return []
        except json.JSONDecodeError as exc:
            return [f"Line {exc.lineno}: {exc.msg}"]
    return []


def extract_code_symbols(rel_path: str, content: str) -> list[str]:
    ext = pathlib.Path(rel_path).suffix.lower()
    out: list[str] = []
    if ext == ".py":
        try:
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    out.append(f"  L{node.lineno} class {node.name}")
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            out.append(f"    L{item.lineno} def {node.name}.{item.name}()")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(f"  L{node.lineno} def {node.name}()")
            return out
        except SyntaxError:
            pass
    for idx, ln in enumerate(content.splitlines(), 1):
        m = re.match(r"^\s*(?:export\s+)?(?:async\s+)?(function|class|interface)\s+([A-Za-z_$][\w$]*)", ln)
        if m:
            out.append(f"  L{idx} {m.group(1)} {m.group(2)}")
            continue
        m2 = re.match(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", ln)
        if m2:
            out.append(f"  L{idx} fn {m2.group(1)}")
    return out


class AgentWorkspace:
    """Path-safe local workspace with undo history for the terminal coding agent."""

    def __init__(self, root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Workspace not found: {self.root}")
        self._undo_store: dict[str, dict[str, Any]] = {}
        self.turn_history: list[list[str]] = []

    def safe_path(self, rel_path: str) -> pathlib.Path:
        raw = (rel_path or "").strip()
        cand = pathlib.Path(raw)
        if cand.is_absolute():
            target = cand.resolve()
        else:
            target = (self.root / raw.replace("\\", "/").lstrip("/")).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"Access outside workspace blocked: {rel_path}") from exc
        return target

    def walk_files(self, max_depth: int = 4, max_items: int = 350) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []

        def _walk(cur: pathlib.Path, depth: int) -> None:
            if depth > max_depth or len(out) >= max_items:
                return
            try:
                entries = sorted(cur.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return
            for e in entries:
                if len(out) >= max_items:
                    return
                if e.is_dir():
                    if e.name in IGNORE_DIRS or e.name.startswith("."):
                        continue
                    _walk(e, depth + 1)
                else:
                    try:
                        sz = e.stat().st_size
                    except OSError:
                        sz = 0
                    out.append((e.relative_to(self.root).as_posix(), sz))

        _walk(self.root, 0)
        return out

    def write_with_undo(self, rel_path: str, content: str) -> dict[str, Any]:
        p = self.safe_path(rel_path)
        existed = p.exists() and p.is_file()
        old_text = p.read_text(encoding="utf-8", errors="replace") if existed else ""
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        rel = p.relative_to(self.root).as_posix()
        uid = uuid.uuid4().hex[:10]
        self._undo_store[uid] = {
            "abs_path": str(p),
            "rel_path": rel,
            "existed": existed,
            "old_content": old_text,
        }
        diff_lines = list(difflib.unified_diff(
            old_text.splitlines(), content.splitlines(),
            fromfile=f"a/{rel}", tofile=f"b/{rel}", n=2, lineterm="",
        ))
        added = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))
        return {"path": rel, "undo_id": uid, "added": added, "removed": removed, "diff": "\n".join(diff_lines[:60])}

    def undo_turn(self, undo_ids: list[str] | None = None) -> int:
        ids = undo_ids if undo_ids is not None else (self.turn_history.pop() if self.turn_history else [])
        count = 0
        for uid in reversed(ids):
            rec = self._undo_store.pop(uid, None)
            if not rec:
                continue
            p = pathlib.Path(rec["abs_path"])
            if rec["existed"]:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(rec["old_content"], encoding="utf-8")
            elif p.exists():
                p.unlink()
            count += 1
        return count


AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List entries in subdir ('' for workspace root).",
            "parameters": {"type": "object", "properties": {"subdir": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file with 1-based start_line and end_line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Case-insensitive substring search across workspace source files.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_map",
            "description": "Get compact AST/symbol outline (classes, functions, methods) of workspace code files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {"type": "string"},
                    "max_files": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Check git branch and modified/untracked files.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Inspect unified git diff for a path (empty for all).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "staged": {"type": "boolean"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact unique target block in an existing file. Runs instant syntax check.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "target": {"type": "string"},
                    "replacement": {"type": "string"},
                },
                "required": ["path", "target", "replacement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with full content. Runs instant syntax check.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in workspace root (PYTHONPATH and PYTHONDONTWRITEBYTECODE set).",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    },
]

READONLY_TOOLS = {"list_files", "read_file", "search_code", "repo_map", "git_status", "git_diff"}
READONLY_TOOL_SPECS = [t for t in AGENT_TOOLS if t["function"]["name"] in READONLY_TOOLS]


def _run_git(ws: AgentWorkspace, args: list[str]) -> tuple[int, str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        res = subprocess.run(
            ["git", *args], cwd=str(ws.root), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=10, creationflags=flags,
        )
        return res.returncode, ((res.stdout or "") + ("\n" + res.stderr if res.stderr else "")).strip()
    except Exception as exc:
        return 1, str(exc)


def get_git_summary(ws: AgentWorkspace) -> tuple[str, int]:
    rc, out = _run_git(ws, ["status", "--porcelain=v1", "-b"])
    if rc != 0 or not out:
        return "", 0
    lines = out.splitlines()
    branch = ""
    if lines and lines[0].startswith("## "):
        m = re.match(r"^## (?:No commits yet on )?([^\s.]+)", lines[0])
        if m:
            branch = m.group(1)
    changes = sum(1 for ln in lines[1:] if ln.strip())
    return branch, changes


def extract_at_files(ws: AgentWorkspace, text: str, max_files: int = 3) -> str:
    """Resolve safe `@path/to/file` mentions in user input and return formatted context snippets."""
    refs = re.findall(r"@([A-Za-z0-9_.\-/\\]+\.[A-Za-z0-9_]+)", text or "")
    if not refs:
        return ""
    blocks: list[str] = []
    seen: set[str] = set()
    for raw_ref in refs:
        rel = raw_ref.replace("\\", "/").lstrip("/")
        if rel in seen:
            continue
        seen.add(rel)
        try:
            p = ws.safe_path(rel)
            if p.is_file() and p.stat().st_size <= 256 * 1024:
                content = p.read_text(encoding="utf-8", errors="replace")[:3200]
                blocks.append(f"\nReferenced file `@{p.relative_to(ws.root).as_posix()}`:\n```\n{content}\n```")
        except Exception:
            continue
        if len(blocks) >= max_files:
            break
    return "\n".join(blocks)


def estimate_tokens(conv: list[dict[str, Any]]) -> int:
    chars = 0
    for m in conv:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        tcs = m.get("tool_calls")
        if tcs:
            chars += len(json.dumps(tcs, ensure_ascii=False))
    return max(1, chars // 4)


def execute_cli_tool(
    ws: AgentWorkspace,
    name: str,
    args: dict[str, Any],
    *,
    auto_approve: bool = False,
    confirm_fn: Any | None = None,
) -> tuple[str, dict[str, Any] | None]:
    try:
        if name == "list_files":
            sub = str(args.get("subdir") or "").strip()
            base = ws.safe_path(sub) if sub else ws.root
            if not base.is_dir():
                return f"ERROR: Directory not found: {sub}", None
            lines = []
            for e in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:120]:
                if e.name in IGNORE_DIRS:
                    continue
                rel = e.relative_to(ws.root).as_posix()
                lines.append(f"{rel}/" if e.is_dir() else f"{rel} ({e.stat().st_size} B)")
            return "\n".join(lines) or "(empty directory)", None

        if name == "read_file":
            rel = str(args.get("path") or "")
            p = ws.safe_path(rel)
            if not p.is_file():
                return f"ERROR: File not found: {rel}", None
            all_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            s = max(1, int(args.get("start_line") or 1))
            e = min(len(all_lines), int(args.get("end_line") or (s + 199)))
            numbered = [f"{i}: {all_lines[i - 1]}" for i in range(s, e + 1)]
            return f"File: {p.relative_to(ws.root).as_posix()} (lines {s}-{e} of {len(all_lines)})\n" + "\n".join(numbered)[:9000], None

        if name == "search_code":
            q = str(args.get("query") or "").strip().lower()
            if not q:
                return "ERROR: Empty query.", None
            hits: list[str] = []
            for rel, sz in ws.walk_files():
                if sz > 512 * 1024 or pathlib.Path(rel).suffix.lower() not in TEXT_EXTS:
                    continue
                try:
                    for idx, ln in enumerate((ws.root / rel).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if q in ln.lower():
                            hits.append(f"{rel}:{idx}: {ln.strip()[:160]}")
                            if len(hits) >= 35:
                                return "\n".join(hits), None
                except OSError:
                    continue
            return "\n".join(hits) if hits else f"No matches for {q!r}.", None

        if name == "repo_map":
            sub = str(args.get("subdir") or "").strip().replace("\\", "/").strip("/")
            if sub:
                ws.safe_path(sub)
            limit = max(1, min(50, int(args.get("max_files") or 25)))
            blocks: list[str] = []
            total = 0
            for rel, sz in ws.walk_files():
                if sz > 200_000 or pathlib.Path(rel).suffix.lower() not in CODE_EXTS:
                    continue
                if sub and not (rel == sub or rel.startswith(sub + "/")):
                    continue
                try:
                    txt = (ws.root / rel).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                syms = extract_code_symbols(rel, txt)
                blk = f"{rel} ({txt.count(chr(10)) + 1}L):\n" + ("\n".join(syms[:30]) if syms else "  (no top-level symbols)")
                if total + len(blk) > 7500:
                    blocks.append("... (truncated)")
                    break
                blocks.append(blk)
                total += len(blk)
                if len(blocks) >= limit:
                    break
            return (f"Repo Map ({len(blocks)} files):\n" + "\n".join(blocks)) if blocks else "No code files found.", None

        if name == "git_status":
            rc, out = _run_git(ws, ["status", "--short", "--branch"])
            return out or ("Working tree clean." if rc == 0 else "Not a git repo."), None

        if name == "git_diff":
            rel = str(args.get("path") or "").strip()
            staged = bool(args.get("staged", False))
            cmd = ["diff", "--no-color"] + (["--cached"] if staged else [])
            if rel:
                p = ws.safe_path(rel)
                cmd += ["--", p.relative_to(ws.root).as_posix()]
            _, out = _run_git(ws, cmd)
            return out[:7000] if out else "No diff.", None

        if name == "edit_file":
            rel = str(args.get("path") or "")
            target = str(args.get("target") or "")
            replacement = str(args.get("replacement") or "")
            p = ws.safe_path(rel)
            if not p.is_file():
                return f"ERROR: File not found: {rel}", None
            old = p.read_text(encoding="utf-8", errors="replace")
            if not target or target not in old:
                return f"ERROR: Target text not found in {rel}. Read the file first.", None
            occ = old.count(target)
            if occ > 1:
                return f"ERROR: Ambiguous edit — target appears {occ} times in {rel}. Include surrounding lines.", None
            new = old.replace(target, replacement, 1)
            res = ws.write_with_undo(rel, new)
            syn = check_syntax_quick(res["path"], new)
            if syn:
                res["syntax_error"] = "; ".join(syn)
            syn_txt = ("\n[SYNTAX ERROR] " + "; ".join(syn) + "\nFix immediately.") if syn else ""
            return f"OK: Edited {res['path']} (+{res['added']}/-{res['removed']}).{syn_txt}", res

        if name == "write_file":
            rel = str(args.get("path") or "")
            content = str(args.get("content") or "")
            res = ws.write_with_undo(rel, content)
            syn = check_syntax_quick(res["path"], content)
            if syn:
                res["syntax_error"] = "; ".join(syn)
            syn_txt = ("\n[SYNTAX ERROR] " + "; ".join(syn) + "\nFix immediately.") if syn else ""
            return f"OK: Wrote {res['path']} (+{res['added']}/-{res['removed']}).{syn_txt}", res

        if name == "run_command":
            cmd = str(args.get("command") or "").strip()
            if not cmd:
                return "ERROR: Empty command.", None
            danger = is_dangerous_command(cmd)
            if danger:
                return f"ERROR: {danger}", None
            if not auto_approve:
                approved = False
                if confirm_fn is not None:
                    approved = bool(confirm_fn(cmd))
                elif sys.stdin.isatty():
                    ans = input(f"  Run `{cmd}`? [y/N]: ").strip().lower()
                    approved = ans in ("y", "yes", "e", "evet")
                if not approved:
                    return "ERROR: User declined running this command.", None
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ws.root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            shell_cmd = ["powershell", "-NoProfile", "-Command", cmd] if os.name == "nt" else ["sh", "-lc", cmd]
            res = subprocess.run(
                shell_cmd, cwd=str(ws.root), env=env, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=45, creationflags=flags,
            )
            out = ((res.stdout or "") + ("\n" + res.stderr if res.stderr else "")).strip()
            return f"exit_code={res.returncode}\n{out[:5000]}", None

        return f"ERROR: Unknown tool {name}", None
    except Exception as exc:
        return f"ERROR: {exc}", None


def compact_messages(conv: list[dict[str, Any]], keep_recent_tools: int = 6) -> list[dict[str, Any]]:
    tool_idxs = [i for i, m in enumerate(conv) if m.get("role") == "tool"]
    if len(tool_idxs) <= keep_recent_tools:
        return conv
    cutoff = tool_idxs[-keep_recent_tools]
    out: list[dict[str, Any]] = []
    for i, m in enumerate(conv):
        if i < cutoff and m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 520:
            lines = m["content"].splitlines()
            head = "\n".join(lines[:3])[:200]
            tail = "\n".join(lines[-2:])[:120] if len(lines) > 5 else ""
            out.append({**m, "content": f"[Compacted: {len(m['content'])} chars]\n{head}" + (f"\n...\n{tail}" if tail else "")})
        else:
            out.append(m)
    return out


def server_is_healthy(base_url: str, timeout: float = 2.0) -> tuple[bool, str]:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with urllib.request.urlopen(f"{root}/health", timeout=timeout) as r:
            if r.status == 200:
                return True, "ok"
    except Exception:
        pass
    try:
        with urllib.request.urlopen(f"{root}/v1/models", timeout=timeout) as r:
            if r.status == 200:
                return True, "ok"
    except Exception as exc:
        return False, str(exc)
    return False, "unreachable"


def _summarize_tool_output(name: str, res_txt: str) -> str:
    if not res_txt:
        return "OK"
    lines = res_txt.splitlines()
    if name == "read_file" and lines:
        return lines[0]
    if name == "repo_map" and lines:
        return lines[0].rstrip(":") + f" ({max(0, len(lines) - 1)} satır sembol)"
    if name == "list_files":
        return f"{len(lines)} öğe listelendi"
    if name == "search_code":
        return lines[0] if lines[0].startswith("No matches") else f"{len(lines)} eşleşme bulundu"
    if name == "run_command":
        rc_ln = lines[0] if lines else "exit_code=0"
        tail = lines[-1][:90] if len(lines) > 1 else ""
        return f"{rc_ln}" + (f" · {tail}" if tail else "")
    return lines[0][:120]


def run_agent_turn(
    base_url: str,
    model: str,
    ws: AgentWorkspace,
    conv: list[dict[str, Any]],
    *,
    plan_only: bool = False,
    auto_approve: bool = False,
    max_steps: int = 16,
    ctx_limit: int = 32768,
    quiet: bool = False,
    ui: TerminalUI | None = None,
) -> dict[str, Any]:
    """Run one complete multi-step tool-calling turn with Claude-Code-style live UI."""
    url = base_url.rstrip("/") + ("/chat/completions" if base_url.rstrip("/").endswith("/v1") else "/v1/chat/completions")
    tools_spec = READONLY_TOOL_SPECS if plan_only else AGENT_TOOLS
    turn_undos: list[str] = []
    tools_called: list[str] = []
    recent_sigs: list[str] = []
    full_answer: list[str] = []
    t_start = time.perf_counter()
    tok_count = 0
    session_auto_approve = auto_approve
    cancelled = False

    def _confirm_cmd(cmd: str) -> bool:
        nonlocal session_auto_approve
        if session_auto_approve:
            return True
        if ui is not None:
            choice = ui.confirm_command(cmd)
            if choice == "a":
                session_auto_approve = True
                ui.print_ok("Bu oturumda güvenli komutlar otomatik onaylanacak.")
                return True
            return choice == "y"
        return False

    try:
        for step in range(1, max_steps + 1):
            conv[:] = compact_messages(conv)
            payload = {
                "model": model,
                "messages": conv,
                "tools": tools_spec,
                "tool_choice": "auto",
                "temperature": 0.25,
                "max_tokens": 4096,
                "stream": True,
            }
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer picocode"},
            )
            step_text: list[str] = []
            tc_map: dict[int, dict[str, Any]] = {}
            step_t0 = time.perf_counter()

            live = (ui.make_live_stream() if (ui is not None and not quiet) else None)
            if live is not None:
                live.__enter__()
                live.update(ui.render_spinner(f"Düşünüyor... (adım {step}/{max_steps})"))
            try:
                with urllib.request.urlopen(req, timeout=360) as resp:
                    last_draw = 0.0
                    for raw in resp:
                        ln = raw.decode("utf-8", errors="replace").strip()
                        if not ln.startswith("data:"):
                            continue
                        ds = ln[5:].strip()
                        if not ds or ds == "[DONE]":
                            continue
                        ev = json.loads(ds)
                        ch = ev.get("choices") or []
                        if not ch:
                            continue
                        delta = ch[0].get("delta") or {}
                        if delta.get("reasoning_content") and live is not None and not step_text:
                            now = time.perf_counter()
                            if now - last_draw > 0.12:
                                live.update(ui.render_spinner(f"Akıl yürütüyor... {now - step_t0:.1f}s"))
                                last_draw = now
                        if delta.get("content"):
                            c_txt = delta["content"]
                            step_text.append(c_txt)
                            full_answer.append(c_txt)
                            tok_count += 1
                            if not quiet:
                                if live is not None:
                                    now = time.perf_counter()
                                    if now - last_draw > 0.06:
                                        live.update(ui.render_markdown("".join(step_text)))
                                        last_draw = now
                                else:
                                    sys.stdout.write(c_txt)
                                    sys.stdout.flush()
                        for tc in delta.get("tool_calls") or []:
                            idx = int(tc.get("index", 0))
                            ent = tc_map.setdefault(
                                idx,
                                {"id": tc.get("id") or f"call_{step}_{idx}", "type": "function", "function": {"name": "", "arguments": ""}},
                            )
                            if tc.get("id"):
                                ent["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                ent["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                ent["function"]["arguments"] += fn["arguments"]
                            if live is not None and not step_text:
                                fn_disp = ent["function"]["name"] or "araç"
                                live.update(ui.render_spinner(f"Araç hazırlanıyor: {fn_disp}..."))
            finally:
                if live is not None:
                    if step_text:
                        live.update(ui.render_markdown("".join(step_text)))
                    else:
                        live.update("")
                    live.__exit__(None, None, None)
                elif step_text and not quiet:
                    print()

            if not tc_map:
                conv.append({"role": "assistant", "content": "".join(step_text)})
                break

            tool_calls = [tc_map[k] for k in sorted(tc_map)]
            conv.append({"role": "assistant", "content": "".join(step_text) or None, "tool_calls": tool_calls})

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                try:
                    parsed = json.loads(raw_args) if raw_args else {}
                    parse_err = None
                except Exception as exc:
                    parsed = {}
                    parse_err = str(exc)

                sig = f"{fn_name}:{json.dumps(parsed, sort_keys=True, ensure_ascii=False)}"
                is_dup = (not parse_err) and len(recent_sigs) >= 2 and recent_sigs[-1] == sig and recent_sigs[-2] == sig
                recent_sigs.append(sig)
                if len(recent_sigs) > 6:
                    recent_sigs.pop(0)

                if parse_err:
                    res_txt, chg = f"ERROR: Invalid JSON arguments ({parse_err}).", None
                elif plan_only and fn_name not in READONLY_TOOLS:
                    res_txt, chg = f"ERROR: Tool '{fn_name}' blocked in read-only /plan mode.", None
                elif is_dup:
                    res_txt, chg = f"ERROR: Repeated identical call to '{fn_name}' 3 times. Try another approach or finish.", None
                else:
                    tools_called.append(fn_name)
                    if ui is not None and not quiet:
                        ui.print_tool_start(fn_name, parsed)
                    res_txt, chg = execute_cli_tool(
                        ws, fn_name, parsed,
                        auto_approve=session_auto_approve,
                        confirm_fn=_confirm_cmd if ui is not None else None,
                    )

                if chg:
                    turn_undos.append(chg["undo_id"])
                    if ui is not None and not quiet:
                        ui.print_diff_card(
                            chg["path"], chg["added"], chg["removed"],
                            chg.get("diff", ""), syntax_err=chg.get("syntax_error", ""),
                        )
                elif ui is not None and not quiet:
                    summary = _summarize_tool_output(fn_name, res_txt)
                    ui.print_tool_result(summary, is_error=res_txt.startswith("ERROR:"))

                conv.append({"role": "tool", "tool_call_id": tc["id"], "content": res_txt})
    except KeyboardInterrupt:
        cancelled = True
        if ui is not None and not quiet:
            ui.print_warn("İptal edildi (Ctrl+C).")

    if turn_undos:
        ws.turn_history.append(turn_undos)

    elapsed = max(0.05, time.perf_counter() - t_start)
    if ui is not None and not quiet and not cancelled:
        ui.print_turn_footer(
            elapsed=elapsed,
            chunks=tok_count,
            tools=len(tools_called),
            edits=len(turn_undos),
            ctx_tokens=estimate_tokens(conv),
            ctx_limit=ctx_limit,
        )

    return {
        "answer": "".join(full_answer).strip(),
        "tools_called": tools_called,
        "undo_ids": turn_undos,
        "elapsed": elapsed,
        "auto_approve": session_auto_approve,
        "cancelled": cancelled,
    }


def _build_system_prompt(ws: AgentWorkspace, plan_only: bool) -> str:
    root_entries = []
    try:
        for e in sorted(ws.root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:25]:
            if e.name in IGNORE_DIRS:
                continue
            root_entries.append(e.name + ("/" if e.is_dir() else ""))
    except OSError:
        pass
    rules = ""
    for rf in (".picostudio.md", "AGENTS.md"):
        p = ws.root / rf
        if p.is_file():
            rules = f"\nProject rules ({rf}):\n{p.read_text(encoding='utf-8', errors='replace')[:1000]}\n"
            break
    greeting_rule = (
        " For simple greetings (e.g. 'hi', 'hello', 'merhaba', 'selam') or general chat that does not ask to inspect/edit code, "
        "reply briefly in 1-2 sentences WITHOUT calling any tools."
    )
    if plan_only:
        return (
            f"You are PicoCode Terminal Planning Agent. Workspace: `{ws.root.name}` (root: {', '.join(root_entries)}). "
            "READ-ONLY PLAN MODE: do not modify files. Use `repo_map`, `read_file`, `search_code`, `git_status`, `git_diff` "
            "when needed and produce a concise numbered plan in the user's language." + greeting_rule + rules
        )
    return (
        f"You are PicoCode Terminal Coding Agent. Workspace: `{ws.root.name}` (root: {', '.join(root_entries)}). "
        "When asked about the codebase or asked to make changes, use `repo_map` or `search_code` to inspect quickly, "
        "`edit_file` for targeted changes, `write_file` for new files, and `run_command` to run tests. "
        "Fix any `[SYNTAX ERROR]` immediately. Be concise; reply in the user's language." + greeting_rule + rules
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"PicoCode v{VERSION} — Claude-Code-Style Local Terminal Coding Agent CLI")
    p.add_argument("prompt", nargs="?", default=None, help="Optional one-shot task prompt (omit for interactive REPL).")
    p.add_argument("--dir", "-d", default=".", help="Target workspace directory (default: current directory).")
    p.add_argument("--model", "-m", default="qwen3-coder-30b", help="Model ID (default: qwen3-coder-30b).")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible server base URL.")
    p.add_argument("--plan", action="store_true", help="Start in read-only planning mode.")
    p.add_argument("--yes", "-y", action="store_true", help="Auto-approve safe shell commands.")
    p.add_argument("--max-steps", type=int, default=16, help="Maximum tool-calling steps per turn (default: 16).")
    p.add_argument("--no-auto-start", action="store_true", help="Do not auto-start local llama-server if unreachable.")
    return p


_MODEL_ALIASES: dict[str, str] = {
    "qwen": "qwen3-coder-30b",
    "qwen3": "qwen3-coder-30b",
    "30b": "qwen3-coder-30b",
    "inkling": "inkling2-mini",
    "inkling2": "inkling2-mini",
    "mini": "inkling2-mini",
    "7b": "qwen25-coder-7b",
    "qwen25": "qwen25-coder-7b",
}

_MODEL_SHORT_DESC: dict[str, str] = {
    "qwen3-coder-30b": "Hızlı (18-30 tok/s · 32K · Hibrit GPU MoE)",
    "inkling2-mini": "En Zeki (~%79 SWE-bench · Vision + DSpark)",
    "qwen25-coder-7b": "Hafif (%100 GPU · 16 GB RAM uyumlu)",
}


def list_available_models() -> list[dict[str, Any]]:
    """Read PicoCode registry and detect which GGUF models are installed locally."""
    reg_path = REPO_ROOT / "configs" / "models_registry.json"
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    files = reg.get("files") or {}
    out: list[dict[str, Any]] = []
    for mid, mcfg in (reg.get("models") or {}).items():
        mf_key = mcfg.get("model_file")
        cf_key = mcfg.get("compact_file")
        installed = False
        for k in (mf_key, cf_key):
            if k and k in files and (REPO_ROOT / files[k]["path"]).is_file():
                installed = True
                break
        ctx_lim = int((mcfg.get("defaults") or {}).get("ctx_size_6gb", 16384))
        out.append({
            "id": mid,
            "label": str(mcfg.get("label") or mid),
            "short": _MODEL_SHORT_DESC.get(mid, str(mcfg.get("label") or mid)),
            "installed": installed,
            "ctx_limit": ctx_lim,
        })
    return out


def resolve_model_alias(raw: str) -> str:
    clean = (raw or "").strip().lower()
    if clean in _MODEL_ALIASES:
        return _MODEL_ALIASES[clean]
    models = list_available_models()
    for m in models:
        if m["id"].lower() == clean:
            return m["id"]
    for m in models:
        if clean and clean in m["id"].lower():
            return m["id"]
    return (raw or "").strip()


def resolve_model_ctx_limit(model_id: str) -> int:
    for m in list_available_models():
        if m["id"] == model_id:
            return int(m["ctx_limit"])
    return 32768 if "qwen3" in (model_id or "").lower() else 16384


def _extract_port(base_url: str) -> str:
    m_port = re.search(r":(\d+)", base_url or "")
    return m_port.group(1) if m_port else "8080"


def _stop_server_on_port(proc: subprocess.Popen[str] | None, port: str) -> None:
    """Cleanly terminate spawned server and any llama-server bound specifically to `port`."""
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        ps = (
            f"Get-CimInstance Win32_Process | Where-Object {{ $_.Name -eq 'llama-server.exe' "
            f"-and ($_.CommandLine -match '--port\\s+{port}(\\s|$)') }} | "
            f"ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}"
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=8, creationflags=flags)
        except Exception:
            pass
    time.sleep(0.5)


def _auto_start_server(ui: TerminalUI, base_url: str, model: str) -> subprocess.Popen[str] | None:
    port = _extract_port(base_url)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPTS_DIR / "serve.py"), "--model", model, "--port", port],
        cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        text=True, creationflags=flags,
    )
    t0 = time.perf_counter()
    live = ui.make_live_stream()
    if live is not None:
        live.__enter__()
    try:
        for _ in range(180):
            if proc.poll() is not None:
                return None
            if server_is_healthy(base_url, timeout=1.0)[0]:
                if live is not None:
                    live.update("")
                return proc
            if live is not None:
                live.update(ui.render_spinner(f"{model} yükleniyor... {time.perf_counter() - t0:.1f}s"))
            time.sleep(0.5)
    finally:
        if live is not None:
            live.__exit__(None, None, None)
    return None


def main(argv: list[str] | None = None) -> int:
    init_windows_console()
    args = build_parser().parse_args(argv)
    ws = AgentWorkspace(args.dir)
    ui = TerminalUI(
        get_files_fn=lambda: [rel for rel, _ in ws.walk_files(max_depth=4, max_items=300)],
        get_models_fn=list_available_models,
    )
    spawned_proc: subprocess.Popen[str] | None = None
    active_model = resolve_model_alias(args.model)
    ctx_limit = resolve_model_ctx_limit(active_model)
    port = _extract_port(args.url)

    ok, _ = server_is_healthy(args.url)
    if not ok and not args.no_auto_start:
        t0 = time.perf_counter()
        spawned_proc = _auto_start_server(ui, args.url, active_model)
        if spawned_proc is not None:
            ok = True
            ui.print_ok(f"{active_model} hazır ({time.perf_counter() - t0:.1f}s · {args.url})")
        else:
            ui.print_error("Yerel PicoCode sunucusu otomatik başlatılamadı.")
            return 1
    if not ok:
        ui.print_error(f"PicoCode sunucusuna bağlanılamadı ({args.url}). Önce `picocode.cmd --model {active_model}` ile başlatın.")
        return 1

    plan_only = bool(args.plan)
    auto_approve = bool(args.yes)
    conv: list[dict[str, Any]] = [{"role": "system", "content": _build_system_prompt(ws, plan_only)}]

    try:
        if args.prompt:
            at_ctx = extract_at_files(ws, args.prompt)
            conv.append({"role": "user", "content": args.prompt + at_ctx})
            run_agent_turn(
                args.url, active_model, ws, conv,
                plan_only=plan_only, auto_approve=auto_approve, max_steps=args.max_steps, ui=ui,
            )
            return 0

        git_branch, git_changes = get_git_summary(ws)
        ui.print_welcome(
            version=VERSION,
            model=active_model,
            workspace_name=ws.root.name,
            workspace_path=str(ws.root),
            git_branch=git_branch,
            git_changes=git_changes,
            plan_only=plan_only,
            auto_approve=auto_approve,
        )

        while True:
            git_branch, git_changes = get_git_summary(ws)
            try:
                user_in = ui.read_prompt(
                    model=active_model,
                    git_branch=git_branch,
                    ctx_tokens=estimate_tokens(conv),
                    ctx_limit=ctx_limit,
                    plan_only=plan_only,
                    auto_approve=auto_approve,
                )
            except (EOFError, KeyboardInterrupt):
                ui.print_info("Görüşmek üzere 👋")
                break
            if not user_in:
                continue
            if user_in in ("/exit", "/quit", "exit", "quit"):
                ui.print_info("Görüşmek üzere 👋")
                break
            if user_in == "/help":
                ui.print_help()
                continue
            if user_in == "/model" or user_in.startswith("/model "):
                models = list_available_models()
                arg_m = user_in[6:].strip()
                if not arg_m:
                    ui.print_models_table(models, active_model)
                    continue
                target_id = resolve_model_alias(arg_m)
                match = next((m for m in models if m["id"] == target_id), None)
                if not match:
                    ui.print_error(f"Bilinmeyen model: '{arg_m}'. Kurulu modelleri görmek için `/model` yazın.")
                    continue
                if not match["installed"]:
                    ui.print_error(f"'{target_id}' diske henüz indirilmemiş. Önce `setup.cmd` ile indirin.")
                    continue
                if target_id == active_model and server_is_healthy(args.url, timeout=1.5)[0]:
                    ui.print_ok(f"Zaten aktif model: {active_model}")
                    continue
                t0 = time.perf_counter()
                _stop_server_on_port(spawned_proc, port)
                spawned_proc = _auto_start_server(ui, args.url, target_id)
                if spawned_proc is None:
                    ui.print_error(f"'{target_id}' başlatılamadı! Eski modele ({active_model}) dönülüyor...")
                    spawned_proc = _auto_start_server(ui, args.url, active_model)
                    continue
                active_model = target_id
                ctx_limit = resolve_model_ctx_limit(active_model)
                ui.print_ok(f"Model değiştirildi → {active_model} ({time.perf_counter() - t0:.1f}s · sohbet geçmişi korundu)")
                continue
            if user_in == "/clear":
                conv = [{"role": "system", "content": _build_system_prompt(ws, plan_only)}]
                ui.clear()
                ui.print_welcome(
                    version=VERSION,
                    model=active_model,
                    workspace_name=ws.root.name,
                    workspace_path=str(ws.root),
                    git_branch=git_branch,
                    git_changes=git_changes,
                    plan_only=plan_only,
                    auto_approve=auto_approve,
                )
                continue
            if user_in == "/plan":
                plan_only = not plan_only
                conv[0] = {"role": "system", "content": _build_system_prompt(ws, plan_only)}
                ui.print_ok(f"Plan Modu: {'AÇIK (salt-okunur)' if plan_only else 'KAPALI (tam ajan)'}")
                continue
            if user_in == "/yes":
                auto_approve = not auto_approve
                ui.print_ok(f"Kabuk komutu otomatik onay: {'AÇIK' if auto_approve else 'KAPALI'}")
                continue
            if user_in == "/diff":
                _, out = _run_git(ws, ["diff", "--stat"])
                ui.print_info(out or "(unstaged git değişikliği yok)")
                continue
            if user_in == "/status":
                toks = estimate_tokens(conv)
                pct = min(100, int(round((toks / max(1, ctx_limit)) * 100)))
                ui.print_ok(
                    f"Model: {active_model} · Proje: {ws.root.name} · Git: {git_branch or 'yok'} ({git_changes} değişiklik) · "
                    f"Bağlam: ~{toks}/{ctx_limit} tok ({pct}%) · Mod: {'PLAN' if plan_only else 'AGENT'}"
                )
                continue
            if user_in == "/undo":
                n = ws.undo_turn()
                if n:
                    ui.print_ok(f"Son turdaki {n} dosya değişikliği geri alındı.")
                else:
                    ui.print_warn("Geri alınacak değişiklik yok.")
                continue

            at_ctx = extract_at_files(ws, user_in)
            conv.append({"role": "user", "content": user_in + at_ctx})
            res = run_agent_turn(
                args.url, active_model, ws, conv,
                plan_only=plan_only, auto_approve=auto_approve, max_steps=args.max_steps, ctx_limit=ctx_limit, ui=ui,
            )
            auto_approve = bool(res.get("auto_approve", auto_approve))
    finally:
        if spawned_proc is not None and spawned_proc.poll() is None:
            _stop_server_on_port(spawned_proc, port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
