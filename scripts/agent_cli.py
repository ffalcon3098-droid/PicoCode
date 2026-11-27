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
import urllib.parse
import urllib.request
import uuid
from typing import Any

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tui import TerminalUI, init_windows_console  # noqa: E402

VERSION = "0.6.0"

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "models", "engine", "build", "dist", "target", ".next", ".nuxt",
    "coverage", "out", ".turbo", ".pytest_cache", ".mypy_cache",
}
TEXT_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".html", ".css", ".json", ".md",
    ".txt", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".cmd", ".bat", ".ps1",
    ".sh", ".sql", ".csv", ".xml", ".c", ".cpp", ".h", ".hpp", ".rs", ".go",
}
CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".rs", ".go"}

_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+[^\n]*(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r|-r\b[^\n]*-f\b|-f\b[^\n]*-r\b)", re.I | re.S),
    re.compile(r"\b(remove-item|ri|del|erase|rmdir|rd)\b[^\n]*(-rec\w*[^\n]*-fo\w*|-fo\w*[^\n]*-rec\w*|-r\b[^\n]*-fo\w*|/[sqf]\s*[^\n]*/[sqf])", re.I | re.S),
    re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f|push\s+[^\n]*(--force|-f\b)|checkout\s+--\s+\.|restore\s+\.)", re.I | re.S),
    re.compile(r"\b(invoke-expression|iex)\b|\|\s*(sh|bash|powershell|pwsh)\b|-(enc|encodedcommand)\b", re.I | re.S),
    re.compile(r"\[(?:system\.)?io\.(?:directory|file)\]::delete\b|\bshutil\.rmtree\b|\bfs\.rmSync\b", re.I | re.S),
    re.compile(r"\b(format-volume|format\s+[a-z]:|diskpart|shutdown|stop-computer|reg\s+delete)\b", re.I | re.S),
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
    if ext == ".rs":
        prev_tauri = False
        for idx, ln in enumerate(content.splitlines(), 1):
            st = ln.strip()
            if "#[tauri::command" in st:
                prev_tauri = True
                continue
            m = re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(fn|struct|enum|trait|mod)\s+([A-Za-z_]\w*)", ln)
            if m:
                tag = " [tauri::cmd]" if (prev_tauri and m.group(1) == "fn") else ""
                out.append(f"  L{idx} {m.group(1)} {m.group(2)}{tag}")
                prev_tauri = False
                continue
            if st and not st.startswith("#[") and not st.startswith("//"):
                prev_tauri = False
        return out
    if ext == ".go":
        for idx, ln in enumerate(content.splitlines(), 1):
            m = re.match(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", ln)
            if m:
                out.append(f"  L{idx} func {m.group(1)}")
                continue
            m2 = re.match(r"^\s*type\s+([A-Za-z_]\w*)\s+(struct|interface)\b", ln)
            if m2:
                out.append(f"  L{idx} {m2.group(2)} {m2.group(1)}")
        return out
    for idx, ln in enumerate(content.splitlines(), 1):
        m = re.match(r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?(function|class|interface|type)\s+([A-Za-z_$][\w$]*)", ln)
        if m:
            out.append(f"  L{idx} {m.group(1)} {m.group(2)}")
            continue
        m2 = re.match(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", ln)
        if m2:
            out.append(f"  L{idx} fn {m2.group(1)}")
    return out


def build_project_overview(ws: AgentWorkspace, max_chars: int = 3800) -> str:
    """Build a compact 1-shot architecture & hotspot summary (manifests, dir counts, largest source files)."""
    files = ws.walk_files(max_depth=5, max_items=600)
    dir_counts: dict[str, int] = {}
    ext_counts: dict[str, int] = {}
    code_files: list[tuple[int, str, int]] = []
    for rel, sz in files:
        parts = rel.split("/")
        top = parts[0] + "/" if len(parts) > 1 else "(root)"
        dir_counts[top] = dir_counts.get(top, 0) + 1
        ext = pathlib.Path(rel).suffix.lower() or "(no-ext)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        if ext in CODE_EXTS and sz <= 300_000 and ".test." not in rel.lower() and "__tests__" not in rel.lower():
            try:
                n_lines = (ws.root / rel).read_text(encoding="utf-8", errors="replace").count("\n") + 1
            except OSError:
                n_lines = max(1, sz // 40)
            code_files.append((n_lines, rel, sz))

    lines: list[str] = [f"Project Overview: `{ws.root.name}` ({len(files)} files indexed)"]
    top_dirs = ", ".join(f"{k}({v})" for k, v in sorted(dir_counts.items(), key=lambda x: -x[1])[:10])
    top_exts = ", ".join(f"{k}:{v}" for k, v in sorted(ext_counts.items(), key=lambda x: -x[1])[:10])
    lines.append(f"Directories: {top_dirs}")
    lines.append(f"Languages/Exts: {top_exts}")

    # Compact manifests (package.json, Cargo.toml, pyproject.toml, README)
    pkg_p = ws.root / "package.json"
    if pkg_p.is_file():
        try:
            pkg = json.loads(pkg_p.read_text(encoding="utf-8", errors="replace"))
            deps = list((pkg.get("dependencies") or {}).keys())[:15]
            scripts = list((pkg.get("scripts") or {}).keys())[:10]
            lines.append(f"package.json: name={pkg.get('name')} v{pkg.get('version')} | scripts={scripts} | deps={deps}")
        except Exception:
            pass
    for cargo_rel in ("Cargo.toml", "src-tauri/Cargo.toml"):
        cp = ws.root / cargo_rel
        if cp.is_file():
            try:
                ctxt = cp.read_text(encoding="utf-8", errors="replace")
                deps = re.findall(r"^([A-Za-z0-9_-]+)\s*=", ctxt[ctxt.find("[dependencies]"):] if "[dependencies]" in ctxt else "", re.M)[:20]
                lines.append(f"{cargo_rel}: deps={deps}")
            except Exception:
                pass
    pyp = ws.root / "pyproject.toml"
    if pyp.is_file():
        lines.append("pyproject.toml: present")
    reqp = ws.root / "requirements.txt"
    if reqp.is_file():
        rdeps = [ln.strip() for ln in reqp.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip() and not ln.strip().startswith("#")][:12]
        lines.append(f"requirements.txt: {rdeps}")
    rdm = ws.root / "README.md"
    if rdm.is_file():
        rhead = "\n".join(rdm.read_text(encoding="utf-8", errors="replace").splitlines()[:12])[:550]
        lines.append(f"README excerpt:\n{rhead}")

    # Largest / core code files (best review hotspots!)
    code_files.sort(key=lambda x: -x[0])
    if code_files:
        lines.append("Key/largest code files (lines):")
        for nl, rel, _ in code_files[:28]:
            lines.append(f"  - {rel} ({nl}L)")

    return "\n".join(lines)[:max_chars]


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
                if e.is_symlink():
                    continue
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
        orig_bytes = p.read_bytes() if existed else b""
        if existed and b"\x00" in orig_bytes[:8192]:
            raise ValueError(f"Binary file cannot be edited as text: {rel_path}")
        orig_eol = "\r\n" if (existed and b"\r\n" in orig_bytes) else "\n"
        old_text = orig_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n") if existed else ""
        norm_new = content.replace("\r\n", "\n")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline=orig_eol) as fh:
            fh.write(norm_new)
        rel = p.relative_to(self.root).as_posix()
        uid = uuid.uuid4().hex[:10]
        self._undo_store[uid] = {
            "abs_path": str(p),
            "rel_path": rel,
            "existed": existed,
            "old_bytes": orig_bytes,
        }
        if len(self._undo_store) > 120:
            del self._undo_store[next(iter(self._undo_store))]
        diff_lines = list(difflib.unified_diff(
            old_text.splitlines(), norm_new.splitlines(),
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
                p.write_bytes(rec["old_bytes"])
            elif p.exists():
                p.unlink()
            count += 1
        return count


AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "project_overview",
            "description": "Get a compact 1-shot project summary: tech stack manifests (package.json/Cargo.toml/README), directory/language breakdown, and largest core source files. Call this FIRST for project reviews.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
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

READONLY_TOOLS = {"project_overview", "list_files", "read_file", "search_code", "repo_map", "git_status", "git_diff"}
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
    compact_budget: bool = False,
) -> tuple[str, dict[str, Any] | None]:
    try:
        if name == "project_overview":
            return build_project_overview(ws, max_chars=3600 if compact_budget else 5200), None

        if name == "list_files":
            sub = str(args.get("subdir") or "").strip()
            base = ws.safe_path(sub) if sub else ws.root
            if not base.is_dir():
                return f"ERROR: Directory not found: {sub}", None
            lines = []
            for e in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:120]:
                if e.name in IGNORE_DIRS or e.is_symlink():
                    continue
                rel = e.relative_to(ws.root).as_posix()
                try:
                    lines.append(f"{rel}/" if e.is_dir() else f"{rel} ({e.stat().st_size} B)")
                except OSError:
                    continue
            return "\n".join(lines) or "(empty directory)", None

        if name == "read_file":
            rel = str(args.get("path") or "")
            p = ws.safe_path(rel)
            if not p.is_file():
                return f"ERROR: File not found: {rel}", None
            all_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            default_win = 119 if compact_budget else 199
            max_chars = 5200 if compact_budget else 9000
            s = max(1, int(args.get("start_line") or 1))
            e = min(len(all_lines), int(args.get("end_line") or (s + default_win)))
            numbered = [f"{i}: {all_lines[i - 1]}" for i in range(s, e + 1)]
            return f"File: {p.relative_to(ws.root).as_posix()} (lines {s}-{e} of {len(all_lines)})\n" + "\n".join(numbered)[:max_chars], None

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
                blk = f"{rel} ({txt.count(chr(10)) + 1}L):\n" + ("\n".join(syms[:25 if compact_budget else 30]) if syms else "  (no top-level symbols)")
                if total + len(blk) > (4600 if compact_budget else 7500):
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
            target = str(args.get("target") or "").replace("\r\n", "\n")
            replacement = str(args.get("replacement") or "").replace("\r\n", "\n")
            p = ws.safe_path(rel)
            if not p.is_file():
                return f"ERROR: File not found: {rel}", None
            old = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
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
            env["PYTHONIOENCODING"] = "utf-8"
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt":
                ps_cmd = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + cmd
                shell_cmd = ["powershell", "-NoProfile", "-Command", ps_cmd]
            else:
                shell_cmd = ["sh", "-lc", cmd]
            proc = subprocess.Popen(
                shell_cmd, cwd=str(ws.root), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", creationflags=flags,
            )
            try:
                out, _ = proc.communicate(timeout=45)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, creationflags=flags)
                else:
                    proc.kill()
                return "ERROR: Command timed out after 45s (process tree terminated).", None
            return f"exit_code={proc.returncode}\n{(out or '').strip()[:5000]}", None

        return f"ERROR: Unknown tool {name}", None
    except Exception as exc:
        return f"ERROR: {exc}", None


def compact_messages(
    conv: list[dict[str, Any]],
    *,
    ctx_limit: int = 32768,
    keep_recent_tools: int = 4,
    high_water_ratio: float = 0.72,
) -> list[dict[str, Any]]:
    """
    Hysteresis/epoch KV-cache-friendly compaction:
    NEVER mutates older messages while below `high_water_ratio * ctx_limit` (preserving 100% KV prefix hits!).
    Only when crossing the high-water mark, batch-compacts older tool outputs once.
    """
    if estimate_tokens(conv) < int(ctx_limit * high_water_ratio):
        return conv
    tool_idxs = [i for i, m in enumerate(conv) if m.get("role") == "tool"]
    if len(tool_idxs) <= keep_recent_tools:
        return conv
    cutoff = tool_idxs[-keep_recent_tools]
    out: list[dict[str, Any]] = []
    for i, m in enumerate(conv):
        if i < cutoff and m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 480 and not m["content"].startswith("[Compacted"):
            lines = m["content"].splitlines()
            head = "\n".join(lines[:3])[:200]
            tail = "\n".join(lines[-2:])[:120] if len(lines) > 5 else ""
            out.append({**m, "content": f"[Compacted: {len(m['content'])} chars]\n{head}" + (f"\n...\n{tail}" if tail else "")})
        else:
            out.append(m)
    return out


def _extract_fallback_tool_calls(text: str, step: int, allowed: set[str]) -> list[dict[str, Any]]:
    """Parse `<tool_call>{...}</tool_call>` or fenced JSON tool calls when a model omits structured tool_calls."""
    found: list[dict[str, Any]] = []
    candidates = re.findall(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", text)
    if not candidates:
        candidates = re.findall(r"```(?:json)?\s*\n(\{[\s\S]*?\})\s*```", text)
    for idx, raw in enumerate(candidates):
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("name") in allowed and isinstance(obj.get("arguments"), dict):
            found.append({
                "id": f"fb_{step}_{idx}",
                "type": "function",
                "function": {"name": obj["name"], "arguments": json.dumps(obj["arguments"], ensure_ascii=False)},
            })
    return found


def get_model_sampling_params(model_id: str) -> dict[str, Any]:
    """Load per-model recommended sampling defaults from models_registry.json."""
    reg_path = REPO_ROOT / "configs" / "models_registry.json"
    defaults: dict[str, Any] = {}
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        defaults = dict((reg.get("models") or {}).get(model_id, {}).get("defaults") or {})
    except Exception:
        pass
    params: dict[str, Any] = {
        "temperature": float(defaults.get("agent_temp", defaults.get("temp", 0.3))),
        "top_p": float(defaults.get("top_p", 0.9)),
        "min_p": float(defaults.get("min_p", 0.05)),
        "cache_prompt": True,
    }
    if "inkling" in (model_id or "").lower():
        params["reasoning_budget"] = int(defaults.get("reasoning_budget", 1536))
    return params


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
    has_final_answer = False
    compact_budget = (ctx_limit <= 16384) or ("inkling" in (model or "").lower())

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
        for step in range(1, max_steps + 2):
            conv[:] = compact_messages(conv, ctx_limit=ctx_limit)
            force_final = (step > max_steps) or (step >= 3 and estimate_tokens(conv) >= int(ctx_limit * 0.78))
            if force_final:
                if has_final_answer:
                    break
                conv.append({
                    "role": "user",
                    "content": (
                        "İnceleme adımları tamamlandı. Başka araç çağırma; şimdiye kadar okuduğun gerçek dosyalara "
                        "dayanarak net, somut, dosya adlarıyla desteklenmiş kapsamlı final yanıtını yaz."
                    ),
                })
            sampling = get_model_sampling_params(model)
            # Keep intermediate tool-selection thinking crisp on Inkling so reviews stay fast
            if "reasoning_budget" in sampling:
                if not force_final and step < max_steps:
                    sampling["reasoning_budget"] = min(int(sampling["reasoning_budget"]), 512)
                elif force_final:
                    sampling["reasoning_budget"] = min(int(sampling["reasoning_budget"]), 768)
            payload: dict[str, Any] = {
                "model": model,
                "messages": conv,
                # Always keep `tools` identical so llama-server's Jinja prefix stays 100% KV-cached (never wipe 10K ctx!)
                "tools": tools_spec,
                "tool_choice": "none" if force_final else "auto",
                "max_tokens": 2048 if compact_budget else 4096,
                "stream": True,
                **sampling,
            }
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer picocode"},
            )
            step_text: list[str] = []
            tc_map: dict[int, dict[str, Any]] = {}
            step_t0 = time.perf_counter()
            step_err: str | None = None

            live = (ui.make_live_stream() if (ui is not None and not quiet) else None)
            if live is not None:
                live.__enter__()
                lbl = "Final yanıt yazılıyor..." if force_final else f"Düşünüyor... (adım {step}/{max_steps})"
                live.update(ui.render_spinner(lbl))
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
                        try:
                            ev = json.loads(ds)
                        except json.JSONDecodeError:
                            continue
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
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                step_err = f"Sunucu bağlantı/HTTP hatası: {exc}"
            finally:
                if live is not None:
                    if step_text:
                        live.update(ui.render_markdown("".join(step_text)))
                    else:
                        live.update("")
                    live.__exit__(None, None, None)
                elif step_text and not quiet:
                    print()

            if step_err:
                if ui is not None and not quiet:
                    ui.print_error(step_err)
                break

            if force_final:
                conv.append({"role": "assistant", "content": "".join(step_text)})
                has_final_answer = True
                break

            if not tc_map:
                allowed = READONLY_TOOLS if plan_only else {t["function"]["name"] for t in AGENT_TOOLS}
                fb_calls = _extract_fallback_tool_calls("".join(step_text), step, allowed)
                if not fb_calls:
                    conv.append({"role": "assistant", "content": "".join(step_text)})
                    has_final_answer = True
                    break
                tool_calls = fb_calls
            else:
                tool_calls = [tc_map[k] for k in sorted(tc_map)]

            conv.append({"role": "assistant", "content": "".join(step_text) or None, "tool_calls": tool_calls})
            answered_ids: set[str] = set()
            n_calls = max(1, len(tool_calls))
            step_char_budget = 6400 if compact_budget else 11000
            per_call_cap = max(2200, step_char_budget // n_calls) if n_calls > 1 else 0
            try:
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    try:
                        parsed = json.loads(raw_args) if raw_args else {}
                        if not isinstance(parsed, dict):
                            raise ValueError("Tool arguments must be a JSON object")
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
                            compact_budget=compact_budget,
                        )
                        if per_call_cap and fn_name in READONLY_TOOLS and len(res_txt) > per_call_cap:
                            res_txt = (
                                res_txt[:per_call_cap].rstrip()
                                + "\n... [paralel adım bütçesi nedeniyle kırpıldı; devamı için read_file(start_line, end_line) çağırın]"
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
                    answered_ids.add(tc["id"])
            except KeyboardInterrupt:
                # Keep conv role/tool_call_id pairing valid for future turns
                for tc in tool_calls:
                    if tc["id"] not in answered_ids:
                        conv.append({"role": "tool", "tool_call_id": tc["id"], "content": "ERROR: Cancelled by user."})
                raise
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


def _build_system_prompt(
    ws: AgentWorkspace,
    plan_only: bool,
    pinned_files: list[str] | None = None,
    *,
    trusted: bool = True,
) -> str:
    root_entries = []
    try:
        for e in sorted(ws.root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:25]:
            if e.name in IGNORE_DIRS or e.is_symlink():
                continue
            root_entries.append(e.name + ("/" if e.is_dir() else ""))
    except OSError:
        pass
    rules = ""
    if trusted:
        for rf in (".picostudio.md", "AGENTS.md"):
            p = ws.root / rf
            if p.is_file() and not p.is_symlink():
                rules = f"\nProject rules ({rf}):\n{p.read_text(encoding='utf-8', errors='replace')[:1000]}\n"
                break
    pinned_block = ""
    if pinned_files:
        chunks = []
        for rel in pinned_files[:4]:
            try:
                fp = ws.safe_path(rel)
                if fp.is_file():
                    txt = fp.read_text(encoding="utf-8", errors="replace")[:2600]
                    chunks.append(f"\nPinned file `{rel}`:\n```\n{txt}\n```")
            except Exception:
                continue
        if chunks:
            pinned_block = "\n" + "\n".join(chunks) + "\n"
    greeting_rule = (
        " For simple greetings (e.g. 'hi', 'hello', 'merhaba', 'selam') or general chat that does not ask to inspect/edit code, "
        "reply briefly in 1-2 sentences WITHOUT calling any tools."
    )
    if plan_only:
        return (
            f"You are PicoCode Terminal Planning & Review Agent. Workspace: `{ws.root.name}` (root: {', '.join(root_entries)}). "
            "READ-ONLY MODE: do not modify files. For project reviews or architecture questions, call `project_overview` FIRST "
            "(1 compact call), then read 2-5 highest-risk source files with `read_file` or `search_code` (and `repo_map` for a subdir), "
            "then stop calling tools and write a concrete, file-and-line-cited review or numbered plan in the user's language."
            + greeting_rule + rules + pinned_block
        )
    return (
        f"You are PicoCode Terminal Coding Agent. Workspace: `{ws.root.name}` (root: {', '.join(root_entries)}). "
        "When reviewing or exploring a project, call `project_overview` FIRST, then inspect specific files with `read_file`, `repo_map`, or `search_code`. "
        "Use `edit_file` for targeted changes, `write_file` for new files, and `run_command` to run tests. "
        "Fix any `[SYNTAX ERROR]` immediately. Be concise and concrete; reply in the user's language." + greeting_rule + rules + pinned_block
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"PicoCode v{VERSION} — Claude-Code-Style Local Terminal Coding Agent CLI")
    p.add_argument("prompt", nargs="?", default=None, help="Optional one-shot task prompt (omit for interactive REPL).")
    p.add_argument("--dir", "-d", default=None, help="Target workspace directory (omit to pick interactively).")
    p.add_argument("--model", "-m", default="qwen3-coder-30b", help="Model ID (default: qwen3-coder-30b).")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible server base URL.")
    p.add_argument("--plan", action="store_true", help="Start in read-only planning mode.")
    p.add_argument("--yes", "-y", action="store_true", help="Auto-approve safe shell commands.")
    p.add_argument("--trust", action="store_true", help="Trust target workspace folder without interactive prompt.")
    p.add_argument("--max-steps", type=int, default=16, help="Maximum tool-calling steps per turn (default: 16).")
    p.add_argument("--no-auto-start", action="store_true", help="Do not auto-start local llama-server if unreachable.")
    return p


class TrustStore:
    """Persists trusted workspace folders in `~/.picocode_trusted_folders.json`."""

    def __init__(self, store_path: pathlib.Path | None = None) -> None:
        self.path = store_path or (pathlib.Path.home() / ".picocode_trusted_folders.json")

    @staticmethod
    def _norm(folder: str | pathlib.Path) -> str:
        return os.path.normcase(str(pathlib.Path(folder).resolve()))

    def _load(self) -> list[str]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data if x]
        except Exception:
            pass
        return []

    def _save(self, items: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_trusted(self) -> list[str]:
        return sorted(self._load(), key=lambda s: s.lower())

    def is_trusted(self, folder: str | pathlib.Path) -> bool:
        target = self._norm(folder)
        return any(self._norm(x) == target for x in self._load())

    def trust(self, folder: str | pathlib.Path) -> str:
        resolved = str(pathlib.Path(folder).resolve())
        items = self._load()
        if not self.is_trusted(resolved):
            items.append(resolved)
            self._save(items)
        return resolved

    def untrust(self, folder: str | pathlib.Path) -> bool:
        target = self._norm(folder)
        items = self._load()
        kept = [x for x in items if self._norm(x) != target]
        if len(kept) == len(items):
            return False
        self._save(kept)
        return True


def discover_projects(base_dir: pathlib.Path, trust_store: TrustStore) -> list[dict[str, Any]]:
    """List sibling project folders alongside current/PicoCode projects root plus trusted folders."""
    candidates: list[pathlib.Path] = []
    seen: set[str] = set()

    def _add(p: pathlib.Path) -> None:
        try:
            rp = p.resolve()
            if not rp.is_dir():
                return
            key = os.path.normcase(str(rp))
            if key in seen:
                return
            seen.add(key)
            candidates.append(rp)
        except OSError:
            pass

    for parent in (base_dir.parent, REPO_ROOT.parent):
        if parent.is_dir():
            try:
                for d in sorted(parent.iterdir(), key=lambda x: x.name.lower()):
                    if d.is_dir() and not d.name.startswith("."):
                        _add(d)
            except OSError:
                pass
    for tf in trust_store.list_trusted():
        _add(pathlib.Path(tf))
    _add(base_dir)

    out: list[dict[str, Any]] = []
    for p in candidates[:30]:
        out.append({
            "name": p.name,
            "path": str(p),
            "trusted": trust_store.is_trusted(p),
            "is_git": (p / ".git").exists(),
        })
    return out


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
            if k and k in files:
                fp = REPO_ROOT / files[k]["path"]
                exp = int(files[k].get("expected_bytes", 0))
                if fp.is_file() and (exp <= 0 or fp.stat().st_size == exp):
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
    p = urllib.parse.urlparse(base_url or "")
    return str(p.port) if p.port else "8080"


def _is_local_url(base_url: str) -> bool:
    h = (urllib.parse.urlparse(base_url or "").hostname or "").lower()
    return h in ("127.0.0.1", "localhost", "::1")


def _stop_server_on_port(proc: subprocess.Popen[str] | None, port: str) -> None:
    """Cleanly terminate spawned server tree and any llama-server bound specifically to `port`."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if proc is not None and proc.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, creationflags=flags)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    if os.name == "nt":
        ps = (
            f"Get-CimInstance Win32_Process | Where-Object {{ $_.Name -eq 'llama-server.exe' "
            f"-and ($_.CommandLine -match '--port\\s+{port}(\\s|$)') }} | "
            f"ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}"
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=8, creationflags=flags)
        except Exception:
            pass
    time.sleep(0.4)


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
        _stop_server_on_port(proc, port)
        return None
    except BaseException:
        _stop_server_on_port(proc, port)
        raise
    finally:
        if live is not None:
            live.__exit__(None, None, None)


def _resolve_project_target(raw: str, current_root: pathlib.Path, projects: list[dict[str, Any]]) -> pathlib.Path:
    """Resolve a user project choice ('new:foo', project name, relative or absolute path)."""
    clean = (raw or "").strip().strip('"').strip("'")
    if clean.startswith("new:"):
        name = clean[4:].strip()
        cand_name = pathlib.Path(name)
        if not name or cand_name.is_absolute() or len(cand_name.parts) != 1 or name in (".", "..") or any(c in name for c in ("/", "\\", ":")):
            raise ValueError("Gecersiz yeni proje adi (sadece tek klasor adi girin).")
        base = (REPO_ROOT.parent if REPO_ROOT.parent.is_dir() else current_root.parent).resolve()
        target = (base / name).resolve()
        if not target.is_relative_to(base):
            raise PermissionError("Proje kok dizini disinda klasor olusturulamaz.")
        target.mkdir(parents=True, exist_ok=True)
        return target
    for p in projects:
        if p["name"].lower() == clean.lower():
            return pathlib.Path(p["path"]).resolve()
    cand = pathlib.Path(clean)
    if not cand.is_absolute():
        sib = (current_root.parent / cand).resolve()
        if sib.is_dir():
            return sib
        cand = (current_root / cand).resolve()
    if not cand.is_dir():
        raise FileNotFoundError(f"Klasor bulunamadi: {clean}")
    return cand


def main(argv: list[str] | None = None) -> int:
    init_windows_console()
    args = build_parser().parse_args(argv)
    trust_store = TrustStore()
    is_interactive = (args.prompt is None) and sys.stdin.isatty()

    # Determine initial workspace — if launched interactively without `--dir` from PicoCode folder, show Project Picker!
    initial_dir = pathlib.Path(args.dir).resolve() if args.dir else pathlib.Path.cwd().resolve()
    pinned_files: list[str] = []
    ws_holder: dict[str, AgentWorkspace] = {"ws": AgentWorkspace(initial_dir)}

    ui = TerminalUI(
        get_files_fn=lambda: [rel for rel, _ in ws_holder["ws"].walk_files(max_depth=4, max_items=300)],
        get_models_fn=list_available_models,
        get_projects_fn=lambda: discover_projects(ws_holder["ws"].root, trust_store),
        get_pinned_fn=lambda: list(pinned_files),
    )

    if is_interactive and args.dir is None:
        projs = discover_projects(initial_dir, trust_store)
        if len(projs) > 1:
            choice = ui.pick_project_interactive(projs, str(initial_dir), title="📁 Hangi Proje Klasöründe Çalışalım?")
            if choice:
                try:
                    chosen_dir = _resolve_project_target(choice, initial_dir, projs)
                    ws_holder["ws"] = AgentWorkspace(chosen_dir)
                except Exception as exc:
                    ui.print_error(str(exc))
                    return 1

    ws = ws_holder["ws"]
    plan_only = bool(args.plan)
    auto_approve = bool(args.yes)

    # Folder Trust Gate (Claude Code style — enforced in BOTH interactive and one-shot modes!)
    if args.trust:
        trust_store.trust(ws.root)
    elif not trust_store.is_trusted(ws.root):
        if is_interactive:
            decision = ui.prompt_trust_folder(str(ws.root))
            if decision == "trust":
                trust_store.trust(ws.root)
                ui.print_ok(f"Güvenilen klasör kaydedildi: {ws.root.name}")
            elif decision == "plan":
                plan_only = True
                ui.print_warn("Klasör salt-okunur Plan Modunda açıldı (dosyalar değiştirilmeyecek).")
            else:
                ui.print_info("İşlem iptal edildi.")
                return 0
        else:
            plan_only = True
            ui.print_warn(f"'{ws.root.name}' güvenilen klasör değil — tek seferlik çalıştırma salt-okunur Plan Modunda sınırlandı (yazma için `--trust` ekleyin).")

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

    conv: list[dict[str, Any]] = [{"role": "system", "content": _build_system_prompt(ws, plan_only, pinned_files, trusted=trust_store.is_trusted(ws.root))}]

    def _show_welcome() -> None:
        gb, gc = get_git_summary(ws)
        ui.print_welcome(
            version=VERSION,
            model=active_model,
            workspace_name=ws.root.name,
            workspace_path=str(ws.root),
            git_branch=gb,
            git_changes=gc,
            plan_only=plan_only,
            auto_approve=auto_approve,
            trusted=trust_store.is_trusted(ws.root),
            ctx_limit=ctx_limit,
        )

    try:
        if args.prompt:
            at_ctx = extract_at_files(ws, args.prompt)
            conv.append({"role": "user", "content": args.prompt + at_ctx})
            run_agent_turn(
                args.url, active_model, ws, conv,
                plan_only=plan_only, auto_approve=auto_approve, max_steps=args.max_steps, ctx_limit=ctx_limit, ui=ui,
            )
            return 0

        _show_welcome()

        while True:
            git_branch, git_changes = get_git_summary(ws)
            try:
                user_in = ui.read_prompt(
                    model=active_model,
                    workspace_name=ws.root.name,
                    git_branch=git_branch,
                    ctx_tokens=estimate_tokens(conv),
                    ctx_limit=ctx_limit,
                    plan_only=plan_only,
                    auto_approve=auto_approve,
                    pinned_count=len(pinned_files),
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
            if user_in in ("/project", "/cd") or user_in.startswith("/project ") or user_in.startswith("/cd "):
                projs = discover_projects(ws.root, trust_store)
                raw_arg = user_in.split(" ", 1)[1].strip() if " " in user_in else ""
                if not raw_arg:
                    choice = ui.pick_project_interactive(projs, str(ws.root))
                    if not choice:
                        continue
                    raw_arg = choice
                elif raw_arg.lower().startswith("new "):
                    raw_arg = "new:" + raw_arg[4:].strip()
                try:
                    new_dir = _resolve_project_target(raw_arg, ws.root, projs)
                except Exception as exc:
                    ui.print_error(str(exc))
                    continue
                if not trust_store.is_trusted(new_dir):
                    dec = ui.prompt_trust_folder(str(new_dir))
                    if dec == "trust":
                        trust_store.trust(new_dir)
                    elif dec == "plan":
                        plan_only = True
                    else:
                        ui.print_warn("Proje değişikliği iptal edildi.")
                        continue
                ws = AgentWorkspace(new_dir)
                ws_holder["ws"] = ws
                pinned_files.clear()
                conv = [{"role": "system", "content": _build_system_prompt(ws, plan_only, pinned_files)}]
                ui.print_ok(f"Proje değiştirildi → {ws.root.name} ({ws.root}) [model VRAM'de sıcak kaldı · 0s]")
                _show_welcome()
                continue
            if user_in == "/trust" or user_in.startswith("/trust "):
                sub = user_in[6:].strip().lower()
                if sub == "add":
                    trust_store.trust(ws.root)
                    ui.print_ok(f"Güvenilen klasörlere eklendi: {ws.root}")
                elif sub in ("remove", "rm", "revoke"):
                    if trust_store.untrust(ws.root):
                        plan_only = True
                        conv[0] = {"role": "system", "content": _build_system_prompt(ws, plan_only, pinned_files)}
                        ui.print_warn(f"Güven kaldırıldı ve Plan Moduna geçildi: {ws.root}")
                    else:
                        ui.print_info("Bu klasör zaten güvenilenler listesinde değil.")
                else:
                    ui.print_trusted_table(trust_store.list_trusted(), str(ws.root))
                continue
            if user_in == "/files" or user_in.startswith("/files "):
                sub = user_in[6:].strip().replace("\\", "/").strip("/")
                all_f = ws.walk_files(max_depth=4, max_items=300)
                if sub:
                    all_f = [(r, s) for r, s in all_f if r == sub or r.startswith(sub + "/")]
                ui.print_files_table(all_f, pinned_files, subdir=sub)
                continue
            if user_in == "/add" or user_in.startswith("/add "):
                rel = user_in[4:].strip()
                if not rel:
                    if pinned_files:
                        ui.print_ok("Bağlama sabitlenmiş dosyalar: " + ", ".join(pinned_files))
                    else:
                        ui.print_info("Sabitlenmiş dosya yok. Eklemek için `/add dosya.py` (Tab ile seçin).")
                    continue
                try:
                    p = ws.safe_path(rel)
                    if not p.is_file():
                        ui.print_error(f"Dosya bulunamadı: {rel}")
                        continue
                    rel_norm = p.relative_to(ws.root).as_posix()
                    if rel_norm not in pinned_files:
                        pinned_files.append(rel_norm)
                    conv[0] = {"role": "system", "content": _build_system_prompt(ws, plan_only, pinned_files)}
                    ui.print_ok(f"Bağlama sabitlendi: 📎 {rel_norm} (toplam {len(pinned_files)} dosya)")
                except Exception as exc:
                    ui.print_error(str(exc))
                continue
            if user_in == "/drop" or user_in.startswith("/drop "):
                rel = user_in[5:].strip().replace("\\", "/").lstrip("/")
                if not rel:
                    pinned_files.clear()
                    conv[0] = {"role": "system", "content": _build_system_prompt(ws, plan_only, pinned_files, trusted=trust_store.is_trusted(ws.root))}
                    ui.print_ok("Tüm sabitlenmiş dosyalar bağlamdan çıkarıldı.")
                elif rel in pinned_files:
                    pinned_files.remove(rel)
                    conv[0] = {"role": "system", "content": _build_system_prompt(ws, plan_only, pinned_files, trusted=trust_store.is_trusted(ws.root))}
                    ui.print_ok(f"Bağlamdan çıkarıldı: {rel}")
                else:
                    ui.print_warn(f"Sabitli dosya bulunamadı: {rel}")
                continue
            if user_in == "/model" or user_in.startswith("/model "):
                models = list_available_models()
                arg_m = user_in[6:].strip()
                if not arg_m:
                    ui.print_models_table(models, active_model)
                    continue
                if not _is_local_url(args.url) or args.no_auto_start:
                    ui.print_error("Uzak sunucuda veya --no-auto-start modunda canlı /model değiştirme devre dışı.")
                    continue
                target_id = resolve_model_alias(arg_m)
                match = next((m for m in models if m["id"] == target_id), None)
                if not match:
                    ui.print_error(f"Bilinmeyen model: '{arg_m}'. Kurulu modelleri görmek için `/model` yazın.")
                    continue
                if not match["installed"]:
                    ui.print_error(f"'{target_id}' diske henüz indirilmemiş (veya yarım kalmış). Önce `setup.cmd` ile indirin.")
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
                    if spawned_proc is None:
                        ui.print_error("Eski model de başlatılamadı. `/exit` ile çıkıp yeniden deneyin.")
                    continue
                active_model = target_id
                ctx_limit = resolve_model_ctx_limit(active_model)
                if estimate_tokens(conv) > int(ctx_limit * 0.72):
                    conv[:] = compact_messages(conv, ctx_limit=ctx_limit, high_water_ratio=0.5)
                    ui.print_warn(f"Yeni model bağlamı ({ctx_limit // 1024}K) için eski araç çıktıları sıkıştırıldı.")
                ui.print_ok(f"Model değiştirildi → {active_model} ({time.perf_counter() - t0:.1f}s · sohbet geçmişi korundu)")
                continue
            if user_in == "/clear":
                conv = [{"role": "system", "content": _build_system_prompt(ws, plan_only, pinned_files)}]
                ui.clear()
                _show_welcome()
                continue
            if user_in == "/plan":
                if plan_only and not trust_store.is_trusted(ws.root):
                    dec = ui.prompt_trust_folder(str(ws.root))
                    if dec != "trust":
                        ui.print_warn("Güven onayı verilmedi; salt-okunur Plan Modunda kalınıyor.")
                        continue
                    trust_store.trust(ws.root)
                plan_only = not plan_only
                conv[0] = {"role": "system", "content": _build_system_prompt(ws, plan_only, pinned_files)}
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
                trust_lbl = "Güvenli" if trust_store.is_trusted(ws.root) else "Salt-Okunur"
                pin_lbl = f" · Sabit: {len(pinned_files)} dosya" if pinned_files else ""
                ui.print_ok(
                    f"Model: {active_model} · Proje: {ws.root.name} ({trust_lbl}) · Git: {git_branch or 'yok'} ({git_changes}) · "
                    f"Bağlam: ~{toks}/{ctx_limit} tok ({pct}%) · Mod: {'PLAN' if plan_only else 'AGENT'}{pin_lbl}"
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
