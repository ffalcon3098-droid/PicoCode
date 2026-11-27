#!/usr/bin/env python3
"""
PicoCode v0.3.0 — Standalone Local Terminal Coding Agent CLI.

Runs an autonomous 16-step local coding agent directly in your terminal backed by
PicoCode's OpenAI-compatible `llama-server` (Qwen3-Coder-30B-A3B up to 30 tok/s,
Inkling 2 Mini, or Qwen2.5-Coder-7B).

Features:
  - Zero third-party dependencies (pure Python stdlib + ANSI color output)
  - 9 safe workspace tools: list_files, read_file, search_code, repo_map (AST outline),
    git_status, git_diff, edit_file, write_file, run_command
  - Path-traversal guard, ambiguous-edit rejection, instant Python/JSON syntax check,
    destructive shell command blocker, and per-turn `/undo` rollback
  - Smart context compaction & consecutive duplicate tool-call loop guard
  - Interactive REPL (`/help`, `/diff`, `/undo`, `/plan`, `/yes`, `/clear`, `/exit`)
    AND one-shot headless execution (`picocode agent "fix bug" --yes`)
  - Auto-connects to an already-running local server or optionally spawns `serve.py`

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

# ANSI helpers
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if USE_COLOR else text


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
            text=True, encoding="utf-8", errors="replace", timeout=15, creationflags=flags,
        )
        return res.returncode, ((res.stdout or "") + ("\n" + res.stderr if res.stderr else "")).strip()
    except Exception as exc:
        return 1, str(exc)


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
            syn_txt = ("\n[SYNTAX ERROR] " + "; ".join(syn) + "\nFix immediately.") if syn else ""
            return f"OK: Edited {res['path']} (+{res['added']}/-{res['removed']}).{syn_txt}", res

        if name == "write_file":
            rel = str(args.get("path") or "")
            content = str(args.get("content") or "")
            res = ws.write_with_undo(rel, content)
            syn = check_syntax_quick(res["path"], content)
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
                    ans = input(_c("33", f"  Run command `{cmd}`? [y/N]: ")).strip().lower()
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


def _print_colored_diff(diff_txt: str) -> None:
    if not diff_txt:
        return
    for ln in diff_txt.splitlines()[:28]:
        if ln.startswith("+") and not ln.startswith("+++"):
            print("    " + _c("32", ln))
        elif ln.startswith("-") and not ln.startswith("---"):
            print("    " + _c("31", ln))
        elif ln.startswith("@@"):
            print("    " + _c("36", ln))
        else:
            print("    " + _c("90", ln))


def run_agent_turn(
    base_url: str,
    model: str,
    ws: AgentWorkspace,
    conv: list[dict[str, Any]],
    *,
    plan_only: bool = False,
    auto_approve: bool = False,
    max_steps: int = 16,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run one complete multi-step tool-calling turn against an OpenAI-compatible server."""
    url = base_url.rstrip("/") + ("/chat/completions" if base_url.rstrip("/").endswith("/v1") else "/v1/chat/completions")
    tools_spec = READONLY_TOOL_SPECS if plan_only else AGENT_TOOLS
    turn_undos: list[str] = []
    tools_called: list[str] = []
    recent_sigs: list[str] = []
    full_answer: list[str] = []
    t_start = time.perf_counter()
    tok_count = 0

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
        streamed_any = False

        with urllib.request.urlopen(req, timeout=360) as resp:
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
                if delta.get("content"):
                    c_txt = delta["content"]
                    step_text.append(c_txt)
                    full_answer.append(c_txt)
                    tok_count += 1
                    streamed_any = True
                    if not quiet:
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

        if streamed_any and not quiet:
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
                if not quiet:
                    print(_c("36", f"  [step {step}/{max_steps}] ⚙ {fn_name}({json.dumps(parsed, ensure_ascii=False)[:100]})"))
                res_txt, chg = execute_cli_tool(ws, fn_name, parsed, auto_approve=auto_approve)

            if chg:
                turn_undos.append(chg["undo_id"])
                if not quiet:
                    print(_c("32", f"  ✏ {chg['path']} (+{chg['added']}/-{chg['removed']})"))
                    _print_colored_diff(chg.get("diff", ""))
            elif not quiet:
                summary = res_txt.splitlines()[0][:110] if res_txt else "OK"
                print(_c("90", f"    ↳ {summary}"))

            conv.append({"role": "tool", "tool_call_id": tc["id"], "content": res_txt})

    if turn_undos:
        ws.turn_history.append(turn_undos)

    elapsed = max(0.05, time.perf_counter() - t_start)
    if not quiet:
        tps = tok_count / elapsed if tok_count else 0.0
        undo_hint = f" · {len(turn_undos)} edits (/undo)" if turn_undos else ""
        print(_c("90", f"  [{elapsed:.1f}s · {tok_count} chunks · ~{tps:.1f} t/s · {len(tools_called)} tools{undo_hint}]"))

    return {
        "answer": "".join(full_answer).strip(),
        "tools_called": tools_called,
        "undo_ids": turn_undos,
        "elapsed": elapsed,
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
    if plan_only:
        return (
            f"You are PicoCode Terminal Planning Agent. Workspace: `{ws.root.name}` (root: {', '.join(root_entries)}). "
            "READ-ONLY PLAN MODE: do not modify files. Use `repo_map`, `read_file`, `search_code`, `git_status`, `git_diff` "
            "and produce a concise numbered plan in the user's language." + rules
        )
    return (
        f"You are PicoCode Terminal Coding Agent. Workspace: `{ws.root.name}` (root: {', '.join(root_entries)}). "
        "Use `repo_map` or `search_code` to inspect quickly, `edit_file` for targeted changes, `write_file` for new files, "
        "and `run_command` to run tests. Fix any `[SYNTAX ERROR]` immediately. Be concise; reply in the user's language." + rules
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PicoCode v0.3.0 — Local Terminal Coding Agent CLI")
    p.add_argument("prompt", nargs="?", default=None, help="Optional one-shot task prompt (omit for interactive REPL).")
    p.add_argument("--dir", "-d", default=".", help="Target workspace directory (default: current directory).")
    p.add_argument("--model", "-m", default="qwen3-coder-30b", help="Model ID (default: qwen3-coder-30b).")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible server base URL.")
    p.add_argument("--plan", action="store_true", help="Start in read-only planning mode.")
    p.add_argument("--yes", "-y", action="store_true", help="Auto-approve safe shell commands.")
    p.add_argument("--max-steps", type=int, default=16, help="Maximum tool-calling steps per turn (default: 16).")
    p.add_argument("--no-auto-start", action="store_true", help="Do not auto-start local llama-server if unreachable.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ws = AgentWorkspace(args.dir)
    spawned_proc: subprocess.Popen[str] | None = None

    ok, _ = server_is_healthy(args.url)
    if not ok and not args.no_auto_start:
        print(_c("33", f"PicoCode server not reachable at {args.url} — starting `{args.model}` via serve.py..."))
        m_port = re.search(r":(\d+)", args.url)
        port = m_port.group(1) if m_port else "8080"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        spawned_proc = subprocess.Popen(
            [sys.executable, str(SCRIPTS_DIR / "serve.py"), "--model", args.model, "--port", port],
            cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            text=True, creationflags=flags,
        )
        for _ in range(90):
            if spawned_proc.poll() is not None:
                print(_c("31", "Failed to auto-start PicoCode server."))
                return 1
            if server_is_healthy(args.url)[0]:
                ok = True
                break
            time.sleep(1.0)
    if not ok:
        print(_c("31", f"Cannot connect to PicoCode server at {args.url}. Start it with `picocode.cmd --model {args.model}`."))
        return 1

    plan_only = bool(args.plan)
    auto_approve = bool(args.yes)
    conv: list[dict[str, Any]] = [{"role": "system", "content": _build_system_prompt(ws, plan_only)}]

    try:
        if args.prompt:
            conv.append({"role": "user", "content": args.prompt})
            run_agent_turn(
                args.url, args.model, ws, conv,
                plan_only=plan_only, auto_approve=auto_approve, max_steps=args.max_steps,
            )
            return 0

        print(_c("36", "============================================================"))
        print(_c("1", f"  PicoCode Terminal Agent v0.3.0  |  Model: {args.model}"))
        print(_c("90", f"  Workspace: {ws.root}"))
        print(_c("90", "  Commands: /help  /diff  /undo  /plan  /yes  /clear  /exit"))
        print(_c("36", "============================================================"))

        while True:
            mode_tag = _c("33", "[PLAN]") if plan_only else _c("32", "[AGENT]")
            try:
                user_in = input(f"\n{mode_tag} > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not user_in:
                continue
            if user_in in ("/exit", "/quit", "exit", "quit"):
                break
            if user_in == "/help":
                print("  /diff   - Show current git diff summary\n"
                      "  /undo   - Roll back all edits from the last agent turn\n"
                      "  /plan   - Toggle read-only Plan Mode\n"
                      "  /yes    - Toggle auto-approval for run_command\n"
                      "  /clear  - Reset conversation history\n"
                      "  /exit   - Quit agent")
                continue
            if user_in == "/clear":
                conv = [{"role": "system", "content": _build_system_prompt(ws, plan_only)}]
                print(_c("90", "  Conversation cleared."))
                continue
            if user_in == "/plan":
                plan_only = not plan_only
                conv[0] = {"role": "system", "content": _build_system_prompt(ws, plan_only)}
                print(_c("33", f"  Plan mode: {'ON (read-only)' if plan_only else 'OFF (full agent)'}"))
                continue
            if user_in == "/yes":
                auto_approve = not auto_approve
                print(_c("33", f"  Auto-approve shell commands: {'ON' if auto_approve else 'OFF'}"))
                continue
            if user_in == "/diff":
                _, out = _run_git(ws, ["diff", "--stat"])
                print(out or "  (no unstaged git diff)")
                continue
            if user_in == "/undo":
                n = ws.undo_turn()
                print(_c("32" if n else "33", f"  Reverted {n} edit(s) from last turn." if n else "  Nothing to undo."))
                continue

            conv.append({"role": "user", "content": user_in})
            run_agent_turn(
                args.url, args.model, ws, conv,
                plan_only=plan_only, auto_approve=auto_approve, max_steps=args.max_steps,
            )
    finally:
        if spawned_proc is not None and spawned_proc.poll() is None:
            spawned_proc.terminate()
            try:
                spawned_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                spawned_proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
