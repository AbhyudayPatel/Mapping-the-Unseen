"""Tokenizer registry: identify and lazily load the tokenizers in the DB.

Loads the frozen ``tokenizers.json`` (65 entries) and exposes encoders that can
turn text into token ids / counts.  This powers the tokenizer-fingerprint probe,
which encodes probe strings with every loadable tokenizer and compares the token
counts against the API's ``usage.prompt_tokens``.

Loading strategy (best effort, never raises for a missing/gated tokenizer):

* ``tiktoken`` encodings bundled with the ``tiktoken`` package
  (``r50k_base``, ``p50k_base``, ``cl100k_base``, ``o200k_base``,
  ``o200k_harmony``) always work **offline** — these are the guaranteed fallback set.
* HuggingFace ``tokenizers`` repos load via ``Tokenizer.from_pretrained`` and are
  cached under ``<project>/.cache/tok/``.  Network / gated failures are caught and
  the entry is marked *unavailable*.
* A custom ``tiktoken.model`` (Kimi) is loaded through ``tiktoken.Encoding`` when
  the file can be fetched; otherwise it is skipped.

Only *currently loadable* tokenizers participate in :meth:`count_all` and
:meth:`candidates`, so those calls are fast and deterministic (no implicit
network fan-out).  Use :meth:`prefetch` to warm the downloadable ones.
"""
from __future__ import annotations

import os
import warnings

import json
import os
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------- paths
_HERE = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = _HERE / "tokenizers.json"
CACHE_DIR = _ROOT / ".cache" / "tok"

# Redirect the HuggingFace cache into the project cache dir (politely: only if the
# user has not configured their own).  Done before ``tokenizers`` touches the network.
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
except OSError:
    pass


class TokEncoder:
    """A loaded tokenizer: ``encode(text) -> list[int]`` and ``count(text) -> int``."""

    def __init__(self, family: str, kind: str, encode_fn: Callable[[str], Any],
                 vocab_size: int | None = None):
        self.family = family
        self.kind = kind                      # "tiktoken" | "hf"
        self.vocab_size = vocab_size
        self._encode = encode_fn

    def encode(self, text: str) -> list[int]:
        return list(self._encode(text))

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def __repr__(self) -> str:  # pragma: no cover
        return f"TokEncoder(family={self.family!r}, kind={self.kind})"


class TokenizerRegistry:
    """Index over ``tokenizers.json`` with lazy, cached encoder loading."""

    _cache: dict[str, "TokenizerRegistry"] = {}

    def __init__(self, entries: list[dict[str, Any]]):
        self._entries: list[dict[str, Any]] = list(entries)
        self._loaded: dict[str, TokEncoder] = {}    # family -> encoder (success cache)
        self._failed: set[str] = set()              # families that failed to load
        self._index: dict[str, dict[str, Any]] | None = None
        self._prefetched = False

    # --------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path | None = None) -> "TokenizerRegistry":
        """Load (and cache) the tokenizer registry."""
        p = Path(path) if path else DEFAULT_PATH
        key = str(p.resolve())
        if key not in cls._cache:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError(f"expected a list of records in {p}, got {type(data)}")
            cls._cache[key] = cls(data)
        return cls._cache[key]

    # --------------------------------------------------------------- access
    def all(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def families(self) -> list[str]:
        return [str(e.get("family")) for e in self._entries if e.get("family")]

    def offline_families(self) -> list[str]:
        """Families guaranteed to load offline (bundled tiktoken encodings)."""
        return [str(e["family"]) for e in self._entries
                if e.get("loads_with_tiktoken_get_encoding") and e.get("family")]

    def candidates(self) -> list[str]:
        """All loadable families. The first call prefetches every registry tokenizer
        (downloads are cached under ``.cache/tok``; ~40 s once, instant afterwards).

        Comparing only against the bundled tiktoken encodings would make every model look
        like an OpenAI tokenizer — the DeepSeek-V3.2 live run did exactly that. Set
        ``FORENSICS_OFFLINE=1`` to restrict to already-cached / bundled tokenizers."""
        if not self._prefetched:
            self._prefetched = True
            offline = os.environ.get("FORENSICS_OFFLINE", "").strip() not in ("", "0", "false")
            self.prefetch(download=True, quiet=True, offline=offline)
        fams = set(self.offline_families())
        fams.update(self._loaded.keys())
        return sorted(fams)

    # --------------------------------------------------------------- lookup
    def _build_index(self) -> dict[str, dict[str, Any]]:
        idx: dict[str, dict[str, Any]] = {}

        def put(k: Any, e: dict[str, Any]) -> None:
            if k:
                idx.setdefault(str(k).lower(), e)

        for e in self._entries:
            fam = e.get("family")
            put(fam, e)
            if fam and str(fam).startswith("tiktoken:"):
                put(str(fam).split(":", 1)[1], e)   # bare encoding name
            put(e.get("repo_or_tiktoken_name"), e)
            for m in (e.get("used_by") or []):
                put(m, e)
            for r in (e.get("alt_repos") or []):
                put(r, e)
        return idx

    def _resolve(self, key: str) -> dict[str, Any] | None:
        if self._index is None:
            self._index = self._build_index()
        return self._index.get(key.lower())

    def get(self, family_or_repo: str) -> dict[str, Any] | None:
        """Return the raw DB record for a family/repo/model alias, or ``None``."""
        return self._resolve(str(family_or_repo))

    # --------------------------------------------------------------- encoders
    def get_encoder(self, family_or_repo: str) -> TokEncoder | None:
        """Return a callable encoder for a family/repo/model name, or ``None``.

        The result is cached (both success and failure).  Loading is lazy and
        never raises: unavailable tokenizers (network/gated/custom) yield ``None``.
        """
        key = str(family_or_repo).strip()
        entry = self._resolve(key)
        if entry is not None:
            fam = str(entry.get("family"))
            if fam in self._loaded:
                return self._loaded[fam]
            if fam in self._failed:
                return None
            enc = self._load_entry(entry)
            if enc is None:
                self._failed.add(fam)
                return None
            self._loaded[fam] = enc
            return enc
        # No DB entry: allow a bare tiktoken encoding name (e.g. "o200k_base").
        return self._load_tiktoken_name(key)

    def _load_entry(self, entry: dict[str, Any]) -> TokEncoder | None:
        fam = str(entry.get("family"))
        typ = str(entry.get("type") or "").lower()
        repo = entry.get("repo_or_tiktoken_name")
        vsz = entry.get("vocab_size")

        # 1. bundled tiktoken (offline-safe)
        if entry.get("loads_with_tiktoken_get_encoding") and repo:
            enc = self._tiktoken_get(str(repo))
            if enc is not None:
                return TokEncoder(fam, "tiktoken", enc.encode, vsz)

        # 2. custom tiktoken.model (Kimi) -> needs a download
        if entry.get("loads_with_tiktoken_custom") and repo:
            return self._load_custom_tiktoken(fam, str(repo), vsz)

        # 3. HuggingFace tokenizers repo
        if entry.get("loads_with_tokenizers_from_pretrained") and repo:
            return self._load_hf(fam, str(repo), vsz)

        # 4. tiktoken mirror stored as an HF repo (e.g. Xenova/gpt-4 -> cl100k)
        if "tiktoken" in typ and repo and "/" in str(repo):
            return self._load_hf(fam, str(repo), vsz)

        # 5. last resort: repo is itself a tiktoken encoding name
        if repo:
            enc = self._tiktoken_get(str(repo))
            if enc is not None:
                return TokEncoder(fam, "tiktoken", enc.encode, vsz)

        return None  # sentencepiece / custom / no loadable repo -> unavailable

    def _load_tiktoken_name(self, name: str) -> TokEncoder | None:
        for candidate in (name, f"{name}_base"):
            enc = self._tiktoken_get(candidate)
            if enc is not None:
                fam = f"tiktoken:{candidate}"
                self._loaded.setdefault(fam, TokEncoder(fam, "tiktoken", enc.encode))
                return self._loaded[fam]
        return None

    @staticmethod
    def _tiktoken_get(name: str):
        try:
            import tiktoken
            if name not in tiktoken.list_encoding_names():
                return None
            return tiktoken.get_encoding(name)
        except Exception:
            return None

    def _load_hf(self, fam: str, repo: str, vsz: int | None) -> TokEncoder | None:
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_pretrained(repo)
            return TokEncoder(fam, "hf", lambda t: tok.encode(t).ids, vsz)
        except Exception:
            return None

    def _load_custom_tiktoken(self, fam: str, repo: str, vsz: int | None) -> TokEncoder | None:
        """Best-effort load of a repo-hosted ``tiktoken.model`` (e.g. Kimi)."""
        try:
            import tiktoken
            from tiktoken.load import load_tiktoken_bpe
            from huggingface_hub import hf_hub_download
            repo_id, _, filename = repo.partition(":")
            filename = filename or "tiktoken.model"
            path = hf_hub_download(repo_id, filename, cache_dir=str(CACHE_DIR / "hub"))
            ranks = load_tiktoken_bpe(path)
            # A generic GPT-style split pattern; adequate for count fingerprints.
            pat = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"""
                   r"""(?!\S)|\s+""")
            enc = tiktoken.Encoding(name=fam, pat_str=pat,
                                    mergeable_ranks=ranks, special_tokens={})
            return TokEncoder(fam, "tiktoken", enc.encode, vsz)
        except Exception:
            return None

    # --------------------------------------------------------------- bulk ops
    def prefetch(self, families: list[str] | None = None, download: bool = True,
                 quiet: bool = True, offline: bool = False) -> list[str]:
        """Eagerly load encoders (graceful skip on failure); return loaded families.

        ``download=False`` loads only the bundled tiktoken set (no network).
        ``offline=True`` allows HF tokenizers already in the local cache but never
        touches the network (``HF_HUB_OFFLINE``).
        """
        want = set(families) if families else None
        loaded: list[str] = []
        env_backup = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE",
                                                     "HF_HUB_DISABLE_SYMLINKS_WARNING")}
        if quiet:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            with warnings.catch_warnings():
                if quiet:
                    warnings.simplefilter("ignore")
                for e in self._entries:
                    fam = str(e.get("family"))
                    if want is not None and fam not in want:
                        continue
                    if not download and not e.get("loads_with_tiktoken_get_encoding"):
                        continue
                    if self.get_encoder(fam) is not None:
                        loaded.append(fam)
        finally:
            for k, v in env_backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return loaded

    def count_all(self, text: str) -> dict[str, int]:
        """Token counts for ``text`` from every *currently loadable* tokenizer.

        Always includes the bundled tiktoken encodings; includes any HF/custom
        tokenizers already loaded (e.g. via :meth:`prefetch` or :meth:`get_encoder`).
        """
        out: dict[str, int] = {}
        for fam in self.candidates():
            enc = self.get_encoder(fam)
            if enc is not None:
                try:
                    out[fam] = enc.count(text)
                except Exception:
                    continue
        return out


def load(path: str | Path | None = None) -> TokenizerRegistry:
    """Convenience module-level loader (see :meth:`TokenizerRegistry.load`)."""
    return TokenizerRegistry.load(path)
