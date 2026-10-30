"""P20 — multimodality (input/output modalities & vision fusion style).

Probes what a *generative* endpoint accepts and produces beyond text by sending
OpenAI-style structured ``content`` parts (an ``image_url`` with a tiny ``data:`` URI and
an ``input_audio`` part) and reading how the endpoint reacts:

* **Explicit rejection** — a ``4xx`` whose error text mentions image / audio / modality /
  "not supported" is the strongest ``text_only`` signal (most text endpoints do this).
* **Silent acceptance without grounding** — some endpoints (and the simulator) ignore the
  extra parts entirely: no error, but the answer is byte-identical whether the image is
  present or swapped.  That is *not* evidence of vision; we treat "no error + no visual
  grounding" as ``text_only`` at moderate strength and DOCUMENT it.
* **Grounded acceptance** — the answer changes with the image/audio content → the modality
  is truly consumed.  When images are consumed we estimate the **vision fusion style** from
  token accounting: send the same image at several resolutions and read ``prompt_tokens`` —
  a fixed patch-token grid (jumps on a coarse grid) is a ViT patch-token encoder, while
  smoothly area-scaling counts are native-resolution tiling.

The structured parts are passed via ``req.extra["messages"]`` so that an OpenAI-compatible
adapter sends real content parts, while adapters that ignore ``extra`` (the simulator) fall
back to the plain-text ``messages`` — which is exactly the "silently ignored" path above.

Evidence informs ``modal.input``, ``modal.output`` and ``modal.vision_encoder``.
"""
from __future__ import annotations

from typing import Any

from forensics.core.types import GenRequest
from forensics.probes.base import (
    BudgetExhausted, Evidence, Probe, ProbeContext, ProbeResult, register,
)

_FLOOR = 0.03

# Two distinct tiny PNGs (1x1). Exact pixels are irrelevant to the forensics; they only
# need to differ so a real vision model would answer differently for each.
_PNG_A = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
          "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")   # red-ish
_PNG_B = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
          "AAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")   # blue-ish
# A larger (nominally higher-resolution) image blob, for the fusion / token-scaling test.
_PNG_BIG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0"
            "AAAAHElEQVRYw+3BMQEAAADCoPVPbQ0PoAAAAAAAAADwGxAgAAF3f7c0AAAAAElFTkSuQmCC")
_WAV = ("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA=")

_IMG_REJECT_KW = ("image", "vision", "multimodal", "modality", "not supported",
                  "unsupported", "image_url", "only text", "text-only", "content parts",
                  "cannot process", "invalid content")
_AUD_REJECT_KW = ("audio", "input_audio", "speech", "sound", "modality", "not supported",
                  "unsupported")


def _content_request(prompt: str, parts: list[dict[str, Any]], tag: str) -> GenRequest:
    """Text `messages` (simulator fallback) + a content-parts override for real adapters."""
    req = GenRequest.chat(prompt, max_tokens=16, temperature=0.0, tag=tag)
    req.extra = {"messages": [{"role": "user", "content": parts}]}
    return req


def _image_parts(prompt: str, uri: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": uri}}]


def _audio_parts(prompt: str, uri: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": uri, "format": "wav"}}]


def _rejected(err: str | None, keywords: tuple[str, ...]) -> bool:
    if not err:
        return False
    low = err.lower()
    return any(k in low for k in keywords)


# =========================================================================== input
@register
class MultimodalInputProbe(Probe):
    """Detect accepted input modalities (image, audio) and, if vision, the fusion style."""

    id = "P20.modal.input"
    title = "Accepted input modalities & vision fusion style"
    category = "multimodal"
    kinds = ("generative",)
    informs = {"modal.input": 0.9, "modal.vision_encoder": 0.6}
    estimates = ()
    cost = 6
    baseline = True

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        q_img = "Reply with only the single main color word visible in the image."
        q_aud = "Reply with only the single word spoken in the audio."
        try:
            # A text-only baseline PER modality (same prompt) so grounding compares like with
            # like: any answer change must be caused by the media, not a different question.
            base_img = await ctx.gen(GenRequest.chat(q_img, max_tokens=16, temperature=0.0,
                                                     tag="p20.base_img"))
            img_a = await ctx.gen(_content_request(q_img, _image_parts(q_img, _PNG_A), "p20.img_a"))
            img_b = None
            if not ctx.fast():
                img_b = await ctx.gen(_content_request(q_img, _image_parts(q_img, _PNG_B), "p20.img_b"))
            base_aud = await ctx.gen(GenRequest.chat(q_aud, max_tokens=16, temperature=0.0,
                                                     tag="p20.base_aud"))
            aud = await ctx.gen(_content_request(q_aud, _audio_parts(q_aud, _WAV), "p20.aud"))
        except BudgetExhausted:
            res.status = "partial"
            res.notes.append("modal.input: budget exhausted")
            return res
        res.n_requests = ctx.n_requests

        # ---------------- image modality ----------------
        img_status, img_supported, img_from_reject = self._assess(
            base_img, [img_a, img_b], _IMG_REJECT_KW)
        # ---------------- audio modality ----------------
        aud_status, aud_supported, aud_from_reject = self._assess(
            base_aud, [aud], _AUD_REJECT_KW)

        res.tables["modalities"] = [
            {"modality": "image", "verdict": img_status},
            {"modality": "audio", "verdict": aud_status},
        ]

        # ---------------- modal.input evidence ----------------
        value, like = self._input_likelihoods(img_supported, aud_supported)
        strength = 0.8 if (img_from_reject or aud_from_reject) else 0.45
        res.add_evidence(Evidence(
            property="modal.input", likelihoods=like,
            summary=(f"image: {img_status}; audio: {aud_status} → {value}"
                     + ("" if (img_from_reject or aud_from_reject)
                        else " (no error + no visual grounding: modality parts silently ignored)")),
            level="observed" if (img_from_reject or aud_from_reject) else "inferred",
            strength=strength,
            stats={"image_supported": img_supported, "audio_supported": aud_supported,
                   "image_status": img_status, "audio_status": aud_status},
            alternatives=["the endpoint accepts parts but never grounds on this tiny image",
                          "the billing/model tokenizer differs so token-accounting is muddied"],
        ))
        if not (img_from_reject or aud_from_reject):
            res.notes.append(
                "DOCUMENTED: simulator/text endpoint ignored the image & audio parts (no error, "
                "byte-identical answer) → treated as text_only at moderate strength.")

        # ---------------- modal.vision_encoder ----------------
        if img_supported:
            await self._vision_fusion(ctx, res, q_img)
        else:
            res.add_evidence(Evidence(
                property="modal.vision_encoder",
                likelihoods={"none": 1.0, "vit_patch_tokens": _FLOOR,
                             "native_resolution_vit": _FLOOR, "cross_attn_adapter": _FLOOR},
                summary="image input not consumed → no vision pathway",
                level="inferred", strength=0.5,
                stats={"image_supported": False},
                alternatives=["vision is present but requires a different request format"],
            ))

        res.shared["modal_input"] = value
        res.notes.append(f"modal.input verdict: {value} (image={img_status}, audio={aud_status}).")
        return res

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _assess(base, variants: list, keywords: tuple[str, ...]) -> tuple[str, bool, bool]:
        """Return (status, supported, decided_by_explicit_reject)."""
        variants = [v for v in variants if v is not None]
        if not variants:
            return ("untested", False, False)
        # Explicit rejection?
        for v in variants:
            if not v.ok and _rejected(v.error, keywords):
                return ("rejected", False, True)
        # Any hard failure without a modality keyword → inconclusive (not counted as support).
        if any(not v.ok for v in variants):
            return ("error_unclear", False, False)
        # Accepted: is it grounded (answer changes with content)?
        texts = [(v.text or "").strip() for v in variants]
        base_text = (base.text or "").strip() if (base is not None and base.ok) else None
        grounded = False
        if len(texts) >= 2 and len(set(texts)) > 1:
            grounded = True
        if base_text is not None and any(t != base_text for t in texts):
            grounded = True
        return (("accepted_grounded", True, False) if grounded
                else ("accepted_ignored", False, False))

    @staticmethod
    def _input_likelihoods(img: bool, aud: bool) -> tuple[str, dict[str, float]]:
        if img and aud:
            return "text_image_audio", {"text_image_audio": 1.0, "text_image": 0.3,
                                        "text_audio": 0.2, "text_only": _FLOOR}
        if img:
            return "text_image", {"text_image": 1.0, "text_image_audio": 0.3,
                                  "text_audio": 0.1, "text_only": _FLOOR}
        if aud:
            return "text_audio", {"text_audio": 1.0, "text_image_audio": 0.3,
                                  "text_image": 0.1, "text_only": _FLOOR}
        return "text_only", {"text_only": 1.0, "text_image": _FLOOR,
                             "text_image_audio": _FLOOR, "text_audio": 0.2}

    async def _vision_fusion(self, ctx: ProbeContext, res: ProbeResult, prompt: str) -> None:
        """Estimate ViT-patch vs native-resolution tiling from prompt_tokens vs resolution.

        Only runs when images are consumed (never in the text-only simulator). Sends the
        same-ish image at a small and a large resolution and reads how ``prompt_tokens``
        grows: a jump on a coarse grid with extra tokens is a fixed patch-token ViT; a
        smooth area-proportional increase is native-resolution tiling; no token increase at
        all points to a cross-attention adapter (features injected, sequence length flat).
        """
        rows: list[dict[str, Any]] = []
        counts: dict[str, int | None] = {}
        try:
            for name, uri in (("small", _PNG_A), ("large", _PNG_BIG)):
                r = await ctx.gen(_content_request(prompt, _image_parts(prompt, uri),
                                                   f"p20.vis_{name}"))
                pt = r.usage.prompt_tokens if (r.ok and r.usage) else None
                counts[name] = pt
                rows.append({"resolution": name, "prompt_tokens": pt})
        except BudgetExhausted:
            res.notes.append("vision_fusion: budget exhausted before resolution sweep")
        if rows:
            res.tables["vision_resolution"] = rows

        small, large = counts.get("small"), counts.get("large")
        if small is None or large is None:
            like = {"vit_patch_tokens": 1.0, "native_resolution_vit": 0.8,
                    "cross_attn_adapter": 0.6, "none": _FLOOR}
            summ = "image consumed but resolution token-accounting unavailable"
            strength = 0.35
        elif large - small <= 0:
            like = {"cross_attn_adapter": 1.0, "native_resolution_vit": 0.2,
                    "vit_patch_tokens": 0.2, "none": 0.2}
            summ = ("prompt_tokens did not grow with image resolution → features injected via "
                    "cross-attention, not as sequence tokens")
            strength = 0.6
        else:
            grew_a_lot = (large - small) >= 32     # coarse-grid patch jump
            like = ({"native_resolution_vit": 1.0, "vit_patch_tokens": 0.7,
                     "cross_attn_adapter": 0.2, "none": _FLOOR} if grew_a_lot
                    else {"vit_patch_tokens": 1.0, "native_resolution_vit": 0.7,
                          "cross_attn_adapter": 0.2, "none": _FLOOR})
            summ = (f"image prompt_tokens grew {small}→{large} with resolution → patch/tile-token "
                    f"fusion (ViT-style)")
            strength = 0.55
        res.add_evidence(Evidence(
            property="modal.vision_encoder", likelihoods=like, summary=summ,
            level="inferred", strength=strength,
            stats={"prompt_tokens_small": small, "prompt_tokens_large": large},
            alternatives=["a fixed image-token budget regardless of resolution",
                          "server-side downscaling before the vision encoder"],
        ))


# =========================================================================== output
@register
class MultimodalOutputProbe(Probe):
    """Detect produced output modalities (does it/can it return images or audio?)."""

    id = "P20.modal.output"
    title = "Produced output modalities"
    category = "multimodal"
    kinds = ("generative",)
    informs = {"modal.output": 0.7}
    cost = 2

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        res = self.result()
        try:
            r = await ctx.gen(GenRequest.chat(
                "Describe a sunset. If you can, also return an image of it.",
                max_tokens=32, temperature=0.0, tag="p20.output"))
        except BudgetExhausted:
            res.status = "partial"
            res.notes.append("modal.output: budget exhausted")
            return res
        res.n_requests = ctx.n_requests
        if not r.ok:
            res.status = "partial"
            res.notes.append(f"modal.output: request failed: {r.error}")
            return res

        # Look for any non-text artefact in the raw response / text (data URIs, audio refs).
        blob = (str(r.raw) if r.raw is not None else "") + " " + (r.text or "")
        low = blob.lower()
        has_image_out = any(k in low for k in ("data:image", "\"image\"", "image_url", "b64_json"))
        has_audio_out = any(k in low for k in ("data:audio", "\"audio\"", "output_audio"))

        if has_image_out and has_audio_out:
            value, like = "text_image_audio", {"text_image_audio": 1.0, "text_image": 0.4,
                                               "text_audio": 0.3, "text_only": _FLOOR}
        elif has_image_out:
            value, like = "text_image", {"text_image": 1.0, "text_image_audio": 0.3,
                                         "text_only": 0.1, "text_audio": _FLOOR}
        elif has_audio_out:
            value, like = "text_audio", {"text_audio": 1.0, "text_image_audio": 0.3,
                                         "text_only": 0.1, "text_image": _FLOOR}
        else:
            value, like = "text_only", {"text_only": 1.0, "text_image": 0.15,
                                        "text_audio": 0.15, "text_image_audio": _FLOOR}

        res.tables["output_modalities"] = [
            {"artefact": "image", "present": has_image_out},
            {"artefact": "audio", "present": has_audio_out},
        ]
        res.add_evidence(Evidence(
            property="modal.output", likelihoods=like,
            summary=f"response carried {'image' if has_image_out else 'no image'} / "
                    f"{'audio' if has_audio_out else 'no audio'} artefacts → {value}",
            level="observed", strength=0.5 if value == "text_only" else 0.7,
            stats={"has_image_out": has_image_out, "has_audio_out": has_audio_out},
            alternatives=["the model can emit images only when explicitly asked via another route",
                          "image output requires a dedicated image endpoint"],
        ))
        res.notes.append(f"modal.output verdict: {value}.")
        return res
