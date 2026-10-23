"""Zero-shot pictured-subject CLASS check (CLIP, local ONNX, no API, no cost).

SCOPE: this is broad visual-class similarity, NOT verification of the full
query. Measured on Commons photos with the query "Clarence Seedorf Corinthians
2012": real Seedorf photos scored 0.93-0.96, but photos of OTHER footballers
(Kaka, Cristiano Ronaldo) scored 0.62-0.85 and Corinthians match photos
0.75-0.83, so they pass too. For "Apple iPhone launch": iPhone photos 0.92-0.95
pass, an old iPhone 0.68 passes, Samsung Galaxy 0.04-0.34 and a fruit bowl
0.01-0.32 fail. So it can reject a clearly wrong CLASS (fruit for a phone, a
chart for a photo) but cannot tell a specific person, team, phone model, year,
launch or event apart from look-alikes. A photo of an iPhone does not establish
"launch"; a photo of a footballer does not establish "Corinthians 2012".

The score is the softmax weight of the prompt among the prompt and nine generic
distractors (food, animal, landscape, building, vehicle, people, text
screenshot, chart/table, logo). It is NOT a calibrated probability and 0.5 is
not a confidence level. Acceptance therefore only means "no clearly wrong
class"; entity/date/event fit rests on source metadata, and the result is
recorded as VERIFICATION_SCOPE in the provenance sidecar. Fails CLOSED:
missing model, missing onnxruntime/tokenizers or any error means "not
verified".

Model: Xenova/clip-vit-base-patch32 (ONNX, quantized), directory from
AUTOSHORTS_CLIP_DIR (default ~/.cache/autoshorts/clip) holding
text_model_quantized.onnx, vision_model_quantized.onnx and tokenizer.json.
`download_model()` fetches them from huggingface.co on explicit request.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

MIN_PROBABILITY = 0.5  # softmax share, not a confidence level
VERIFICATION_SCOPE = "clip-broad-class-similarity (entity, date and event NOT verified)"
_FILES = ("text_model_quantized.onnx", "vision_model_quantized.onnx", "tokenizer.json")
_BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
_REMOTE = {
    "text_model_quantized.onnx": "onnx/text_model_quantized.onnx",
    "vision_model_quantized.onnx": "onnx/vision_model_quantized.onnx",
    "tokenizer.json": "tokenizer.json",
}
DISTRACTORS = (
    "a photo of food",
    "a photo of an animal",
    "a photo of a landscape",
    "a photo of a building",
    "a photo of a vehicle",
    "a photo of people",
    "a screenshot of text",
    "a chart or table of numbers",
    "a logo or icon",
)


class SubjectUnavailableError(RuntimeError):
    pass


def model_dir() -> Path:
    return Path(
        os.getenv("AUTOSHORTS_CLIP_DIR", str(Path.home() / ".cache/autoshorts/clip"))
    )


def download_model(target: Path | None = None) -> Path:
    import requests

    target = target or model_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        dest = target / name
        if dest.exists():
            continue
        resp = requests.get(_BASE + _REMOTE[name], timeout=300)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return target


@lru_cache(maxsize=1)
def _load(directory: str):
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SubjectUnavailableError(f"missing optional dependency: {exc}") from exc
    d = Path(directory)
    if not all((d / f).exists() for f in _FILES):
        raise SubjectUnavailableError(f"CLIP model files not found in {d}")
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    tok.enable_padding(length=77)
    tok.enable_truncation(77)
    return (
        tok,
        ort.InferenceSession(str(d / "text_model_quantized.onnx")),
        ort.InferenceSession(str(d / "vision_model_quantized.onnx")),
    )


def _norm(x):
    import numpy as np

    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def subject_probability(path: Path, subject: str) -> float:
    """P(the image shows `subject`) among the subject and generic distractors."""
    import numpy as np
    from PIL import Image

    tok, text_session, vision_session = _load(str(model_dir()))
    labels = [f"a photo of {subject}", *DISTRACTORS]
    enc = tok.encode_batch(labels)
    text = _norm(
        text_session.run(
            None, {"input_ids": np.array([e.ids for e in enc], dtype=np.int64)}
        )[0]
    )
    with Image.open(path) as raw:
        img = raw.convert("RGB")
    w, h = img.size
    scale = 224 / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = (
        np.asarray(img, dtype=np.float32) / 255
        - np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    ) / np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    vision = _norm(
        vision_session.run(
            None, {"pixel_values": arr.transpose(2, 0, 1)[None].astype(np.float32)}
        )[0]
    )
    logits = (100 * vision @ text.T)[0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return float(probs[0])


def check_subject(path: Path, subject: str) -> str | None:
    """None when the pixels show the subject; otherwise the rejection reason.

    Raises SubjectUnavailableError when the check cannot run (fail closed).
    """
    try:
        p = subject_probability(path, subject)
    except SubjectUnavailableError:
        raise
    except Exception as exc:
        raise SubjectUnavailableError(f"subject check failed: {exc}") from exc
    if p < MIN_PROBABILITY:
        return f"image does not look like {subject!r} (class score {p:.2f})"
    return None


# Generic scene/object words (en + pt-BR). A request made ONLY of these (plus
# function words) asks for a broad class, which the class check can meaningfully
# confirm. Anything else (a name, team, model, year, event, unknown word) is a
# specific request that no free local check can verify: web images are rejected
# for it and the pipeline falls back (AI image).
_FUNCTION_WORDS = frozenset(
    "a an the of in on at with and or for to from by de da do das dos e em no na "
    "nos nas com para por um uma o as os ao aos sobre over under near".split()
)
_GENERIC_WORDS = frozenset(
    """
    bowl cup glass plate up down open closed noite dia manhã manha tarde tigela copo
    prato photo image picture close closeup view scene background wide aerial night day
    sunrise sunset morning city street road bridge building house home room kitchen
    office desk table chair sofa bed window door garden park forest tree trees
    flower flowers mountain mountains beach sea ocean river lake sky clouds cloud
    rain snow field farm road car cars bus train plane airplane boat ship bicycle
    bike motorcycle truck dog dogs cat cats bird birds horse cow fish animal animals
    people person man woman child children crowd family hands hand team group
    food fruit fruits vegetables bread coffee tea water fire light lights
    computer laptop keyboard screen phone smartphone camera book books paper pen
    money coins chart graph server servers cable cables robot technology science
    laboratory factory machine machines tool tools work working
    foto imagem cena fundo cidade rua estrada ponte prédio predio casa quarto
    cozinha escritório escritorio mesa cadeira janela porta jardim parque floresta
    árvore arvore árvores arvores flor flores montanha praia mar oceano rio lago
    céu ceu nuvens chuva neve carro carros ônibus onibus trem avião aviao barco
    bicicleta moto caminhão caminhao cachorro gato pássaro passaro cavalo vaca peixe
    animal animais pessoas pessoa homem mulher criança crianca crianças criancas
    multidão multidao família familia mãos maos mão mao grupo comida fruta frutas
    legumes pão pao café cafe chá cha água agua fogo luz luzes computador teclado
    tela telefone celular câmera camera livro livros papel dinheiro moedas gráfico
    grafico servidor cabos robô robo tecnologia ciência ciencia laboratório
    laboratorio fábrica fabrica máquina maquina ferramenta trabalho
    """.split()
)


def is_broad_request(query: str) -> bool:
    """True only for plainly generic requests (no names, numbers or unknown words)."""
    import re

    tokens = re.findall(r"[^\W\d_]+|\d+", query or "", re.UNICODE)
    if not tokens:
        return False
    if any(t.isdigit() for t in tokens):
        return False
    for word in re.findall(r"[^\W\d_]+", query, re.UNICODE):
        if word[:1].isupper() and word.lower() not in _FUNCTION_WORDS:
            return False  # capitalized word: a proper name or title
        if word.lower() in _FUNCTION_WORDS:
            continue
        if word.lower() not in _GENERIC_WORDS:
            return False
    return any(w.lower() in _GENERIC_WORDS for w in tokens)
