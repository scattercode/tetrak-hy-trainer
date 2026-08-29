#!/usr/bin/env python3
"""Prove the EasyOCR custom-model loading contract, end to end, without training.

The riskiest part of this project is not the model — it is the loading
contract: does a ``tetrak_hy`` bundle produced by our packaging actually
load and run through stock ``easyocr.Reader``? This spike answers that with
a *randomly initialised* model, so the contract is proven before any
training compute is spent. The output text is garbage by construction; what
is asserted is that every link in the chain holds:

  1. ``write_bundle`` emits tetrak_hy.yaml and tetrak_hy.py;
  2. a state dict for our (num_class, network_params) saves as tetrak_hy.pth
     in the format EasyOCR expects — **keys prefixed ``module.``**, because
     get_recognizer's CPU path strips the first seven characters of every
     key unconditionally (easyocr 1.7.2, verified in source);
  3. ``easyocr.Reader(['en'], recog_network='tetrak_hy', ...)`` imports the
     module, builds the model, loads the weights (``['en']``, not
     ``['hy']`` — see the comment at the call site);
  4. ``readtext`` runs the full detect-and-recognise pipeline over a real
     image and returns results whose text draws on our charset.

Prerequisites: pip install easyocr (torch arrives with it). First run
downloads EasyOCR's CRAFT detection weights (~80 MB).

Run:
    python scripts/spike_easyocr_loading.py
"""

from __future__ import annotations

import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tetrak_hy_trainer import charset, packaging  # noqa: E402


def build_random_state_dict():
    """A state dict for our architecture, keys in EasyOCR's expected format."""

    from easyocr.model.vgg_model import Model

    model = Model(num_class=charset.num_class(), **packaging.DEFAULT_NETWORK_PARAMS)
    # get_recognizer (CPU path) does `new_key = key[7:]` on every key -- the
    # DataParallel convention. Weights saved without the prefix would load
    # with every key mangled.
    return OrderedDict((f"module.{key}", value) for key, value in model.state_dict().items())


def spike_image(directory: Path) -> Path:
    """A small high-contrast test image with text-like content."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 120), "white")
    draw = ImageDraw.Draw(image)
    # Default font: the glyphs only need to *look like text* to CRAFT.
    # Recognition output is random-weight garbage either way.
    draw.text((20, 40), "ARMENIAN OCR SPIKE 123", fill="black")
    destination = directory / "spike.png"
    image.save(destination)
    return destination


def main() -> int:
    import easyocr
    import torch

    workdir = Path(tempfile.mkdtemp(prefix="tetrak_hy_spike_"))
    print(f"workdir: {workdir}")

    yaml_path, module_path = packaging.write_bundle(workdir)
    print(f"wrote {yaml_path.name}, {module_path.name}")

    weights = workdir / f"{packaging.NETWORK_NAME}.pth"
    torch.save(build_random_state_dict(), weights)
    print(f"wrote {weights.name} (random weights, {charset.num_class()} classes)")

    # ['en'], not ['hy'] -- a spike finding. setLanguageList reads
    # easyocr/character/<lang>_char.txt for every requested language and no
    # hy_char.txt ships with EasyOCR. The file's contents do not matter for a
    # custom model: the decode filter is set(model charset) - set(lang_char),
    # and lang_char always contains the yaml's full character_list, so the
    # filter is empty whichever language is requested. 'en' is authorised by
    # the yaml's lang_list and its char file exists.
    reader = easyocr.Reader(
        ["en"],
        recog_network=packaging.NETWORK_NAME,
        user_network_directory=str(workdir),
        model_storage_directory=str(workdir),
        gpu=False,
        verbose=False,
    )
    print("Reader constructed: module imported, yaml parsed, weights loaded")

    results = reader.readtext(str(spike_image(workdir)))
    print(f"readtext returned {len(results)} region(s)")

    allowed = set(charset.character_list())
    for _box, text, confidence in results:
        assert set(text) <= allowed, f"decoded characters outside the charset: {text!r}"
        print(f"  {confidence:.3f}  {text!r}")

    print("\nSPIKE PASSED: the tetrak_hy bundle loads and runs through stock EasyOCR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
