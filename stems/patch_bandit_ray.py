#!/usr/bin/env python3
"""Macht den ray-Import in bandit-v2 `src/system/utils.py` optional.

`ray` ist in bandit-v2 ein reines TRAININGS-Paket (verteiltes Training via Ray
Train). Auf Modul-Ebene steht dort `from ray.train.lightning import (...)`, was
beim blossen Import von inference.py `ModuleNotFoundError: No module named 'ray'`
ausloest. Empirisch verifiziert: Der INFERENZ-Pfad (inference.py -> build_system)
ruft keines der vier Ray-Symbole auf. Wir kapseln den Import daher in try/except
und setzen die Symbole im Fehlerfall auf None. ray wird NICHT installiert.

Idempotent (Marker-Kommentar). Bricht ab, wenn ein ray-Import existiert, dessen
Form wir nicht kennen (Upstream-Aenderung) -> dann bewusst pruefen.
Aufruf: python3 patch_bandit_ray.py [pfad-zu-utils.py]
"""
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/opt/bandit-v2/src/system/utils.py"
MARKER = "# BANDIT_INFERENCE_RAY_OPTIONAL_PATCH"

with open(PATH, encoding="utf-8") as f:
    s = f.read()

if MARKER in s:
    print("ray-Patch bereits vorhanden -> uebersprungen")
    sys.exit(0)

# 1) mehrzeiliger Import-Block:  from ray.train.lightning import ( ... )
m = re.search(r"from ray\.train\.lightning import \([^)]*\)", s, re.DOTALL)
# 2) Fallback: einzeiliger Import
if not m:
    m = re.search(r"^from ray\.train\.lightning import .+$", s, re.MULTILINE)

if not m:
    if "import ray" not in s and "from ray" not in s:
        print("kein ray-Import gefunden -> nichts zu patchen")
        sys.exit(0)
    raise SystemExit("ray-Import vorhanden, aber unbekannte Form -> Patch pruefen!")

block = m.group(0)
indented = "\n".join(("    " + ln) if ln.strip() else ln for ln in block.splitlines())
new = (
    f"try:  {MARKER}\n"
    f"{indented}\n"
    "except ModuleNotFoundError:  # ray nur fuer Training, nicht fuer Inferenz\n"
    "    RayDDPStrategy = RayLightningEnvironment = RayTrainReportCallback = prepare_trainer = None"
)
with open(PATH, "w", encoding="utf-8") as f:
    f.write(s.replace(block, new, 1))
print("ray-Patch angewendet")
