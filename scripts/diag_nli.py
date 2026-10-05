import json
import sys

from app.config import Settings
from app.rag.generator import _decouper_en_affirmations, _extraits, _preselectionner
from app.rag.nli import get_nli_provider

sys.stdout.reconfigure(encoding="utf-8")

s = Settings(); nli = get_nli_provider(s)
with open(".cache/traces_k5.jsonl", encoding="utf-8") as f:
    traces = [json.loads(ligne) for ligne in f]
for x in traces:
    if x["verdict_local"] not in ("refus_fidelite", "fondee") or not x["contexts"]:
        continue
    ex = _extraits(x["contexts"])
    print(f"===== {x['id']} {x['verdict_local']}")
    for a in _decouper_en_affirmations(x["reponse"]):
        sc = nli.evaluer(_preselectionner(ex, a), a)
        imp = max(t.implication for t in sc); con = max(t.contradiction for t in sc)
        tag = "OK" if imp >= s.seuil_nli_haut and con < s.seuil_nli_contradiction else "JUGE"
        print(f"  {tag:5} imp={imp:.2f} con={con:.2f} | {a[:140]}")
