import json
import re
import sys
import time

import torch

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from app.config import Settings
from app.rag.bm25_index import tokeniser_fr
from app.rag.nli import LocalNLI

# Diagnostic du garde-fou étage 2, version rapide : au lieu de confronter
# chaque affirmation à TOUS les groupes de phrases de l'article (des dizaines
# de milliers de passages NLI), une présélection lexicale (mots communs,
# tokenizer FR du BM25) ne garde que les TOP_K groupes les plus proches.
TOP_K = 8
s = Settings(); nli = LocalNLI(s.nli_model, 512); nli._charger()  # float32 depuis le correctif de nli.py
tok, mod, ie = nli._tokenizer, nli._model, nli._indice_entailment

def phrases_source(t):
    # phrases de l'article : coupe aussi aux retours à la ligne (alinéas)
    out=[]
    for l in re.split(r"\n+", t):
        out += [p.strip() for p in re.split(r"(?<=[.!?;:])\s+", l) if p.strip()]
    return out
def fenetres_source(t, n=3):
    p = phrases_source(t)
    if len(p) <= n: return [" ".join(p)]
    return [" ".join(p[i:i+n]) for i in range(len(p)-n+1)]

ITEM = re.compile(r"^\s*(?:\d+[.)]|[-•*])\s+")
OUIN = re.compile(r"^(?:oui|non)\s*[,.:]\s*", re.IGNORECASE)
def affirmations(rep):
    res=[]; intro=None
    for ligne in [l for l in rep.split("\n") if l.strip()]:
        if ITEM.match(ligne):
            item = ITEM.sub("", ligne).strip()
            res.append(f"{intro} {item}" if intro else item); continue
        for p in [x.strip() for x in re.split(r"(?<=[.!?])\s+", ligne) if x.strip()]:
            if p.endswith(":"): intro = p[:-1].strip(); continue
            intro=None; res.append(p)
    return [OUIN.sub("", a) for a in res if len(a) > 3]

def score(ctx, a):
    f = [w for c in ctx for w in fenetres_source(c)]
    mots = set(tokeniser_fr(a))
    prem = sorted(f, key=lambda w: -len(mots & set(tokeniser_fr(w))))[:TOP_K]
    e = tok(prem, [a]*len(prem), truncation="only_first", max_length=512, padding=True, return_tensors="pt")
    with torch.no_grad():
        return float(torch.softmax(mod(**e).logits, -1)[:, ie].max())

with open(".cache/eval_traces.jsonl",encoding="utf-8") as f:
    traces=[json.loads(l) for l in f]
H=s.seuil_nli_haut
res={}; t0=time.time()
cas=[t for t in traces if t["verdict_local"] in ("refus_fidelite","fondee")]
for n,t in enumerate(cas,1):
    sc=[(a,score(t["contexts"],a)) for a in affirmations(t["reponse"])]
    mn=min(x for _,x in sc) if sc else 1
    v="OK" if mn>=H else "JUGE"
    res.setdefault((t["verdict_local"],v),0); res[(t["verdict_local"],v)]+=1
    pire=min(sc,key=lambda z:z[1]) if sc else ("",1)
    print(f"[{n}/{len(cas)} {time.time()-t0:.0f}s] {t['id']} ({t['verdict_local']}) -> {v} min={mn:.2f} nb_aff={len(sc)} | pire: « {pire[0][:90]} »")
print(res)
# contrôles négatifs : affirmations fausses qui DOIVENT échouer
neg={"q04":"Le dépôt de garantie ne peut excéder deux mois de loyer en principal.",
     "q15":"Aucun état des lieux n'est exigé à l'entrée dans les lieux.",
     "q26":"Un agent immobilier peut exercer sans carte professionnelle.",
     "q11":"Les charges locatives ne sont jamais régularisées.",
     "q24":"Les travaux d'économies d'énergie peuvent être décidés par le syndic seul."}
for i,a in neg.items():
    t=next(x for x in traces if x["id"]==i); print("NEG",i,f"{score(t['contexts'],a):.2f}",a)
print(f"durée totale : {time.time()-t0:.0f}s")
