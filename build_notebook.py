#!/usr/bin/env python3
"""Genera il notebook didattico 'panorama_download.ipynb'.
Esegui con il python del venv:  .venv/bin/python build_notebook.py
"""
import nbformat as nbf

# Le celle sono separate da ===MD=== (markdown) e ===CODE=== (codice).
# Il blocco e' una r-string con apici singoli tripli: il codice interno usa
# solo virgolette doppie/triple-doppie, quindi non ci sono conflitti.
SRC = r'''
===MD===
# 📥 PANORAMA – Download selettivo di esami TC del pancreas

Questo notebook scarica un **sottoinsieme a scelta** del dataset [PANORAMA](https://panorama.grand-challenge.org/)
(TC con contrasto del pancreas, per lo studio del tumore PDAC) **senza dover scaricare i ~194 GB interi**.

**Come funziona (in breve):**
1. Le immagini stanno in 4 archivi `.zip` su Zenodo. Grazie alle richieste *HTTP-range* possiamo
   estrarre **solo i singoli esami che scegliamo**, senza scaricare lo zip intero.
2. Le **segmentazioni** (maschere) stanno in un repo GitHub separato e sono piccole.
3. Alla fine viene creato un **file Excel** che riassume cosa hai scaricato, con i dati clinici e
   il dettaglio delle segmentazioni disponibili (quali distretti anatomici, manuale o AI).

> ⚠️ **Licenza dati: CC BY-NC 4.0** → solo uso **non commerciale**. Cita lo studio PANORAMA (Alves et al. 2024).

> 🧠 **Kernel:** assicurati in alto a destra che sia selezionato **"PANORAMA PDAC (.venv 3.12)"**.

> ♻️ **Resumabile:** se interrompi, basta ri-eseguire: i file già scaricati vengono saltati.
===MD===
## 1) Configurazione

Qui imposti **quanto** scaricare, con un solo parametro `MODE`:

| `MODE` | Cosa fa | Parametro |
|---|---|---|
| `"percent"` | una **percentuale** del dataset (2238 casi), divisa 50/50 PDAC/controlli | `TARGET_PERCENT` |
| `"count"` | numeri espliciti | `N_PDAC`, `N_CONTROL` |
| `"all"` | **tutti** i casi (676 PDAC + 1562 controlli → NON bilanciato) | — |
| `"explicit"` | solo gli `study_id` elencati | `STUDY_IDS` |

**📈 Download incrementale (cumulativo):** la selezione è *annidata*, cioè con lo stesso `SEED`
una run all'1% è **sempre contenuta** in una al 5%, che è contenuta nel 25%, e così via.
Quindi puoi alzare `TARGET_PERCENT` giorno dopo giorno e il notebook **scarica solo i nuovi**
casi (salta quelli già presenti). Quando vuoi tutto, metti `MODE = "all"`.

> ⚠️ Per il download incrementale **non cambiare** `SEED`, `CONTROL_STRATEGY` o `PDAC_PREFER_MANUAL`
> tra una run e l'altra: cambierebbero quali casi vengono scelti (e l'annidamento si romperebbe).
===CODE===
import os

# --- COSA scaricare ---------------------------------------------------------

# Come dimensionare la selezione (UN solo parametro):
#   "percent"  -> TARGET_PERCENT % del dataset (2238 casi), diviso 50/50 PDAC/controlli
#   "count"    -> numeri espliciti N_PDAC / N_CONTROL
#   "all"      -> TUTTI i casi (676 PDAC + 1562 non-PDAC): NON bilanciato (piu' controlli)
#   "explicit" -> esattamente gli study_id elencati in STUDY_IDS
MODE = os.environ.get("PANORAMA_MODE", "percent")

# Usato se MODE="percent". E' una percentuale CUMULATIVA: alzala tra una run e l'altra
# (es. 1 -> 5 -> 25) e il notebook scarichera' solo i nuovi casi, senza riscaricare i vecchi.
TARGET_PERCENT = float(os.environ.get("PANORAMA_TARGET_PERCENT", 1.0))

# Usati se MODE="count"
N_PDAC    = int(os.environ.get("PANORAMA_N_PDAC", 150))
N_CONTROL = int(os.environ.get("PANORAMA_N_CONTROL", 150))

# Come scegliere i CONTROLLI ("sani"):
#   "sure_negatives" -> prima gli 80 casi NIH (pancreas normale), poi i negativi
#                       radiologici con follow-up, poi gli altri negativi radiologici.
#                       (= negativi piu' affidabili, esclude altre patologie)
#   "nih_only"       -> solo i casi NIH (massimo ~80)
#   "any_nonpdac"    -> un campione casuale qualunque tra i non-PDAC
CONTROL_STRATEGY = os.environ.get("PANORAMA_CONTROL_STRATEGY", "sure_negatives")

# Per i PDAC: se True, scegli PRIMA i casi con segmentazione della LESIONE fatta
# MANUALMENTE da esperto (cartella 'manual_labels'); solo se ne servono di piu' usa
# quelli con lesione AI. Ci sono ~482 PDAC con lesione manuale (su 676): quindi con
# N_PDAC <= 482 TUTTI i PDAC scelti avranno la lesione annotata a mano (gold standard).
PDAC_PREFER_MANUAL = os.environ.get("PANORAMA_PDAC_PREFER_MANUAL", "1") == "1"

# Solo per MODE="explicit": elenco di study_id, es. ["100002_00001", "100010_00001"]
STUDY_IDS = []

# Seme casuale: rende la selezione RIPRODUCIBILE (stesso seme = stessi casi)
SEED = int(os.environ.get("PANORAMA_SEED", 42))

# --- DOVE salvare -----------------------------------------------------------
PROJECT_DIR = os.path.abspath(".")
IMAGES_DIR  = os.path.join(PROJECT_DIR, "imagesTr")   # volumi TC (.nii.gz)
LABELS_DIR  = os.path.join(PROJECT_DIR, "labelsTr")   # maschere di segmentazione (.nii.gz)
CACHE_DIR   = os.path.join(PROJECT_DIR, "cache")      # file di appoggio (indice, info cliniche)
REPORT_XLSX = os.path.join(PROJECT_DIR, "panorama_report.xlsx")

# Avvisa se lo spazio libero stimato scende sotto questa soglia (GB)
MIN_FREE_GB = float(os.environ.get("PANORAMA_MIN_FREE_GB", 5))

print("Configurazione caricata:")
print(f"  MODE = {MODE}")
if MODE == "percent":
    print(f"  TARGET_PERCENT = {TARGET_PERCENT}%  (del dataset, diviso 50/50)")
elif MODE == "count":
    print(f"  N_PDAC = {N_PDAC} | N_CONTROL = {N_CONTROL}")
print(f"  strategia controlli = {CONTROL_STRATEGY} | PDAC manuali per primi = {PDAC_PREFER_MANUAL}")
print(f"  SEED = {SEED}")
print(f"  Output in: {PROJECT_DIR}")
===MD===
## 2) Controlli ambiente e import

Verifichiamo Python, i pacchetti necessari e prepariamo le cartelle.
Se qualcosa manca, qui viene segnalato chiaramente.
===CODE===
import sys, time, json, gzip, struct, shutil, random
from collections import Counter

print("Python:", sys.version.split()[0], "| eseguibile:", sys.executable)
if sys.version_info[:2] != (3, 12):
    print("⚠️ ATTENZIONE: questo notebook e' pensato per Python 3.12 "
          "(kernel 'PANORAMA PDAC (.venv 3.12)'). Controlla il kernel in alto a destra.")

# Verifica pacchetti
mancanti = []
for pkg in ["remotezip", "requests", "pandas", "numpy", "nibabel", "openpyxl"]:
    try:
        __import__(pkg)
    except ImportError:
        mancanti.append(pkg)
if mancanti:
    raise ImportError("Pacchetti mancanti: " + ", ".join(mancanti) +
                      "\n-> Installa con:  pip install " + " ".join(mancanti))

import requests, pandas as pd, numpy as np, nibabel as nib
from remotezip import RemoteZip
print("✅ Tutti i pacchetti necessari sono presenti.")

# Prepara cartelle
for d in (IMAGES_DIR, LABELS_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)
print("✅ Cartelle pronte.")
===MD===
## 3) Costanti del dataset

Indirizzi degli archivi Zenodo, del repo delle segmentazioni e la **legenda delle classi**
delle maschere (ogni valore di voxel = un distretto anatomico).
===CODE===
# Le immagini: 4 archivi zip su Zenodo (uno per "batch")
ZENODO_BATCHES = {
    "batch_1": "https://zenodo.org/records/13715870/files/batch_1.zip",
    "batch_2": "https://zenodo.org/records/13742336/files/batch_2.zip",
    "batch_3": "https://zenodo.org/records/11034011/files/batch_3.zip",
    "batch_4": "https://zenodo.org/records/10999754/files/batch_4.zip",
}

# Le segmentazioni: repo GitHub DIAGNijmegen/panorama_labels
GH_RAW  = "https://raw.githubusercontent.com/DIAGNijmegen/panorama_labels/main/"
GH_TREE = "https://api.github.com/repos/DIAGNijmegen/panorama_labels/git/trees/main?recursive=1"
CLINICAL_XLSX_URL = GH_RAW + "clinical_information.xlsx"

# Legenda classi delle maschere multi-classe
LABEL_CLASSES = {
    0: "Sfondo",
    1: "Lesione PDAC (tumore)",
    2: "Vene",
    3: "Arterie",
    4: "Parenchima pancreatico",
    5: "Dotto pancreatico",
    6: "Coledoco",
}
# Nome breve per le colonne del report (una colonna per distretto)
DISTRICT_SHORT = {1: "lesione", 2: "vene", 3: "arterie", 4: "pancreas", 5: "dotto", 6: "coledoco"}

print("Distretti segmentabili:")
for k, v in LABEL_CLASSES.items():
    print(f"  {k} = {v}")
===MD===
## 4) Informazioni cliniche

Scarichiamo `clinical_information.xlsx` dal repo: contiene per ogni esame l'etichetta
**PDAC / non-PDAC**, il *reference standard* (come e' stata stabilita la diagnosi), eta', sesso, scanner.
===CODE===
def carica_info_cliniche():
    """Scarica (se serve) e legge clinical_information.xlsx; indicizza per study_id."""
    local = os.path.join(CACHE_DIR, "clinical_information.xlsx")
    if not os.path.exists(local):
        print("Scarico clinical_information.xlsx ...")
        r = requests.get(CLINICAL_XLSX_URL, timeout=60); r.raise_for_status()
        with open(local, "wb") as f:
            f.write(r.content)
    df = pd.read_excel(local)
    attese = {"PANORAMA_study_id", "patient_age", "patient_sex", "scanner", "label", "level"}
    mancanti = attese - set(df.columns)
    if mancanti:
        print("⚠️ ATTENZIONE: colonne mancanti nel file clinico:", mancanti)
    df = df.set_index("PANORAMA_study_id")
    df.index.name = "study_id"
    return df

clinical = carica_info_cliniche()
print(f"\nCasi totali nel file clinico: {len(clinical)}")
print("Distribuzione PDAC / non-PDAC:")
print(clinical["label"].value_counts().to_string())
clinical.head(3)
===MD===
## 5) Mappa delle segmentazioni (GitHub)

Per ogni esame troviamo **dove** sta la sua maschera e se la lesione e' stata annotata
**manualmente** (`manual_labels/`) o dall'**AI** (`automatic_labels/`).

> Nota: le strutture anatomiche (pancreas, dotto, vasi, coledoco) sono **sempre generate da AI**;
> solo la *lesione* puo' essere manuale.
===CODE===
def carica_mappa_maschere():
    """Legge l'albero del repo GitHub: per ogni study_id -> percorso maschera + manuale/AI."""
    cache = os.path.join(CACHE_DIR, "labels_tree.json")
    if not os.path.exists(cache):
        print("Scarico l'elenco delle maschere dal repo GitHub ...")
        r = requests.get(GH_TREE, timeout=60); r.raise_for_status()
        with open(cache, "w") as f:
            f.write(r.text)
    tree = json.load(open(cache))
    if "tree" not in tree:
        raise RuntimeError("Risposta GitHub inattesa (forse limite di richieste API). Riprova piu' tardi.")
    mappa = {}
    for node in tree["tree"]:
        p = node.get("path", "")
        if p.endswith(".nii.gz"):
            sid = os.path.basename(p).replace(".nii.gz", "")
            folder = p.split("/")[0]  # manual_labels | automatic_labels
            mappa[sid] = {"path": p,
                          "source": folder,
                          "lesion_annotation": "manuale" if folder == "manual_labels" else "AI"}
    return mappa

mask_map = carica_mappa_maschere()
print(f"Maschere disponibili: {len(mask_map)}")
print("Per provenienza:", dict(Counter(v["source"] for v in mask_map.values())))
===MD===
## 6) Indice delle immagini su Zenodo

Leggiamo l'**elenco dei file e le loro dimensioni** dentro i 4 zip **senza scaricarli**
(tramite `remotezip`). La prima volta richiede ~1 minuto; poi viene salvato in cache.
===CODE===
def costruisci_indice_zenodo(forza=False):
    """Indice (study_id, batch, percorso interno, dimensioni) dei 4 zip Zenodo, via HTTP-range."""
    cache = os.path.join(CACHE_DIR, "zenodo_index.csv")
    if os.path.exists(cache) and not forza:
        return pd.read_csv(cache)
    righe = []
    for batch, url in ZENODO_BATCHES.items():
        print(f"Leggo l'indice di {batch} (può richiedere qualche secondo) ...")
        with RemoteZip(url) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                sid = os.path.basename(info.filename).replace("_0000.nii.gz", "")
                righe.append({"study_id": sid, "batch": batch, "zip_url": url,
                              "zip_internal_path": info.filename,
                              "file_size": info.file_size})
    idx = pd.DataFrame(righe)
    idx.to_csv(cache, index=False)
    print("Indice salvato in cache.")
    return idx

zindex = costruisci_indice_zenodo()
print(f"\nFile immagine indicizzati: {len(zindex)}")
print(zindex.groupby("batch").agg(n=("study_id", "count"),
                                  GB=("file_size", lambda s: round(s.sum()/1e9, 1))))
===MD===
## 7) Selezione dei casi

Costruiamo l'elenco esatto degli esami da scaricare, in base alla configurazione del punto 1.
La selezione e' **riproducibile** (dipende solo da `SEED`).

Per i **PDAC**, di default (`PDAC_PREFER_MANUAL=True`) vengono scelti **prima i casi con la lesione
segmentata manualmente da esperto** (gold standard); per i **controlli** si usa la strategia scelta.
===CODE===
def costruisci_selezione():
    """Restituisce un DataFrame dei casi scelti, con info cliniche + indice immagine + maschera."""
    rng = random.Random(SEED)
    # esami che hanno SIA l'immagine SIA la riga clinica
    disponibili = [s for s in zindex["study_id"].tolist() if s in clinical.index]

    if MODE == "explicit":
        scelti = [s for s in STUDY_IDS if s in disponibili]
        ignorati = set(STUDY_IDS) - set(scelti)
        if ignorati:
            print("⚠️ study_id non trovati e ignorati:", ignorati)
        gruppi = {s: ("PDAC" if clinical.loc[s, "label"] == "PDAC" else "control") for s in scelti}
    else:
        lev = clinical["level"]
        n_dataset = len(disponibili)

        # --- ordinamento STABILE dei PDAC (manuali prima, se richiesto) ---
        # "Stabile" = con lo stesso SEED l'ordine non cambia tra una run e l'altra,
        # quindi una selezione piccola e' SEMPRE contenuta in una piu' grande
        # (download incrementale sicuro: aggiunge i nuovi, non riscarica i vecchi).
        pdac = [s for s in disponibili if clinical.loc[s, "label"] == "PDAC"]
        if PDAC_PREFER_MANUAL:
            man = [s for s in pdac if mask_map.get(s, {}).get("source") == "manual_labels"]
            ai  = [s for s in pdac if mask_map.get(s, {}).get("source") != "manual_labels"]
            rng.shuffle(man); rng.shuffle(ai); pdac_order = man + ai
        else:
            rng.shuffle(pdac); pdac_order = pdac

        # --- ordinamento STABILE dei controlli, secondo la strategia ---
        nonpdac = [s for s in disponibili if clinical.loc[s, "label"] != "PDAC"]
        nih = [s for s in nonpdac if lev.get(s) == "NIH_dataset"]
        fu  = [s for s in nonpdac if lev.get(s) == "radiology / 3yFU"]
        rad = [s for s in nonpdac if lev.get(s) == "radiology"]
        altri = [s for s in nonpdac if lev.get(s) not in ("NIH_dataset", "radiology / 3yFU", "radiology")]
        for _L in (nih, fu, rad, altri):
            rng.shuffle(_L)
        if CONTROL_STRATEGY == "nih_only":
            control_order = nih
        else:  # "sure_negatives" e "any_nonpdac": NIH -> radiology+3yFU -> radiology -> altri
            control_order = nih + fu + rad + altri   # copre TUTTI i non-PDAC (serve per MODE='all')

        # --- quanti casi per gruppo, secondo MODE ---
        if MODE == "all":
            n_p, n_c = len(pdac_order), len(control_order)
            print(f"MODE 'all': scarico TUTTI -> {n_p} PDAC + {n_c} controlli "
                  f"(⚠️ NON bilanciato: ci sono piu' controlli che PDAC).")
        elif MODE == "percent":
            n_tot = round(TARGET_PERCENT / 100.0 * n_dataset)
            n_p = n_c = n_tot // 2
            print(f"MODE 'percent': {TARGET_PERCENT}% di {n_dataset} ≈ {n_tot} casi -> {n_p} PDAC + {n_c} controlli.")
        else:  # "count"
            n_p, n_c = N_PDAC, N_CONTROL
            print(f"MODE 'count': {n_p} PDAC + {n_c} controlli.")

        # se non bastano i casi, riduci (restando bilanciato finche' possibile)
        if MODE != "all":
            if n_p > len(pdac_order):
                print(f"⚠️ Richiesti {n_p} PDAC ma disponibili solo {len(pdac_order)}. "
                      f"Per andare oltre serve MODE='all' (sbilanciato). "
                      f"Uso {len(pdac_order)} PDAC e altrettanti controlli.")
                n_p = len(pdac_order); n_c = min(n_c, n_p)
            if n_c > len(control_order):
                print(f"⚠️ Richiesti {n_c} controlli ma disponibili solo {len(control_order)}. Uso {len(control_order)}.")
                n_c = len(control_order)

        pdac_sel = pdac_order[:n_p]
        controls = control_order[:n_c]
        n_man = sum(1 for s in pdac_sel if mask_map.get(s, {}).get("source") == "manual_labels")
        print(f"PDAC con lesione MANUALE: {n_man}/{len(pdac_sel)} (il resto: lesione AI)")
        scelti = pdac_sel + controls
        gruppi = {**{s: "PDAC" for s in pdac_sel}, **{s: "control" for s in controls}}

    sub = clinical.loc[scelti].copy()
    sub["group"] = [gruppi[s] for s in scelti]
    sub = sub.join(zindex.set_index("study_id")[["batch", "zip_url", "zip_internal_path", "file_size"]])
    sub["has_mask"]          = [s in mask_map for s in scelti]
    sub["mask_path"]         = [mask_map.get(s, {}).get("path", "") for s in scelti]
    sub["mask_source"]       = [mask_map.get(s, {}).get("source", "") for s in scelti]
    sub["lesion_annotation"] = [mask_map.get(s, {}).get("lesion_annotation", "") for s in scelti]
    return sub.reset_index()

selection = costruisci_selezione()
print(f"\nCasi selezionati: {len(selection)}")
print(selection["group"].value_counts().to_string())
if MODE == "balanced":
    print("\nControlli per reference standard:")
    print(selection[selection["group"] == "control"]["level"].value_counts().to_string())
stima_gb = selection["file_size"].sum() / 1e9
print(f"\nDimensione stimata immagini: {stima_gb:.1f} GB (le maschere sono trascurabili)")
selection.head()
===MD===
## 8) Controllo dello spazio su disco

Prima di scaricare, verifichiamo che ci sia spazio sufficiente (con un margine di sicurezza).
===CODE===
def controllo_spazio(stima_gb):
    total, used, free = shutil.disk_usage(PROJECT_DIR)
    free_gb = free / 1e9
    print(f"Spazio libero: {free_gb:.1f} GB | servono ~{stima_gb:.1f} GB (+ margine {MIN_FREE_GB} GB)")
    if free_gb < stima_gb + MIN_FREE_GB:
        print("⚠️ ATTENZIONE: spazio potenzialmente INSUFFICIENTE. "
              "Riduci N_PDAC / N_CONTROL e ri-esegui dal punto 7.")
        return False
    print("✅ Spazio sufficiente.")
    return True

spazio_ok = controllo_spazio(stima_gb)
===MD===
## 9) Download delle immagini

Scarichiamo i volumi TC estraendo **solo i file scelti** dagli zip Zenodo
(sequenziale, una connessione per batch). Con verifica della dimensione e ritentativi automatici.

> A ~1 MB/s, 24 GB richiedono diverse ore: puoi lasciarlo girare. Se interrompi, ri-esegui e riprende.
===CODE===
def _retry(fn, descr, tries=4, pausa=3):
    """Esegue fn() ritentando in caso di errore di rete."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"   ! tentativo {k+1}/{tries} fallito ({descr}): {type(e).__name__}: {e}")
            time.sleep(pausa * (k + 1))
    raise RuntimeError(f"Impossibile completare: {descr}")

def scarica_immagini(sel):
    """Scarica i volumi TC (resumabile: salta i file gia' presenti e completi).
    Robusto: timeout di lettura (se la connessione si pianta scatta l'errore),
    riapertura automatica della connessione e ritentativi. Se un file continua a
    fallire viene SALTATO (lo recuperi semplicemente ri-eseguendo)."""
    fatti = saltati = 0
    falliti = []
    tot = len(sel)
    # (connessione 10s, lettura 60s): se per 60s non arrivano byte, requests solleva
    # un errore -> scatta il ritentativo, invece di restare appeso all'infinito.
    TIMEOUT = (10, 60)
    for batch, gruppo in sel.groupby("batch"):
        url = gruppo["zip_url"].iloc[0]
        da_fare = []
        for _, r in gruppo.iterrows():
            dst = os.path.join(IMAGES_DIR, f"{r['study_id']}_0000.nii.gz")
            if os.path.exists(dst) and os.path.getsize(dst) == int(r["file_size"]):
                saltati += 1
            else:
                da_fare.append(r)
        print(f"[{batch}] {len(gruppo)} casi: {len(gruppo)-len(da_fare)} gia' presenti, {len(da_fare)} da scaricare")
        if not da_fare:
            continue
        zbox = {"z": _retry(lambda: RemoteZip(url, timeout=TIMEOUT), f"apertura {batch}")}
        try:
            for r in da_fare:
                sid = r["study_id"]
                dst = os.path.join(IMAGES_DIR, f"{sid}_0000.nii.gz")
                exp = int(r["file_size"])
                name = r["zip_internal_path"]
                # protezione anti-disco-pieno: ferma in modo pulito se manca spazio
                free = shutil.disk_usage(IMAGES_DIR).free
                if free < exp + MIN_FREE_GB * 1e9:
                    print(f"⛔ STOP spazio: liberi {free/1e9:.1f} GB (servono ~{exp/1e6:.0f} MB "
                          f"+ margine {MIN_FREE_GB} GB). Scaricate {fatti} immagini finora. "
                          f"Libera spazio o riduci TARGET_PERCENT, poi ri-esegui (riprende da qui).")
                    return
                def grab():
                    tmp = dst + ".part"
                    try:
                        with zbox["z"].open(name) as src, open(tmp, "wb") as out:
                            shutil.copyfileobj(src, out, length=1024 * 1024)
                    except Exception:
                        # connessione probabilmente morta: riapro lo zip per il tentativo dopo
                        try: zbox["z"].close()
                        except Exception: pass
                        zbox["z"] = RemoteZip(url, timeout=TIMEOUT)
                        raise
                    got = os.path.getsize(tmp)
                    if got != exp:
                        raise IOError(f"dimensione errata {got} != {exp}")
                    os.replace(tmp, dst)
                try:
                    _retry(grab, f"immagine {sid}", tries=5)
                    fatti += 1
                    print(f"   ✓ [{fatti+saltati}/{tot}] {sid}  ({exp/1e6:.0f} MB)")
                except Exception as e:
                    falliti.append(sid)
                    try: os.remove(dst + ".part")
                    except OSError: pass
                    print(f"   ⚠️ salto {sid} dopo vari tentativi ({type(e).__name__}). Lo recuperi ri-eseguendo.")
        finally:
            try: zbox["z"].close()
            except Exception: pass
    msg = f"\nImmagini: scaricate ora {fatti}, gia' presenti {saltati}, totale {tot}."
    if falliti:
        msg += f"\n⚠️ {len(falliti)} non riuscite (ri-esegui per recuperarle): {falliti}"
    print(msg)

scarica_immagini(selection)
===MD===
## 10) Download delle maschere (segmentazioni)

Le maschere sono piccole e arrivano da GitHub. Anche questo passo e' resumabile.
===CODE===
def scarica_maschere(sel):
    fatti = saltati = senza = 0
    for _, r in sel.iterrows():
        sid = r["study_id"]
        if not r["has_mask"]:
            senza += 1
            continue
        dst = os.path.join(LABELS_DIR, f"{sid}.nii.gz")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            saltati += 1
            continue
        url = GH_RAW + r["mask_path"]
        def grab():
            tmp = dst + ".part"
            with requests.get(url, timeout=120, stream=True) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as out:
                    for chunk in resp.iter_content(256 * 1024):
                        out.write(chunk)
            if os.path.getsize(tmp) == 0:
                raise IOError("maschera vuota")
            os.replace(tmp, dst)
        _retry(grab, f"maschera {sid}")
        fatti += 1
        if fatti % 25 == 0:
            print(f"   maschere scaricate: {fatti}")
    print(f"Maschere: scaricate ora {fatti}, gia' presenti {saltati}, senza maschera {senza}.")

scarica_maschere(selection)
===MD===
## 11) Validazione dei file scaricati

Controlliamo che ogni file sia un `.nii.gz` valido (intestazione NIfTI corretta).
===CODE===
def valida_nifti(path):
    """Verifica gzip + intestazione NIfTI-1. Restituisce (ok, dimensioni)."""
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\x1f\x8b":
                return False, None
        with gzip.open(path, "rb") as g:
            hdr = g.read(348)
        if struct.unpack("<i", hdr[:4])[0] != 348:
            return False, None
        dims = struct.unpack("<8h", hdr[40:56])
        return True, tuple(dims[1:1 + dims[0]])
    except Exception:
        return False, None

n_img = n_bad = 0
for _, r in selection.iterrows():
    img = os.path.join(IMAGES_DIR, f"{r['study_id']}_0000.nii.gz")
    if os.path.exists(img):
        n_img += 1
        ok, _ = valida_nifti(img)
        if not ok:
            n_bad += 1
            print("⚠️ immagine NON valida:", img)
print(f"Validazione: {n_img} immagini presenti, {n_bad} non valide.")
if n_bad:
    print("Suggerimento: cancella i file non validi e ri-esegui il punto 9 per ri-scaricarli.")
===MD===
## 12) Analisi delle segmentazioni

Per ogni maschera leggiamo **quali distretti anatomici sono effettivamente segmentati**
(quali classi 1–6 sono presenti) e se la lesione e' manuale o AI.
===CODE===
def analizza_maschere(sel):
    """Per ogni maschera: distretti presenti (classi 1-6) + provenienza."""
    righe = []
    for _, r in sel.iterrows():
        sid = r["study_id"]
        dst = os.path.join(LABELS_DIR, f"{sid}.nii.gz")
        rec = {"study_id": sid, "mask_presente": os.path.exists(dst)}
        presenti = set()
        if rec["mask_presente"]:
            try:
                arr = np.asanyarray(nib.load(dst).dataobj)
                presenti = {int(v) for v in np.unique(arr)}
            except Exception as e:
                print(f"⚠️ errore lettura maschera {sid}: {e}")
        for cls, short in DISTRICT_SHORT.items():
            rec[f"seg_{short}"] = cls in presenti
        rec["distretti_segmentati"] = ", ".join(
            DISTRICT_SHORT[c] for c in sorted(DISTRICT_SHORT) if c in presenti)
        righe.append(rec)
    return pd.DataFrame(righe)

analisi = analizza_maschere(selection)
print("Esempio analisi segmentazioni:")
analisi.head()
===MD===
## 13) Report Excel

Creiamo il file Excel finale con 4 fogli: **scaricati** (dettaglio per esame),
**riepilogo**, **legenda** delle classi e **note**.
===CODE===
def crea_report_excel(sel, analisi):
    df = sel.merge(analisi, on="study_id", how="left")
    colonne = ["study_id", "group", "label", "level", "patient_age", "patient_sex", "scanner",
               "anonymized_study_date", "batch", "file_size", "mask_source", "lesion_annotation",
               "distretti_segmentati", "seg_lesione", "seg_pancreas", "seg_dotto", "seg_vene",
               "seg_arterie", "seg_coledoco", "mask_presente"]
    colonne = [c for c in colonne if c in df.columns]
    out = df[colonne].rename(columns={"file_size": "image_bytes", "level": "reference_standard",
                                      "anonymized_study_date": "study_date"})
    out["image_MB"] = (out["image_bytes"] / 1e6).round(1)
    out["immagine_scaricata"] = [os.path.exists(os.path.join(IMAGES_DIR, f"{s}_0000.nii.gz"))
                                 for s in out["study_id"]]

    riepilogo = out.groupby("group").agg(
        n=("study_id", "count"),
        GB=("image_bytes", lambda s: round(s.sum() / 1e9, 2)),
        con_lesione=("seg_lesione", "sum"),
    ).reset_index()

    legenda = pd.DataFrame({"classe": list(LABEL_CLASSES.keys()),
                            "distretto": list(LABEL_CLASSES.values())})
    note = pd.DataFrame({"NOTE": [
        "Strutture anatomiche (pancreas, dotto, vene, arterie, coledoco): SEMPRE generate da AI (Alves et al. 2021).",
        "Solo la LESIONE (classe 1) e' annotata manualmente nei casi della cartella 'manual_labels'.",
        "lesion_annotation='manuale' -> lesione rivista da esperto; 'AI' -> generata automaticamente (o assente nei controlli).",
        "Licenza dati: CC BY-NC 4.0 (solo uso non commerciale). Citare PANORAMA (Alves et al. 2024).",
    ]})

    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="scaricati", index=False)
        riepilogo.to_excel(w, sheet_name="riepilogo", index=False)
        legenda.to_excel(w, sheet_name="legenda", index=False)
        note.to_excel(w, sheet_name="note", index=False)
    print(f"✅ Report salvato: {REPORT_XLSX}  ({len(out)} righe)")
    return out

report = crea_report_excel(selection, analisi)
print("\nRiepilogo:")
report.head()
===MD===
## ✅ Fatto!

Hai in `imagesTr/` i volumi TC e in `labelsTr/` le maschere, piu' `panorama_report.xlsx` con tutti i dettagli.

**Per scaricare più esami in futuro (incrementale):**
- alza `TARGET_PERCENT` al punto 1 (es. `1` → `5` → `25`) e ri-esegui: scarica **solo i nuovi**;
- per avere **tutto** il dataset: `MODE = "all"` (676 PDAC + 1562 controlli, sbilanciato);
- per casi specifici: `MODE = "explicit"` con una lista di `study_id`.

> ⚠️ Nelle run incrementali non cambiare `SEED` / `CONTROL_STRATEGY` / `PDAC_PREFER_MANUAL`.
> Ricorda il vincolo di spazio: il punto 8 ti avvisa se rischi di riempire il disco.
'''

# --- assemblaggio notebook --------------------------------------------------
nb = nbf.v4.new_notebook()
cells = []
for blocco in SRC.split("===MD===")[1:]:
    # ogni blocco puo' contenere una parte markdown e poi una o piu' code dopo ===CODE===
    parti = blocco.split("===CODE===")
    md_text = parti[0].strip("\n")
    if md_text.strip():
        cells.append(nbf.v4.new_markdown_cell(md_text))
    for code_text in parti[1:]:
        code_text = code_text.strip("\n")
        if code_text.strip():
            cells.append(nbf.v4.new_code_cell(code_text))

nb.cells = cells
nb.metadata["kernelspec"] = {"name": "panorama-pdac",
                             "display_name": "PANORAMA PDAC (.venv 3.12)",
                             "language": "python"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12.13"}

out_path = "panorama_download.ipynb"
nbf.write(nb, out_path)
print(f"Notebook creato: {out_path} con {len(cells)} celle "
      f"({sum(c.cell_type=='markdown' for c in cells)} markdown, "
      f"{sum(c.cell_type=='code' for c in cells)} codice).")
