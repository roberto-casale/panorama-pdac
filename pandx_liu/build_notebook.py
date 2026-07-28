#!/usr/bin/env python3
"""Genera `pandx_inference.ipynb`.

Il notebook esegue PanDx (H. Liu, team DTI — Siemens Healthineers), l'algoritmo
che ha VINTO la PANORAMA challenge, su nuovi esami TC e produce per ogni caso un
punteggio 0-1 di sospetto PDAC.

Riproduce fedelmente la pipeline di `main.py` del repo han-liu/PDAC_Detection,
usando l'API Python di nnU-Net invece della riga di comando (stessi risultati,
nessun sottoprocesso e nessun rischio di deadlock del multiprocessing su macOS).
"""
import json

nb = {"cells": [], "metadata": {
    "kernelspec": {"display_name": "PANORAMA baseline (.venv-nnunet)",
                   "language": "python", "name": "panorama-nnunet"},
    "language_info": {"name": "python", "version": "3.12.13"}},
    "nbformat": 4, "nbformat_minor": 5}


_counter = [0]


def _cell_id():
    _counter[0] += 1
    return f"cell{_counter[0]:03d}"


def md(text):
    # `source` come stringa unica: evita il bug delle righe concatenate senza "\n"
    nb["cells"].append({"cell_type": "markdown", "id": _cell_id(), "metadata": {},
                        "source": text.strip("\n")})


def code(text):
    nb["cells"].append({"cell_type": "code", "id": _cell_id(), "metadata": {},
                        "execution_count": None, "outputs": [], "source": text.strip("\n")})


# ---------------------------------------------------------------- intro
md(r"""
# PanDx (H. Liu) — inferenza PDAC su nuovi esami TC

**PanDx** è l'algoritmo che ha **vinto** la PANORAMA challenge (team DTI, Siemens Healthineers).
Questo notebook prende esami TC addominali **con contrasto, fase porto-venosa** (`.nii.gz`)
e per ognuno restituisce un **punteggio 0–1**: quanto è probabile che il paziente abbia un
adenocarcinoma duttale del pancreas (PDAC).

Con quei punteggi calcola poi **AUROC** (diagnosi paziente) e, se hai le segmentazioni
manuali del tumore, anche **AP** (localizzazione della lesione).

## Come funziona (due reti in fila)

| | Cosa fa | Rete |
|---|---|---|
| **Stadio 1** | trova il pancreas su una versione a bassa risoluzione | nnU-Net del **baseline**, riusato senza modifiche |
| *in mezzo* | ritaglia una scatola attorno al pancreas (margine 100×50×15 mm) | pura geometria |
| **Stadio 2** | dentro la scatola cerca il tumore | rete **di Liu** (`Dataset107`), addestrata da lui |
| *alla fine* | estrae le lesioni candidate con soglia adattiva; il **massimo** è il punteggio del paziente | post-processing |

## In cosa differisce dal baseline

Liu parte dallo stesso schema a due stadi, ma cambia tre cose. Sono poche e mirate,
ed è da lì che viene il suo vantaggio (AP 0,720 contro 0,634 sulla coorte di test
sequestrata di 1130 casi, figura S8 dell'appendice di Alves et al.):

| | Baseline | **PanDx (Liu)** |
|---|---|---|
| Stadio 1 | proprio | **lo stesso**, riusato tale e quale |
| Stadio 2 | U-Net standard, `Dataset104` | **ResU-Net**, `Dataset107`, addestrata da lui |
| Estrazione lesioni | stessa soglia adattiva col fattore di default: τ = picco / 2,5 | **fattore 15**: τ = picco / 15 |
| Filtro anti-falsi-positivi | azzera fuori dal pancreas dilatato | **nessun filtro** |

> **Importante:** i modelli ricevono **solo l'immagine TC**. Le segmentazioni manuali,
> se le hai, servono unicamente a *valutare* i risultati — non entrano mai nel modello.
> Nemmeno età e sesso vengono usati.

## Fonti (codice riprodotto fedelmente)

- Codice: <https://github.com/han-liu/PDAC_Detection>
- Pesi: cartella Google Drive indicata nel loro README (vedi `SETUP_GUIDE.md`)
- Articolo: Liu H, Gao R, Krieg E, Grbic S. *PanDx*, MICCAI Workshop on Applications of
  Medical AI, 2025 ([arXiv 2503.10068](https://arxiv.org/abs/2503.10068))
- Challenge: Alves N, et al. *Lancet Oncol* 2026;27(1):116–124

⚠️ **Strumento di ricerca**, non un dispositivo medico approvato. Non usare per decisioni cliniche.
I pesi del baseline (usati nello stadio 1) sono **CC BY-NC 4.0**: solo uso non commerciale.
""")

# ---------------------------------------------------------------- 1. config
md(r"""
---
## 1 · Configurazione

**È l'unica cella che devi modificare.** Tutto il resto gira da solo.

### Come organizzare i dati

```
CARTELLA_DATI/
├── PDAC/
│   ├── images/          ← TC dei pazienti CON tumore   (.nii.gz)
│   └── labels/          ← (opzionale) segmentazioni manuali del tumore
└── health/
    └── images/          ← TC dei controlli SENZA tumore (.nii.gz)
```

I nomi dei file devono corrispondere tra `images/` e `labels/` (es. `001.nii.gz` in entrambe).
""")

code(r'''
from pathlib import Path

# ============================================================================
#  DOVE SIAMO  —  rilevato da solo, non serve toccarlo
# ============================================================================
# BASE_DIR = la cartella che contiene questo notebook (`pandx_liu`).
# Usiamo percorsi RELATIVI, così l'intera cartella si può copiare su un altro
# computer (es. il Linux con GPU) e funziona senza modifiche.
#
# Il riconoscimento si basa su un file che c'è SEMPRE (questo notebook), non sulla
# cartella `models`: quest'ultima non esiste prima di aver scaricato i pesi, e usarla
# come riferimento renderebbe impossibile il primo avvio.
_MARKER = "pandx_inference.ipynb"
BASE_DIR = Path.cwd()
if not (BASE_DIR / _MARKER).exists():
    if (BASE_DIR / "pandx_liu" / _MARKER).exists():
        BASE_DIR = BASE_DIR / "pandx_liu"          # lanciato dalla cartella superiore
    else:
        print(f"ATTENZIONE: non trovo '{_MARKER}' in {BASE_DIR}.\n"
              f"            Se i percorsi non funzionano, imposta BASE_DIR a mano qui sotto.")

MODELS_DIR = BASE_DIR / "models"                 # pesi scaricati dal Drive di Liu
OUTPUT_DIR = BASE_DIR / "risultati"              # punteggi e mappe di detection

# ============================================================================
#  I TUOI DATI  —  modifica questi
# ============================================================================
# Metti None se un gruppo non ce l'hai. Accetta percorsi assoluti o relativi.
PDAC_IMAGES   = None    # es. Path("/percorso/PDAC/images")
PDAC_LABELS   = None    # es. Path("/percorso/PDAC/labels")   -> serve solo per l'AP
HEALTH_IMAGES = None    # es. Path("/percorso/health/images")

# ============================================================================
#  OPZIONI DI ESECUZIONE
# ============================================================================
# "auto" sceglie la GPU se c'è, altrimenti la CPU
DEVICE = "auto"

# MODALITA' VELOCE: usa 1 fold e disattiva la TTA -> ~40x piu' rapido, ma i
# risultati NON sono identici a quelli ufficiali. Utile solo per provare che
# la pipeline funzioni. Per i risultati veri lascia False.
FAST_MODE = False

# Fedeltà massima (impostazioni ufficiali di PanDx): 5 fold + TTA attiva
FOLDS    = (0,) if FAST_MODE else (0, 1, 2, 3, 4)
USE_TTA  = not FAST_MODE

# Soglia adattiva per estrarre le lesioni: tau = picco / INV_ALPHA.
# 15 è il valore scelto da Liu dopo ricerca a griglia (nel suo codice: --inv_alpha).
# È una delle differenze chiave rispetto al baseline: NON modificarlo se vuoi
# risultati confrontabili con quelli pubblicati.
INV_ALPHA = 15

# Salva le mappe di detection 3D (servono per calcolare l'AP)
SAVE_DETECTION_MAPS = True

# Se un caso è già stato processato, saltalo (permette di riprendere)
RESUME = True

print("Configurazione caricata.")
print(f"  modelli : {MODELS_DIR}")
print(f"  output  : {OUTPUT_DIR}")
print(f"  fold    : {FOLDS}   TTA: {USE_TTA}" + ("   [MODALITA' VELOCE]" if FAST_MODE else "   [fedeltà massima]"))
''')

# ---------------------------------------------------------------- 2. env
md(r"""
---
## 2 · Controllo dell'ambiente

Verifica che tutto sia installato e coerente. **Se qui compare un errore, fermati e
risolvilo** prima di andare avanti: gli errori a valle sarebbero molto più difficili da capire.
""")

code(r'''
import sys, os, platform, shutil, warnings, time
warnings.filterwarnings("ignore", category=DeprecationWarning)

problemi = []

print("=" * 62)
print("AMBIENTE")
print("=" * 62)
print(f"  Python     : {sys.version.split()[0]}")
print(f"  Eseguibile : {sys.executable}")
print(f"  Sistema    : {platform.system()} {platform.machine()}")

# --- il kernel giusto? (l'ambiente puo' chiamarsi diversamente: controlliamo i pacchetti) ---
try:
    import nnunetv2 as _nn  # noqa: F401
except ImportError:
    problemi.append("nnunetv2 non è installato in QUESTO kernel: probabilmente hai selezionato "
                    "il kernel sbagliato (serve quello dell'ambiente con nnU-Net).")

import numpy as np
import torch
print(f"  NumPy      : {np.__version__}")
print(f"  PyTorch    : {torch.__version__}")

# --- numpy e torch devono essere compatibili ---
if int(np.__version__.split(".")[0]) >= 2 and torch.__version__ < "2.3":
    problemi.append(f"NumPy {np.__version__} non è compatibile con PyTorch {torch.__version__}: serve numpy<2.")

import nibabel, SimpleITK as sitk
print(f"  nibabel    : {nibabel.__version__}")
print(f"  SimpleITK  : {sitk.__version__}")

try:
    import nnunetv2
    print(f"  nnU-Net    : {nnunetv2.__version__ if hasattr(nnunetv2,'__version__') else 'installato'}")
except ImportError:
    problemi.append("nnunetv2 non installato.")

try:
    from report_guided_annotation import extract_lesion_candidates
    print("  report-guided-annotation : ok")
except ImportError:
    problemi.append("report-guided-annotation non installato.")

# --- torch >= 2.6 non riesce a caricare questi checkpoint ---
# dalla 2.6 `torch.load` usa weights_only=True di default e i checkpoint del
# baseline (che contengono oggetti numpy) vengono rifiutati.
_tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _tv >= (2, 6):
    problemi.append(f"PyTorch {torch.__version__} è troppo recente per nnunetv2 2.5.1: "
                    f"dalla 2.6 il caricamento dei checkpoint fallisce "
                    f"(weights_only=True). Usa torch < 2.6, es. 2.5.1.")

# --- device ---
if DEVICE == "auto":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device(DEVICE)
print(f"\n  Device     : {device}")
if device.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU        : {p.name}  ({p.total_memory/1e9:.1f} GB)")
else:
    print("  ATTENZIONE : nessuna GPU -> inferenza su CPU, da 1 a 4 ORE per caso.")
    print("               Va bene per provare su pochi casi; per lotti grandi serve una GPU.")

# --- il trainer personalizzato del baseline deve essere raggiungibile ---
try:
    import nnunetv2, os as _os
    _tdir = _os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
    if not _os.path.exists(_os.path.join(_tdir, "variants", "loss", "liuPanDxTrainers.py")):
        problemi.append("Manca 'liuPanDxTrainers.py' dentro nnunetv2 (vedi SETUP_GUIDE.md).")
    else:
        print("  Trainer personalizzato di PanDx : ok")
except Exception as e:
    problemi.append(f"Non riesco a controllare il trainer: {e}")

# --- spazio disco ---
if OUTPUT_DIR.parent.exists():
    free_gb = shutil.disk_usage(OUTPUT_DIR.parent).free / 1e9
    print(f"\n  Spazio libero : {free_gb:.1f} GB")
    if SAVE_DETECTION_MAPS and free_gb < 5:
        problemi.append(f"Solo {free_gb:.1f} GB liberi: le mappe di detection occupano spazio.")

print("\n" + "=" * 62)
if problemi:
    print("PROBLEMI DA RISOLVERE:")
    for p in problemi:
        print("   [X]", p)
    raise SystemExit("Sistema i problemi elencati prima di proseguire.")
print("AMBIENTE OK")
print("=" * 62)
''')

# ---------------------------------------------------------------- 3. modelli
md(r"""
---
## 3 · I modelli ci sono?

Controlla che i pesi scaricati dal Google Drive di Liu siano al loro posto e completi.
Se mancano, la cella stampa il comando esatto per scaricarli.
""")

code(r'''
# I pesi di Liu arrivano dalla sua cartella Google Drive e conservano la struttura
# `workspace/nnUNet_results/...`. Se manca, cerchiamo anche senza quel prefisso.
_ROOTS = [MODELS_DIR / "workspace" / "nnUNet_results", MODELS_DIR]
_R = next((r for r in _ROOTS if (r / "Dataset107_PDAC_Detection").exists()), _ROOTS[0])

M_STAGE1 = _R / "Dataset103_PANORAMA_baseline_Pancreas_Segmentation" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
M_STAGE2 = _R / "Dataset107_PDAC_Detection" / "nnUNetTrainerCELossLesionSplit__nnUNetPlans_v3__3d_fullres"
CK_STAGE1 = "checkpoint_final.pth"
CK_STAGE2 = "checkpoint_final.pth"

ISTRUZIONI = f"""
I pesi non ci sono. Scaricali (~5,1 GB) dalla cartella Google Drive di Liu:

  pip install gdown
  python -c "import gdown; gdown.download_folder(
      'https://drive.google.com/drive/folders/1RpbofQDrQNzwfYjFhQYRRWCN8HhIoZQP',
      output='{MODELS_DIR}')"

Se Google Drive rifiuta il download automatico (capita per limiti di traffico),
scarica la cartella a mano dal browser e mettila in:
  {MODELS_DIR}
mantenendo la struttura `workspace/nnUNet_results/Dataset...`.
"""

mancanti = []
for nome, folder, ck in [("Stadio 1", M_STAGE1, CK_STAGE1), ("Stadio 2", M_STAGE2, CK_STAGE2)]:
    if not folder.exists():
        mancanti.append(f"{nome}: cartella assente ({folder})")
        continue
    for f in FOLDS:
        p = folder / f"fold_{f}" / ck
        if not p.exists():
            mancanti.append(f"{nome}: manca {p.name} in fold_{f}")
    for j in ["plans.json", "dataset.json"]:
        if not (folder / j).exists():
            mancanti.append(f"{nome}: manca {j}")

if mancanti:
    for m in mancanti:
        print("  [X]", m)
    print(ISTRUZIONI)
    raise SystemExit("Modelli mancanti.")

tot = sum(f.stat().st_size for f in MODELS_DIR.rglob("*.pth")) / 1e6
print(f"  Modelli presenti e completi  ({tot:.0f} MB di pesi, fold richieste: {list(FOLDS)})")
''')

# ---------------------------------------------------------------- 4. pipeline
md(r"""
---
## 4 · La pipeline

Qui sotto c'è la traduzione fedele di `main.py` del repo di Liu (han-liu/PDAC_Detection).
Ogni funzione ha un commento che spiega **cosa fa e perché**.

Le differenze rispetto all'originale, nessuna delle quali cambia i risultati:

1. usiamo l'**API Python** di nnU-Net invece di lanciare `nnUNetv2_predict` come
   sottoprocesso (evita anche il blocco del multiprocessing su macOS);
2. gli intermedi finiscono in una cartella **temporanea** che viene cancellata da sola,
   invece che in `output_dir/itm`;
3. usiamo `nnunetv2 2.5.1` da PyPI invece della copia inclusa nel repo di Liu. Quella
   copia calcola in modo diverso la **gaussiana** con cui si fondono le patch, e la
   differenza NON è trascurabile: per questo il notebook la sostituisce con la versione
   di Liu (vedi il commento nella cella che carica i modelli).
""")

code(r'''
import numpy as np
import SimpleITK as sitk
from report_guided_annotation import extract_lesion_candidates
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# Valori ufficiali del baseline: NON modificarli se vuoi risultati confrontabili
SPACING_STAGE1 = (4.5, 4.5, 9.0)   # risoluzione grossolana per lo stadio 1
CROP_MARGINS   = [100, 50, 15]     # margine in MILLIMETRI su x, y, z (per lato)


def resample_img(itk_image, out_spacing, is_label=False):
    """Ricampiona a una data risoluzione (identica a `resample_img` di main.py)."""
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_size = [int(np.round(original_size[i] * (original_spacing[i] / out_spacing[i]))) for i in range(3)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(out_spacing)
    r.SetSize(out_size)
    r.SetOutputDirection(itk_image.GetDirection())
    r.SetOutputOrigin(itk_image.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return r.Execute(itk_image)


def crop_pancreas_roi(image, low_res_segmentation, margins=CROP_MARGINS):
    """Ritaglia la scatola attorno al pancreas predetto.

    Prende il bounding box della maschera, lo allarga del margine (in mm, su
    ENTRAMBI i lati di ogni asse) e taglia l'immagine a piena risoluzione.
    Identica a `crop_roi` di main.py.
    """
    mask_np = sitk.GetArrayFromImage(low_res_segmentation)
    # Liu tiene SOLO la classe 1 (pancreas) prima di calcolare il bounding box:
    #     pancreas_mask_np[pancreas_mask_np != 1] = 0
    # Con i pesi attuali dello stadio 1 (2 classi) non cambia nulla, ma se un giorno
    # ci si mettesse un modello multi-classe la scatola si allargherebbe a vasi e dotti
    # senza dare errore. Lo replichiamo per fedelta'.
    mask_np = (mask_np == 1)
    if mask_np.max() == 0:
        raise RuntimeError("Lo stadio 1 non ha trovato il pancreas: impossibile ritagliare.")

    nz = np.nonzero(mask_np)                      # array numpy in ordine (z, y, x)
    start_idx = (int(nz[2].min()), int(nz[1].min()), int(nz[0].min()))   # -> (x, y, z)
    finish_idx = (int(nz[2].max()), int(nz[1].max()), int(nz[0].max()))

    # dagli indici della maschera a bassa risoluzione alle coordinate fisiche,
    # e da queste agli indici dell'immagine ad alta risoluzione
    start_phys = low_res_segmentation.TransformIndexToPhysicalPoint(start_idx)
    finish_phys = low_res_segmentation.TransformIndexToPhysicalPoint(finish_idx)
    start_point = image.TransformPhysicalPointToIndex(start_phys)
    finish_point = image.TransformPhysicalPointToIndex(finish_phys)

    spacing, size = image.GetSpacing(), image.GetSize()
    m = [int(margins[i] / spacing[i]) for i in range(3)]   # margine mm -> voxel

    x0, x1 = max(0, start_point[0] - m[0]), min(size[0], finish_point[0] + m[0])
    y0, y1 = max(0, start_point[1] - m[1]), min(size[1], finish_point[1] + m[1])
    z0, z1 = max(0, start_point[2] - m[2]), min(size[2], finish_point[2] + m[2])

    cropped = image[x0:x1, y0:y1, z0:z1]
    coords = dict(x_start=x0, x_finish=x1, y_start=y0, y_finish=y1, z_start=z0, z_finish=z1)
    return cropped, coords


def post_processing(seg_stage2, probs_stage2):
    """Prende la probabilità di tumore, senza alcun filtro.

    ATTENZIONE — qui PanDx differisce dal baseline:
    il baseline azzera la probabilità fuori dal pancreas dilatato (filtro
    anti-falsi-positivi), Liu NON lo fa. Tiene la mappa di probabilità così
    com'è e lascia che sia la soglia adattiva a selezionare le lesioni.
    Identica a `PostProcessing` di main.py (repo han-liu/PDAC_Detection).
    """
    return probs_stage2[1].astype(np.float32)   # canale 1 = tumore


def to_full_size(tumor_prob_cropped, coords, full_image, inv_alpha=None):
    """Estrae le lesioni candidate e rimette la mappa nel volume originale.

    ATTENZIONE — seconda differenza dal baseline: la SOGLIA ADATTIVA.
    Il baseline usa la soglia di default della libreria; Liu passa
    `dynamic_threshold_factor = 15`, cioè tau = (probabilità di picco) / 15.
    Soglia bassa e proporzionale al picco: le lesioni si espandono di più.
    È il parametro che lui ha ottimizzato con ricerca a griglia.

    Restituisce (mappa di detection a dimensione piena, punteggio, n. candidati).
    Identica a `GetFullSizDetectionMap` di main.py.
    """
    # letto qui e non come default della funzione: così se cambi INV_ALPHA al punto 1
    # e rilanci, il nuovo valore viene davvero usato (un default sarebbe congelato
    # al momento in cui la cella è stata eseguita la prima volta)
    if inv_alpha is None:
        inv_alpha = INV_ALPHA
    lesion_candidates, confidences, _ = extract_lesion_candidates(
        tumor_prob_cropped, dynamic_threshold_factor=inv_alpha)
    patient_score = float(np.max(lesion_candidates))

    # GetSize() e' (x, y, z), gli array numpy sono (z, y, x) -> invertiamo.
    # (usare GetArrayFromImage solo per leggere la forma sprecherebbe centinaia di MB)
    full_shape = full_image.GetSize()[::-1]
    full_map = np.zeros(full_shape, dtype=np.float32)
    full_map[coords["z_start"]:coords["z_finish"],
             coords["y_start"]:coords["y_finish"],
             coords["x_start"]:coords["x_finish"]] = lesion_candidates
    return full_map, patient_score, len(confidences)


print("Funzioni della pipeline definite.")
''')

md(r"""
### Caricamento dei due modelli

I modelli si caricano **una volta sola** e restano in memoria per tutti i casi
(caricarli a ogni esame sarebbe uno spreco di minuti).
""")

code(r'''
# ============================================================================
#  FEDELTA' A PanDx: la gaussiana con cui si fondono le patch
# ============================================================================
# nnU-Net predice a finestre sovrapposte e le fonde pesandole con una gaussiana.
# Liu usa nnU-Net 2.5, dove quella gaussiana viene convertita a float16 PRIMA di
# essere normalizzata; dalla 2.5.1 (quella che installiamo) la normalizzazione
# avviene in float64 e la conversione arriva dopo.
#
# Non è un dettaglio da nulla: in float16 il picco della gaussiana cade nei numeri
# subnormali, quindi nella versione di Liu circa il 70% di ogni patch finisce
# schiacciato sullo stesso peso minimo. I pesi di fusione risultano diversi e
# di conseguenza lo sono anche i punteggi finali.
#
# Per riprodurre PanDx replichiamo QUI la sua versione. La sostituzione vale solo
# dentro questo notebook: non modifica i file dell'ambiente, quindi il notebook
# del baseline continua a usare la versione corretta per lui.
from functools import lru_cache
from scipy.ndimage import gaussian_filter
import nnunetv2.inference.sliding_window_prediction as _swp
import nnunetv2.inference.predict_from_raw_data as _pfrd


@lru_cache(maxsize=2)
def _compute_gaussian_liu(tile_size, sigma_scale=1. / 8, value_scaling_factor=1,
                          dtype=torch.float16, device=torch.device('cuda', 0)):
    """Identica a `compute_gaussian` del nnU-Net incluso nel repo di Liu."""
    tmp = np.zeros(tile_size)
    tmp[tuple(i // 2 for i in tile_size)] = 1
    g = gaussian_filter(tmp, [i * sigma_scale for i in tile_size], 0, mode='constant', cval=0)
    g = torch.from_numpy(g).type(dtype).to(device)          # <-- conversione PRIMA
    g = (g / torch.max(g) * value_scaling_factor).type(dtype)
    g[g == 0] = torch.min(g[g != 0])
    return g


# va sostituita in DUE punti: `predict_from_raw_data` importa il nome direttamente,
# quindi rimpiazzare solo il modulo di origine non basterebbe
_swp.compute_gaussian = _compute_gaussian_liu
_pfrd.compute_gaussian = _compute_gaussian_liu
print("Gaussiana di fusione: sostituita con quella di Liu (nnU-Net 2.5).")


def carica_predictor(model_folder, checkpoint, folds, device, use_tta):
    p = nnUNetPredictor(
        tile_step_size=0.5,          # sovrapposizione della finestra scorrevole (default ufficiale)
        use_gaussian=True,
        use_mirroring=use_tta,       # TTA: ribaltamenti speculari
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    p.initialize_from_trained_model_folder(str(model_folder), use_folds=folds, checkpoint_name=checkpoint)
    return p


t0 = time.time()
print("Carico i modelli (una volta sola)...")
PRED1 = carica_predictor(M_STAGE1, CK_STAGE1, FOLDS, device, USE_TTA)
print(f"  stadio 1 pronto  ({PRED1.network.__class__.__name__}, {sum(x.numel() for x in PRED1.network.parameters())/1e6:.1f} M parametri)")
PRED2 = carica_predictor(M_STAGE2, CK_STAGE2, FOLDS, device, USE_TTA)
print(f"  stadio 2 pronto  ({PRED2.network.__class__.__name__}, {sum(x.numel() for x in PRED2.network.parameters())/1e6:.1f} M parametri)")
print(f"Fatto in {time.time()-t0:.1f} s")
''')

md(r"""
### La funzione che elabora un caso

Mette insieme tutti i pezzi: legge la TC, esegue i due stadi, applica il
post-processing e restituisce il punteggio del paziente.
""")

code(r'''
import tempfile, re

def elabora_caso(ct_path, salva_mappa_in=None):
    """Esegue l'intera pipeline su un esame. Restituisce un dizionario con il punteggio."""
    tempi = {}
    t_start = time.time()

    # --- lettura ---
    itk_img = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)

    # --- STADIO 1: dov'è il pancreas? (su immagine a bassa risoluzione) ---
    t = time.time()
    img_lowres = resample_img(itk_img, SPACING_STAGE1, is_label=False)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "scan_0000.nii.gz"
        sitk.WriteImage(img_lowres, str(tmp))
        arr, props = SimpleITKIO().read_images([str(tmp)])
        seg1 = PRED1.predict_single_npy_array(arr, props, None, None, False)
    tempi["stadio1"] = time.time() - t

    mask_lowres = sitk.GetImageFromArray(seg1.astype(np.uint8))
    mask_lowres.CopyInformation(img_lowres)

    # --- RITAGLIO: la scatola attorno al pancreas ---
    cropped, coords = crop_pancreas_roi(itk_img, mask_lowres)

    # --- STADIO 2: c'è un tumore dentro la scatola? ---
    t = time.time()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "scan_0000.nii.gz"
        sitk.WriteImage(cropped, str(tmp))
        arr, props = SimpleITKIO().read_images([str(tmp)])
        seg2, probs2 = PRED2.predict_single_npy_array(arr, props, None, None, True)
    tempi["stadio2"] = time.time() - t

    # --- post-processing e punteggio ---
    tumor_prob = post_processing(seg2, probs2)
    full_map, score, n_candidati = to_full_size(tumor_prob, coords, itk_img)

    if salva_mappa_in is not None:
        salva_mappa_in.parent.mkdir(parents=True, exist_ok=True)
        out = sitk.GetImageFromArray(full_map)
        out.CopyInformation(itk_img)
        sitk.WriteImage(out, str(salva_mappa_in), useCompression=True)

    return {
        "score": score,
        "volume_lesione_voxel": int((full_map > 0).sum()),
        "n_candidati": n_candidati,          # conteggio esatto, dalle confidenze
        "secondi": time.time() - t_start,
        "tempi": tempi,
        "crop": coords,
    }


def id_del_caso(path):
    """Identificativo di un esame a partire dal nome del file.

    Toglie l'estensione e l'eventuale suffisso `_0000` che nnU-Net usa per il canale.
    Così `100015_00001_0000.nii.gz` e `100015_00001.nii.gz` danno lo STESSO id,
    e le immagini si accoppiano correttamente con le segmentazioni manuali.
    """
    nome = Path(path).name
    for est in (".nii.gz", ".nii", ".mha", ".mhd"):
        if nome.endswith(est):
            nome = nome[: -len(est)]
            break
    return re.sub(r"_0000$", "", nome)


print("Funzioni `elabora_caso` e `id_del_caso` pronte.")
''')

# ---------------------------------------------------------------- 5. test
md(r"""
---
## 5 · Prova su un caso solo

**Fallo sempre prima del lotto.** Serve a verificare che tutto funzioni e,
soprattutto, a misurare **quanto tempo ci mette un caso** sul tuo hardware:
da lì stimi la durata dell'intero lotto.
""")

code(r'''
# Scegli un caso qualsiasi: il primo che trova nelle cartelle configurate
candidati = []
for d in [PDAC_IMAGES, HEALTH_IMAGES]:
    if d is not None and Path(d).exists():
        candidati += sorted(Path(d).glob("*.nii.gz"))

if not candidati:
    print("Nessuna immagine trovata: controlla PDAC_IMAGES / HEALTH_IMAGES al punto 1.")
else:
    caso = candidati[0]
    print(f"Caso di prova : {caso.name}  ({caso.stat().st_size/1e6:.1f} MB)")
    print("In corso... (su CPU può richiedere da 1 a 4 ore: la rete di Liu è grande)\n")

    r = elabora_caso(caso)

    print("=" * 62)
    print(f"  PUNTEGGIO PDAC : {r['score']:.4f}   (0 = nessun sospetto, 1 = massimo sospetto)")
    print("=" * 62)
    print(f"  candidati lesione : {r['n_candidati']}")
    print(f"  voxel evidenziati : {r['volume_lesione_voxel']:,}")
    print(f"  tempo totale      : {r['secondi']/60:.1f} min"
          f"   (stadio 1: {r['tempi']['stadio1']/60:.1f} min, stadio 2: {r['tempi']['stadio2']/60:.1f} min)")
    n_tot = len(candidati)
    print(f"\n  Stima per {n_tot} casi: circa {r['secondi']*n_tot/3600:.1f} ore")
''')

# ---------------------------------------------------------------- 6. batch
md(r"""
---
## 6 · Inferenza su tutti i casi

Elabora tutte le immagini delle cartelle configurate e salva i punteggi in un CSV.

- **Riprendibile**: se lo interrompi, alla riesecuzione riparte da dove era arrivato (`RESUME = True`).
- **Robusto**: se un caso va in errore, viene registrato e si prosegue col resto.
""")

code(r'''
import pandas as pd
import traceback

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "punteggi.csv"
MAPS_DIR = OUTPUT_DIR / "detection_maps"

# Elenco dei casi: (chiave, percorso, gruppo, etichetta vera).
# La CHIAVE è "gruppo/id": due file con lo stesso nome in PDAC/ e health/
# resterebbero altrimenti confusi tra loro.
lavoro = []
if PDAC_IMAGES is not None and Path(PDAC_IMAGES).exists():
    lavoro += [(f"PDAC/{id_del_caso(p)}", p, "PDAC", 1)
               for p in sorted(Path(PDAC_IMAGES).glob("*.nii.gz"))]
if HEALTH_IMAGES is not None and Path(HEALTH_IMAGES).exists():
    lavoro += [(f"health/{id_del_caso(p)}", p, "health", 0)
               for p in sorted(Path(HEALTH_IMAGES).glob("*.nii.gz"))]

if not lavoro:
    raise SystemExit("Nessun caso trovato: controlla i percorsi al punto 1.")

doppioni = len(lavoro) - len({k for k, *_ in lavoro})
if doppioni:
    print(f"ATTENZIONE: {doppioni} nomi duplicati nella stessa cartella.")

# --- ripresa da un'esecuzione precedente ---
# dtype=str: senza, id come "001" verrebbero letti come interi (1) e non
#            corrisponderebbero piu' -> ogni caso verrebbe rifatto da capo.
# keep_default_na=False: senza, la colonna "errore" vuota diventa NaN e i casi
#            riusciti sparirebbero silenziosamente dalle metriche.
fatti = {}
_chiavi_attuali = {k for k, *_ in lavoro}   # solo i casi di QUESTA coorte
if RESUME and CSV_PATH.exists():
    prec = pd.read_csv(CSV_PATH, dtype={"case_id": str, "chiave": str, "errore": str},
                       keep_default_na=False)
    # I casi falliti scrivono celle vuote nelle colonne numeriche: senza questa
    # riconversione pandas leggerebbe TUTTA la colonna come testo e le metriche
    # piu' avanti fallirebbero con un errore incomprensibile.
    for _c in ("score", "n_candidati", "volume_lesione_voxel", "secondi"):
        if _c in prec.columns:
            prec[_c] = pd.to_numeric(prec[_c], errors="coerce")
    for _, r in prec.iterrows():
        d = r.to_dict()                      # dict, non Series (altrimenti pandas va in errore)
        k = d.get("chiave") or f"{d['gruppo']}/{d['case_id']}"
        # si considera completato solo se ha davvero un punteggio: una riga
        # troncata da un'interruzione ha "errore" vuoto ma punteggio mancante,
        # e verrebbe altrimenti data per fatta e poi scartata dalle metriche
        if k in _chiavi_attuali and not d.get("errore") and pd.notna(d.get("score")):
            fatti[k] = d
    n_err_prec = len(prec) - len(fatti)
    print(f"Riprendo: {len(fatti)} casi già completati" +
          (f", {n_err_prec} in errore che verranno ritentati." if n_err_prec else "."))

restanti = sum(1 for k, *_ in lavoro if k not in fatti)
print(f"Da elaborare: {restanti} casi su {len(lavoro)} totali\n")

risultati = list(fatti.values())
t_inizio = time.time()

for i, (chiave, path, gruppo, etichetta) in enumerate(lavoro, 1):
    if chiave in fatti:
        continue
    case_id = id_del_caso(path)

    print(f"[{i}/{len(lavoro)}] {case_id} ({gruppo}) ... ", end="", flush=True)
    try:
        mappa = (MAPS_DIR / gruppo / f"{case_id}.nii.gz") if SAVE_DETECTION_MAPS else None
        r = elabora_caso(path, salva_mappa_in=mappa)
        riga = {"chiave": chiave, "case_id": case_id, "gruppo": gruppo, "etichetta": etichetta,
                "score": r["score"], "n_candidati": r["n_candidati"],
                "volume_lesione_voxel": r["volume_lesione_voxel"],
                "secondi": round(r["secondi"], 1), "errore": ""}
        print(f"score = {r['score']:.4f}   ({r['secondi']/60:.1f} min)")
    except Exception as e:
        riga = {"chiave": chiave, "case_id": case_id, "gruppo": gruppo, "etichetta": etichetta,
                "score": np.nan, "n_candidati": np.nan, "volume_lesione_voxel": np.nan,
                "secondi": np.nan, "errore": f"{type(e).__name__}: {e}"}
        print(f"ERRORE -> {type(e).__name__}: {e}")
        traceback.print_exc(limit=1)

    risultati.append(riga)
    pd.DataFrame(risultati).to_csv(CSV_PATH, index=False)   # salva dopo ogni caso

df = pd.DataFrame(risultati)
df["errore"] = df["errore"].fillna("").astype(str)
print(f"\nCompletato in {(time.time()-t_inizio)/3600:.2f} ore")
print(f"Punteggi salvati in: {CSV_PATH}")
n_err = (df["errore"].str.len() > 0).sum()
if n_err:
    print(f"ATTENZIONE: {n_err} casi in errore (colonna 'errore' del CSV). "
          f"Rilanciando questa cella verranno ritentati.")
df.head(10)
''')

# ---------------------------------------------------------------- 7. metriche
md(r"""
---
## 7 · Valutazione: AUROC e AP

- **AUROC** — quanto bene il modello separa i pazienti con PDAC dai controlli.
  Serve solo il punteggio e il gruppo di appartenenza.
- **AP** — quanto bene *localizza* la lesione. Richiede le segmentazioni manuali del tumore.
  Una lesione predetta conta come corretta se si sovrappone al tumore vero (IoU ≥ 0.10).

Gli intervalli di confidenza sono calcolati con bootstrap: **con poche decine di casi
sono larghi**, quindi vanno sempre riportati.
""")

code(r'''
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

d = df[df["errore"].fillna("").astype(str).str.len() == 0].dropna(subset=["score"])
y = d["etichetta"].astype(int).values
s = d["score"].values

_esclusi = len(df) - len(d)
if _esclusi:
    print(f"ATTENZIONE: {_esclusi} casi esclusi (errore o punteggio mancante).\n")

if len(np.unique(y)) < 2:
    print("Servono entrambi i gruppi (PDAC e controlli) per calcolare l'AUROC.")
    print("(se ti aspettavi entrambi, guarda la colonna 'errore' del CSV)")
else:
    auroc = roc_auc_score(y, s)

    # intervallo di confidenza con bootstrap
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boot.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print("=" * 62)
    print(f"  AUROC = {auroc:.3f}   (IC 95%: {lo:.3f} – {hi:.3f})")
    print("=" * 62)
    print(f"  casi: {int((y==1).sum())} PDAC / {int((y==0).sum())} controlli")
    print(f"  punteggio mediano  PDAC : {np.median(s[y==1]):.3f}")
    print(f"  punteggio mediano  sani : {np.median(s[y==0]):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    fpr, tpr, _ = roc_curve(y, s)
    ax[0].plot(fpr, tpr, lw=2, color="#126E82", label=f"AUROC = {auroc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="0.7", lw=1)
    ax[0].set_xlabel("1 − specificità"); ax[0].set_ylabel("sensibilità")
    ax[0].set_title("Curva ROC (livello paziente)"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].hist(s[y == 0], bins=20, alpha=.65, label="controlli", color="#3B5B92")
    ax[1].hist(s[y == 1], bins=20, alpha=.65, label="PDAC", color="#C1381E")
    ax[1].set_xlabel("punteggio del modello"); ax[1].set_ylabel("numero di casi")
    ax[1].set_title("Distribuzione dei punteggi"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()
''')

md(r"""
### AP a livello di lesione (facoltativo)

Questa cella gira **solo** se hai impostato `PDAC_LABELS` e salvato le mappe di detection.
Usa `picai_eval`, lo stesso strumento della challenge, così i numeri sono confrontabili
con quelli pubblicati.
""")

code(r'''
if PDAC_LABELS is None or not SAVE_DETECTION_MAPS:
    print("AP non calcolata: servono PDAC_LABELS e SAVE_DETECTION_MAPS = True.")
else:
    from picai_eval import evaluate

    # ATTENZIONE — la ground truth deve contenere SOLO il tumore.
    # Le segmentazioni PANORAMA sono multi-classe in un unico file
    # (0=sfondo, 1=TUMORE, 2=vene, 3=arterie, 4=parenchima, 5=dotto, 6=coledoco):
    # prendere "tutto ciò che è > 0" farebbe passare l'intero pancreas per tumore
    # e l'AP risulterebbe priva di senso. Il valutatore ufficiale usa infatti
    # `y_true_postprocess_func=lambda lbl: (lbl == 1)`. Con maschere gia' binarie
    # (solo tumore) il risultato e' identico, quindi va bene in entrambi i casi.
    SOLO_TUMORE = lambda lbl: (lbl == 1).astype(int)

    # Passiamo PERCORSI, non array: picai_eval li legge uno alla volta.
    # Tenere in memoria tutte le mappe (fino a ~900 MB l'una) esaurirebbe la RAM.
    ZERI_DIR = OUTPUT_DIR / "gt_controlli"          # GT vuote per i controlli
    ZERI_DIR.mkdir(parents=True, exist_ok=True)

    y_det, y_true, ids, saltati = [], [], [], []
    for _, r in d.iterrows():
        mappa = MAPS_DIR / r["gruppo"] / f"{r['case_id']}.nii.gz"
        if not mappa.exists():
            saltati.append(f"{r['case_id']}: manca la mappa di detection"); continue

        if r["gruppo"] == "PDAC":
            gt_path = Path(PDAC_LABELS) / f"{r['case_id']}.nii.gz"
            if not gt_path.exists():
                saltati.append(f"{r['case_id']}: manca la segmentazione manuale"); continue
            # verifica che ci sia davvero del tumore (classe 1) in quel file
            if int((sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))) == 1).sum()) == 0:
                saltati.append(f"{r['case_id']}: nessun voxel di tumore (classe 1) nella GT"); continue
        else:
            # i controlli non hanno lesioni: creiamo una GT tutta a zero, una volta sola
            gt_path = ZERI_DIR / f"{r['gruppo']}_{r['case_id']}.nii.gz"
            if not gt_path.exists():
                det_img = sitk.ReadImage(str(mappa))
                vuota = sitk.GetImageFromArray(np.zeros(det_img.GetSize()[::-1], dtype=np.uint8))
                vuota.CopyInformation(det_img)
                sitk.WriteImage(vuota, str(gt_path), useCompression=True)

        # id univoco "gruppo_caso": picai_eval indicizza per id e due omonimi
        # in PDAC/ e health/ si sovrascriverebbero a vicenda, sparendo dal conteggio
        y_det.append(str(mappa)); y_true.append(str(gt_path))
        ids.append(f"{r['gruppo']}_{r['case_id']}")

    for s in saltati[:10]:
        print(f"  (salto {s})")
    if len(saltati) > 10:
        print(f"  (... e altri {len(saltati)-10} casi saltati)")

    n_pos = sum(1 for _, r in d.iterrows() if r["gruppo"] == "PDAC")
    if not y_det:
        print("\nNessun caso valutabile: controlla PDAC_LABELS e le mappe di detection.")
    elif n_pos == 0 or all(Path(t).parent == ZERI_DIR for t in y_true):
        # senza nemmeno un caso positivo picai_eval restituisce AP = -0.0 e AUROC = nan,
        # numeri che sembrano risultati ma non lo sono: meglio fermarsi qui.
        print("\nAP NON calcolabile: non c'è nessun caso con tumore annotato.")
    else:
        metrics = evaluate(y_det=y_det, y_true=y_true, subject_list=ids,
                           min_overlap=0.10,                 # criterio ufficiale: IoU >= 0.10
                           y_true_postprocess_func=SOLO_TUMORE)
        print("\n" + "=" * 62)
        print(f"  AP    = {metrics.AP:.3f}   (localizzazione della lesione)")
        print(f"  AUROC = {metrics.auroc:.3f}   (ricalcolato da picai_eval)")
        print("=" * 62)
        print(f"  casi valutati: {len(ids)}   (saltati: {len(saltati)})")

        plt.figure(figsize=(5.5, 4.2))
        plt.plot(metrics.recall, metrics.precision, lw=2, color="#C18A1E", label=f"AP = {metrics.AP:.3f}")
        plt.xlabel("recall"); plt.ylabel("precision")
        plt.title("Curva Precision–Recall (livello lesione)")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
''')

# ---------------------------------------------------------------- 8. benchmark
md(r"""
---
## 8 · Perché qui NON c'è il benchmark out-of-fold

Nel notebook del baseline c'è una sezione che valuta il modello sui 2238 casi pubblici
di PANORAMA in modo *onesto*, usando per ogni caso il modello della fold che non lo
aveva visto in addestramento. **Per PanDx non è possibile**, ed è giusto spiegare perché.

### Il motivo

Liu non ha usato la suddivisione ufficiale in fold. Ha costruito la propria, stratificata
per dimensione della lesione, età e sesso — è uno dei suoi contributi ("data splitting
strategy"). Quella suddivisione **non è stata pubblicata**: non è nel repo né nella
cartella dei pesi.

Conseguenza pratica:

- non sappiamo quali casi fossero in validazione per ciascuna delle sue 5 fold;
- usare la suddivisione *del baseline* darebbe un risultato **sbagliato**, perché per
  molti casi sceglieremmo un modello che quel caso lo aveva invece già visto;
- far girare PanDx sui 2238 casi pubblici senza distinzione produce punteggi
  **ottimisticamente gonfiati**, non confrontabili con nulla.

### Cosa fare invece

L'unica valutazione onesta di PanDx è su **dati esterni**, mai visti dal modello:
cioè esattamente la coorte che metti al punto 1. È anche la domanda scientificamente
più interessante — *questo modello, addestrato su dati olandesi, funziona sui miei
pazienti?* — e la sezione 7 la risponde già.

> Se in futuro Liu pubblicasse i suoi split, il benchmark out-of-fold diventerebbe
> possibile: basterebbe copiare la sezione 8 dal notebook del baseline e sostituire
> il file delle fold.
""")

# ---------------------------------------------------------------- 9. note
md(r"""
---
## 9 · Note, limiti e problemi frequenti

### Tempi
| Hardware | Per caso | 100 casi |
|---|---|---|
| GPU NVIDIA (A6000, RTX 6000 Ada) | 3–8 min | 5–13 ore |
| CPU (MacBook Intel) | 1–4 h | **settimane** |

PanDx è **più lento del baseline**: la sua rete ha 113,9 milioni di parametri contro
30,7 milioni, e la soglia bassa (τ = picco/15) fa crescere lesioni più grandi, quindi
anche il post-processing costa di più.

Sul Mac usa `FAST_MODE = True` per verificare che tutto funzioni, poi fai girare
i lotti veri sulla macchina con GPU.

### Cosa serve che i dati siano
- TC addominale **con mezzo di contrasto, fase porto-venosa** — i modelli sono addestrati
  solo su questa. Con altre fasi i risultati calano, anche parecchio.
- Formato `.nii.gz`, orientamento e spacing qualsiasi (ci pensa il ricampionamento).

### Problemi frequenti
| Sintomo | Causa e rimedio |
|---|---|
| `Detected old nnU-Net plans format` | **normale, ignoralo.** I pesi sono del 2024: nnU-Net ricostruisce l'architettura dai plans e funziona correttamente (verificato). |
| `perform_everything_on_device=True is only supported for cuda` | normale su CPU, si adatta da solo |
| `nnUNet_raw is not defined` | normale: quelle variabili servono solo per l'addestramento, non per l'inferenza |
| `Unable to locate trainer class` | manca `liuPanDxTrainers.py` dentro `nnunetv2` → vedi `SETUP_GUIDE.md` |
| `NumPy 1.x cannot be run in NumPy 2.x` | kernel sbagliato, oppure numpy aggiornato per errore: serve `numpy<2` con torch 2.2.2 |
| `Lo stadio 1 non ha trovato il pancreas` | l'immagine non contiene l'addome, o non è una TC con contrasto |
| Memoria esaurita | riduci a `FOLDS = (0,)`, oppure usa una macchina con più RAM |

### Onestà sui risultati
- Se valuti su casi **già usati per addestrare** il modello (cioè i 2238 pubblici di
  PANORAMA), i punteggi sono **ottimisticamente distorti**. Per una valutazione onesta
  servono casi mai visti dal modello. Per PanDx la valutazione *out-of-fold*
  NON è praticabile: vedi il punto 8.
- Con poche decine di casi gli intervalli di confidenza sono larghi: dichiarali sempre.
- Confrontare "PDAC vs sani" è **più facile** di "PDAC vs altre patologie pancreatiche",
  che è il problema clinico vero. L'AUROC che ottieni non è direttamente confrontabile
  né con lo **0,916** di PanDx nella figura S8 dell'appendice di Alves et al. (coorte
  sequestrata, 1130 casi) né con lo **0,9263** dell'articolo PanDx (test set della
  challenge, 957 casi). In quelle coorti il 41% dei controlli aveva altre patologie
  pancreatiche: un confronto con controlli sani non regge.
""")

import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pandx_inference.ipynb")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook creato: {OUT}")
print(f"  celle: {len(nb['cells'])}")
