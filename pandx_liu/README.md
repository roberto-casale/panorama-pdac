# pandx_liu — inferenza PDAC con PanDx (vincitore PANORAMA)

Tutto il necessario per far girare **PanDx** (H. Liu, team DTI — Siemens Healthineers),
l'algoritmo che ha **vinto** la PANORAMA challenge, su nuovi esami TC e ottenere per ogni
paziente un punteggio 0–1 di sospetto adenocarcinoma duttale del pancreas (PDAC),
più AUROC e AP.

Questa cartella è **autosufficiente**: usa solo percorsi relativi, quindi puoi copiarla
su un altro computer (per esempio il Linux con GPU) e funziona senza modifiche.

---

## Contenuto

| File / cartella | Cosa è |
|---|---|
| `pandx_inference.ipynb` | **Il notebook principale** — apri questo |
| `SETUP_GUIDE.md` | installazione passo passo (macOS Intel e Linux GPU) |
| `requirements-macos-intel.txt` | dipendenze per Mac Intel (CPU) |
| `requirements-linux-cuda.txt` | dipendenze per Linux con GPU |
| `build_notebook.py` | rigenera il notebook (opzionale) |
| `models/` | pesi scaricati dal Google Drive di Liu (**~5,1 GB**) |
| `risultati/` | creata all'esecuzione: `punteggi.csv` e mappe di detection |

## Avvio rapido

1. Leggi **`SETUP_GUIDE.md`** e prepara l'ambiente (la prima volta).
2. Apri `pandx_inference.ipynb`.
3. Seleziona il kernel **PANORAMA baseline (.venv-nnunet)**.
4. Nella **cella 1** indica dove sono i tuoi esami.
5. Esegui le celle in ordine.

> **Prima prova:** metti `FAST_MODE = True` nella cella 1. È ~40 volte più veloce e
> serve a verificare che la catena funzioni. Per i numeri veri rimetti `FAST_MODE = False`.

---

## In cosa differisce dal baseline

PanDx usa lo **stesso schema a due stadi** del baseline, ma cambia tre cose mirate.
È da lì che viene il suo vantaggio (**AP 0,720 contro 0,634** sulla coorte di test sequestrata (fig. S8, Lancet Oncol)):

| | Baseline | **PanDx (Liu)** |
|---|---|---|
| Stadio 1 (trova il pancreas) | proprio | **lo stesso**, riusato senza modifiche |
| Stadio 2 (trova il tumore) | U-Net standard, `Dataset104` | **ResU-Net**, `Dataset107` |
| Estrazione delle lesioni | soglia adattiva, fattore di default: τ = picco / 2,5 | **fattore 15**: τ = picco / 15 |
| Filtro anti-falsi-positivi | azzera fuori dal pancreas dilatato | **nessun filtro** |

Nelle sue parole: riusa i modelli del baseline «*eliminating the need for retraining*»
sostituisce la loss Dice+CE con la sola CE «*to better fit the detection task*»
e la U-Net con una ResU-Net.

---

## Rapporto con `baseline_pdac/`

Le due cartelle sono indipendenti ma condividono l'ambiente Python:

- **Stesso venv** (`.venv-nnunet` nella cartella superiore): non serve crearne un secondo.
- **Due trainer personalizzati diversi**, entrambi necessari se usi entrambi i notebook:
  `customTrainerCEcheckpoints.py` (baseline) e `liuPanDxTrainers.py` (PanDx).
  Non si sostituiscono a vicenda.
- **Lo stadio 1 è lo stesso modello.** I pesi sono duplicati in tutte e due le cartelle
  per renderle autonome; se ti serve spazio puoi tenerne una copia sola.

---

## Come organizzare i dati

```
CARTELLA_DATI/
├── PDAC/
│   ├── images/     ← TC dei pazienti con tumore     (.nii.gz)
│   └── labels/     ← segmentazioni manuali del tumore (opzionale, serve per l'AP)
└── health/
    └── images/     ← TC dei controlli                (.nii.gz)
```

Devono essere **TC con mezzo di contrasto in fase porto-venosa**: è l'unica fase su
cui i modelli sono addestrati. Il suffisso `_0000` nei nomi viene gestito in automatico.

---

## Un limite da conoscere

**Per PanDx non è possibile il benchmark out-of-fold** che invece c'è nel notebook del
baseline: Liu non ha pubblicato la sua suddivisione in fold (stratificata per dimensione
della lesione, età e sesso). Far girare il modello sui casi pubblici PANORAMA darebbe
punteggi **gonfiati**, perché li ha già visti in addestramento.

L'unica valutazione onesta è su **dati esterni**, cioè la tua coorte. Il punto 8 del
notebook lo spiega per esteso.

---

## Riferimenti e licenze

- Codice: <https://github.com/han-liu/PDAC_Detection>
- Articolo: Liu H, Gao R, Krieg E, Grbic S. *PanDx*, MICCAI Workshop on Applications of
  Medical AI, 2025, pp. 63–71 ([arXiv 2503.10068](https://arxiv.org/abs/2503.10068))
- Challenge: Alves N, et al. *Lancet Oncol* 2026;27(1):116–124
- Pesi dello stadio 1: <https://zenodo.org/records/11160381> — **CC BY-NC 4.0**, solo uso non commerciale

⚠️ Strumento di **ricerca**, non un dispositivo medico approvato. Non usare per
decisioni cliniche su pazienti reali.
