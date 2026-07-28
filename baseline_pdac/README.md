# baseline_pdac — inferenza PDAC con il baseline PANORAMA

Tutto il necessario per far girare l'algoritmo **baseline della PANORAMA challenge**
(Radboud UMC) su nuovi esami TC e ottenere, per ogni paziente, un punteggio 0–1 di
sospetto adenocarcinoma duttale del pancreas (PDAC), più AUROC e AP.

Questa cartella è **autosufficiente**: usa solo percorsi relativi, quindi puoi copiarla
su un altro computer (per esempio il Linux con GPU) e funziona senza modifiche.

---

## Contenuto

| File / cartella | Cosa è |
|---|---|
| `panorama_baseline_inference.ipynb` | **Il notebook principale** — apri questo |
| `SETUP_GUIDE.md` | installazione passo passo (macOS Intel e Linux GPU) |
| `requirements-macos-intel.txt` | dipendenze per Mac Intel (CPU) |
| `requirements-linux-cuda.txt` | dipendenze per Linux con GPU |
| `build_notebook.py` | rigenera il notebook (opzionale) |
| `models/` | pesi scaricati da Zenodo (~3,4 GB) |
| `risultati/` | creata all'esecuzione: `punteggi.csv` e mappe di detection |

## Avvio rapido

1. Leggi **`SETUP_GUIDE.md`** e prepara l'ambiente (la prima volta).
2. Apri `panorama_baseline_inference.ipynb`.
3. Seleziona il kernel **PANORAMA baseline (.venv-nnunet)**.
4. Nella **cella 1** indica dove sono i tuoi esami.
5. Esegui le celle in ordine.

> **Prima prova:** metti `FAST_MODE = True` nella cella 1. È ~40 volte più veloce e
> serve a verificare che la catena funzioni. I risultati **non** sono quelli ufficiali:
> per i numeri veri rimetti `FAST_MODE = False`.

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
cui i modelli sono addestrati.

## Dove sta l'ambiente Python

Il virtualenv `.venv-nnunet` sta nella **cartella superiore** (la radice del progetto),
non qui dentro. Due motivi:

- un venv **non è portabile** tra sistemi operativi: sul Linux va comunque ricreato,
  quindi copiarlo insieme alla cartella sarebbe inutile (e sono 13 GB);
- lo stesso ambiente servirà anche al notebook di PanDx (il modello di Liu).

## Stato: verificato

La pipeline è stata eseguita davvero su casi PANORAMA reali:

| Caso | Verità | Punteggio del modello |
|---|---|---|
| `100030_00001` | PDAC (istologia) | **0.9984** |
| `100015_00001` | controllo (NIH) | **0.0118** |

## Riferimenti e licenze

- Codice baseline: <https://github.com/DIAGNijmegen/PANORAMA_baseline> (Apache-2.0)
- Pesi: <https://zenodo.org/records/11160381> — **CC BY-NC 4.0, solo uso non commerciale**
- Articolo: Alves N, Schuurmans M, Rutkowski D, et al. *Lancet Oncol* 2026;27(1):116–124

⚠️ Strumento di **ricerca**, non un dispositivo medico approvato. Non usare per
decisioni cliniche su pazienti reali.
