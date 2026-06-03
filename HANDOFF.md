# HANDOFF — Progetto PANORAMA PDAC (studio TC del tumore del pancreas)

> Documento di passaggio di consegne: stato del progetto, come è organizzato e come
> continuare (su questo PC o su un altro, es. Linux). Aggiornato: 2026-06-03.

---

## 1. Obiettivo
Studio su TC con contrasto (CECT) del tumore del pancreas (**PDAC**) usando il dataset
pubblico **PANORAMA** (grand-challenge.org). Si scarica un sottoinsieme a scelta delle
immagini + le segmentazioni, e si produce un report Excel con clinica e dettaglio
segmentazioni.

Fonti dati:
- https://panorama.grand-challenge.org/datasets-imaging-labels/
- https://github.com/DIAGNijmegen/panorama_labels

---

## 2. Struttura della cartella (`/Volumes/Transcend/dataset_PANORAMA`)
```
panorama_download.ipynb   # NOTEBOOK PRINCIPALE (apri questo e fai Run All)
build_notebook.py         # genera il notebook (opzionale, solo se vuoi rigenerarlo)
requirements.txt          # pacchetti Python esatti (per ricreare il venv, es. su Linux)
.vscode/settings.json     # interprete = ./.venv (per Cursor/VSCode)
.venv/                    # ambiente Python 3.12.13 (NON copiare tra OS: ricrearlo)
cache/                    # clinical_information.xlsx, labels_tree.json, zenodo_index.csv
imagesTr/                 # volumi TC scaricati  ({study_id}_0000.nii.gz)
labelsTr/                 # maschere segmentazione ({study_id}.nii.gz)
panorama_report.xlsx      # report generato dal notebook (rigenerato ad ogni run)
HANDOFF.md                # questo file
dataset_NIfTI_ERASME_PDAC/    # DATASET SEPARATO (Erasme) — NON parte di PANORAMA
dataset_NIfTI_ERASME_health/  # DATASET SEPARATO (Erasme) — NON parte di PANORAMA
```
> Nota: la cartella un tempo si chiamava `PANORAMA_PDAC`; rinominata in `dataset_PANORAMA`
> il 2026-06-03. È rimasto un symlink di compatibilità `PANORAMA_PDAC → dataset_PANORAMA`
> (rimovibile con `rm /Volumes/Transcend/PANORAMA_PDAC`, toglie solo il link non i dati).

---

## 3. Ambiente Python / kernel
- Python **3.12.13** (Homebrew: `/usr/local/bin/python3.12`).
- venv in `./.venv`. Kernel Jupyter registrato: **`panorama-pdac`** ("PANORAMA PDAC (.venv 3.12)").
- Interprete impostato in `.vscode/settings.json` → `${workspaceFolder}/.venv/bin/python`.

**Per usarlo (in Cursor/VSCode):** apri la cartella, apri `panorama_download.ipynb`,
seleziona il kernel **PANORAMA PDAC (.venv 3.12)** (o l'interprete `.venv`), poi **Run All**.

**Per ricreare il venv da zero (es. su Linux) — Python ESATTAMENTE 3.12.13:**
```bash
pyenv install 3.12.13      # se non gia' presente (oppure installa 3.12.13 da sorgente/deadsnakes)
pyenv local  3.12.13       # in questa cartella; legge/crea .python-version
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name panorama-pdac \
    --display-name "PANORAMA PDAC (.venv 3.12)"
```
> La versione di Python è fissata a **3.12.13** in `.python-version` (riproducibilità).
> Il venv NON è portabile tra macOS e Linux: va sempre ricreato (i dati e il notebook sì).

---

## 4. Come funziona il download (notebook)
Le immagini stanno in 4 zip su Zenodo (~194 GB totali). Grazie alle richieste HTTP-range
(`remotezip`) il notebook estrae **solo i casi scelti** senza scaricare gli zip interi.
Le segmentazioni arrivano dal repo GitHub (piccole).

**Configurazione (punto 1 del notebook):**
- `MODE`:
  - `"percent"` → `TARGET_PERCENT` % del dataset (2238 casi), diviso 50/50 PDAC/controlli.
    **È CUMULATIVO e incrementale**: alza la % (es. 1 → 5 → 25) e scarica solo i nuovi.
  - `"count"` → numeri espliciti `N_PDAC` / `N_CONTROL`.
  - `"all"`  → tutti i casi (676 PDAC + 1562 controlli, **sbilanciato**).
  - `"explicit"` → lista `STUDY_IDS`.
- `CONTROL_STRATEGY="sure_negatives"` → controlli = negativi sicuri: prima NIH (pancreas
  normale), poi radiology+follow-up 3 anni, poi radiology.
- `PDAC_PREFER_MANUAL=True` → i PDAC scelti hanno la **lesione segmentata a mano** (esperto)
  finché ce ne sono (~482 manuali su 676).
- `SEED=42` → selezione **riproducibile e annidata** (1% ⊆ 5% ⊆ 25% ⊆ all).

> ⚠️ Per l'incrementale NON cambiare `SEED`, `CONTROL_STRATEGY`, `PDAC_PREFER_MANUAL` tra le run.

**Robustezza:** download resumabile (salta i file già presenti), timeout di lettura 60s +
riapertura connessione + 5 ritentativi + skip-on-fail, e stop pulito se il disco si riempie.

**Output:** `panorama_report.xlsx` (riscritto ad ogni run) con fogli `scaricati`, `riepilogo`,
`legenda`, `note`. Per ogni esame: clinica + distretti segmentati + lesione manuale/AI.

---

## 5. Fatti chiave sul dataset (verificati)
- **2238** TC porto-venose (.nii.gz). Composizione: **676 PDAC** / **1562 non-PDAC**.
- NON esiste una classe "sano" pura: i sani sono dentro i non-PDAC; gli unici controlli
  esplicitamente normali sono gli **80 casi NIH**.
- Segmentazioni: maschere multi-classe `.nii.gz`, 7 classi
  (0=sfondo,1=lesione PDAC,2=vene,3=arterie,4=pancreas,5=dotto,6=coledoco).
  Le strutture anatomiche sono SEMPRE generate da AI; solo la lesione può essere manuale
  (cartella `manual_labels`, ~482 casi).
- Spessore/spacing: NON sono nel file clinico → si leggono dall'header NIfTI
  (`nibabel ... .header.get_zooms()`). In-plane 0.39–0.98 mm, spessore slice 0.45–5.0 mm
  (mediana 1.0), multi-scanner.
- Licenza: **CC BY-NC 4.0** (solo uso non commerciale). Download Zenodo/GitHub senza
  registrazione. Citare PANORAMA (Alves et al. 2024).

---

## 6. Stato attuale
- Scaricati finora: ~**173** immagini + 173 maschere (≈8% del dataset).
- Disco: **molto pieno** (~10 GiB liberi). Su questo disco entra fino a ~**15%** (~336 casi);
  per 25%/all serve un disco più capiente.
- Banda utente ~2 MB/s (≈ ore per le percentuali alte).

---

## 7. Continuare su un altro PC (es. Linux + Cursor)
1. **Codice** via GitHub (repo privato): `panorama_download.ipynb`, `build_notebook.py`,
   `requirements.txt`, `HANDOFF.md`, `.vscode/`. Con `.gitignore` che esclude
   `.venv/ imagesTr/ labelsTr/ cache/ *.xlsx` (dati grossi + licenza → NON su GitHub).
2. **Dati**: porta il disco esterno (i dati sono già lì) o `rsync` di `imagesTr/ labelsTr/ cache/`.
   In alternativa lascia ri-scaricare dal notebook (riproducibile, ma lento).
3. **Venv**: ricrealo da `requirements.txt` (vedi §3). Registra il kernel.
4. Apri in Cursor, seleziona il kernel/interprete `.venv`, `Run All` → riprende dai file presenti.

---

## 8. Note / gotcha
- La config globale di Jupyter sul Mac è rotta (manca `jupyter_contrib_nbextensions`):
  per eseguire i notebook headless usare **`nbclient`**, NON `jupyter nbconvert`.
- Chiudere `panorama_report.xlsx` in Excel prima di rieseguire (altrimenti non si sovrascrive).
- Le due cartelle `dataset_NIfTI_ERASME_*` sono un dataset diverso: non c'entrano col notebook PANORAMA.
