# PANORAMA PDAC — downloader & report

Strumenti per scaricare **a percentuale** un sottoinsieme del dataset pubblico
[PANORAMA](https://panorama.grand-challenge.org/) (TC con contrasto del pancreas, per lo
studio del tumore **PDAC**) e generare un report Excel con clinica e dettaglio delle
segmentazioni. Pensato per essere eseguito in Cursor/VSCode (estensione Jupyter).

> ℹ️ Questo repo contiene **solo il codice**. I dati immagine/segmentazione e i metadati
> clinici **non** sono inclusi (sono grandi e soggetti a licenza **CC BY-NC 4.0**): vengono
> scaricati dal notebook stesso da Zenodo/GitHub.

---

## Contenuto del repo
| File | Cosa è |
|---|---|
| `panorama_download.ipynb` | **Notebook principale** (apri questo e fai Run All) |
| `build_notebook.py` | genera il notebook (opzionale) |
| `requirements.txt` | dipendenze Python (Python 3.12) |
| `.vscode/settings.json` | interprete = `./.venv` |
| `HANDOFF.md` | documento di stato dettagliato del progetto |

Cartelle ignorate da git (vedi `.gitignore`): `.venv/`, `imagesTr/`, `labelsTr/`, `cache/`,
`*.xlsx`, e i dataset Erasme separati.

---

## Avvio rapido (macchina già configurata)
1. Apri la cartella in Cursor/VSCode.
2. Apri `panorama_download.ipynb`, seleziona il kernel **PANORAMA PDAC (.venv 3.12)** (o l'interprete `.venv`).
3. Al **punto 1** imposta quanto scaricare, poi **Run All**.

## Setup su un NUOVO computer (es. Linux)
```bash
# 1) codice
git clone https://github.com/roberto-casale/panorama-pdac.git
cd panorama-pdac

# 2) ambiente (Python 3.12)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name panorama-pdac \
    --display-name "PANORAMA PDAC (.venv 3.12)"

# 3) apri in Cursor, seleziona il kernel/interprete .venv, Run All
```
- I **dati** non sono nel repo: alla prima esecuzione il notebook li ri-scarica da Zenodo
  (lento), **oppure** copia le cartelle `imagesTr/ labelsTr/ cache/` dal vecchio computer
  (es. disco esterno o `rsync`) per riprendere dove eri (il download è *resumabile*).
- Il `.venv` **non** va copiato tra sistemi: va sempre ricreato (passo 2).

---

## Configurazione (punto 1 del notebook)
- `MODE`: `"percent"` (default) usa `TARGET_PERCENT` % del dataset (2238 casi) 50/50 PDAC/controlli —
  **cumulativo/incrementale**: alza la % (1 → 5 → 25) e scarica solo i nuovi.
  Altri modi: `"count"`, `"all"` (tutti, sbilanciato), `"explicit"` (lista `STUDY_IDS`).
- `CONTROL_STRATEGY="sure_negatives"`: controlli = negativi sicuri (NIH → radiology+3yFU → radiology).
- `PDAC_PREFER_MANUAL=True`: PDAC con lesione segmentata a mano per primi.
- `SEED=42`: selezione riproducibile e **annidata** (1% ⊆ 5% ⊆ … ⊆ all) → incrementale sicuro.

Output: `panorama_report.xlsx` (clinica + distretti segmentati + lesione manuale/AI), rigenerato ad ogni run.

---

## Dati e licenza
- Dataset **PANORAMA** — licenza **CC BY-NC 4.0** (solo uso non commerciale). Citare Alves et al. 2024.
- Immagini: 4 zip su Zenodo (~194 GB); segmentazioni: repo GitHub `DIAGNijmegen/panorama_labels`.
- 2238 TC: 676 PDAC / 1562 non-PDAC (nessuna classe "sano" pura; 80 controlli NIH normali).
