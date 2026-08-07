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
| `panorama_baseline_inference_v2_plus_segmentation.ipynb` | come sopra, **+ salva la segmentazione a 7 classi** |
| `panorama_baseline_inference_v3_seg_plus_embeddings.ipynb` | come la v2, **+ estrae l'embedding del bottleneck** |
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

---

## Le tre versioni del notebook

Le versioni v2 e v3 **aggiungono** output senza toccare il codice originale: punteggio,
mappa di detection, AUROC e AP escono dallo stesso identico codice della v1.

| | Punteggio + AUROC/AP | Segmentazione 7 classi | Embedding 320-d |
|---|---|---|---|
| `..._inference.ipynb` (v1) | sì | — | — |
| `..._v2_plus_segmentation.ipynb` | sì | sì | — |
| `..._v3_seg_plus_embeddings.ipynb` | sì | sì | sì |

**Come fanno a non cambiare i risultati.** La v2 avvolge `predict_single_npy_array` con una
funzione che chiama il metodo originale e ne restituisce l'uscita immutata, mettendo da parte una
copia della segmentazione. La v3 aggiunge due *forward hook*, che sono in sola lettura: osservano i
tensori mentre passano senza modificarli. In entrambi i casi `elabora_caso` viene eseguita
invariata.

**Verificato** su un caso PDAC e un controllo, con tutte e 5 le fold: punteggi identici a 10
decimali, `n_candidati` e `volume_lesione_voxel` identici, AUROC e AP identiche, e mappe di
detection identiche **voxel per voxel**.

### Cosa producono in più

**v2** — `risultati/segmentazioni/<gruppo>/<caso>.nii.gz`, l'argmax grezzo dello stadio 2
(0 sfondo, 1 lesione PDAC, 2 vene, 3 arterie, 4 parenchima, 5 dotto pancreatico, 6 coledoco),
reincollato alle dimensioni della TC. Nessun post-processing: quel filtro appartiene alla mappa di
detection, non alla segmentazione.

**v3** — due CSV in formato lungo, una riga per `(paziente, fold)` e 320 colonne `emb_000…emb_319`:

- `embedding_media_semplice.csv` — finestre pesate tutte uguali (descrive l'intera scatola)
- `embedding_media_pesata.csv` — finestre pesate per probabilità di tumore (descrive la lesione)

più `rapporto_rumore_segnale_*.csv`, che confronta la variabilità fra fold con quella fra pazienti.

### Avvertenze

- **`RESUME = True` salta i casi già elaborati**, che quindi non producono né segmentazioni né
  embedding. Per generarle su tutta la coorte usa `RESUME = False` o una `OUTPUT_DIR` nuova.
- **Non impostare `nnUNet_compile`**: `torch.compile` può saltare i forward hook. La v3 si ferma
  con un errore esplicito se la trova attiva.
- **File `._*`**: copiando dati da macOS si creano file nascosti che il ciclo raccoglie come se
  fossero esami e che finiscono in errore. Vanno rimossi prima del lotto.
- L'embedding descrive **il bounding box attorno al pancreas**, non il pancreas: il ritaglio è
  allargato di 100 × 50 × 15 mm per lato.
