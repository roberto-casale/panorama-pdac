# PROMEMORIA — PanDx bloccato da crash GPU (29 lug 2026)

## Cosa devo fare
Far girare l'inferenza PanDx (Liu) sui casi PANORAMA. È l'unica cosa
rimasta: ambiente, trainer e baseline sono già completi e funzionanti.

## Macchina
- Ubuntu, utente `radiology` (account CONDIVISO — non toccare ~/.gitconfig,
  identità git impostata con --local nei repo)
- NVIDIA RTX A6000 48 GB, driver 535.309.01, **GPU UNICA** (nessuna grafica
  integrata: il monitor deve restare sulla A6000)
- 30 GB RAM, swap 2 GB
- Disco: partizione unica, ~96% pieno, ~43 GB liberi
- Lavoro **da remoto via rustdesk** (no SSH verificato)

## Ambiente (già pronto, non reinstallare)
Cartella unica: `~/Desktop/PhD_pancreas/panorama-inference`
Repo: github.com/roberto-casale/panorama-pdac (ultimo commit mio: bb1e1d1)

- `.venv-nnunet` → inferenza. Python 3.12.13, torch **2.5.1+cu121** (NON
  aggiornare: da 2.6 in su i checkpoint non si caricano), nnunetv2 2.5.1,
  numpy 2.5.1, report_guided_annotation 0.3.4, picai_eval 1.4.13
- `.venv` → downloader (numpy 2.4.6, pandas 3.0.3). INCOMPATIBILE col primo
- Kernel Jupyter: "PANORAMA baseline (.venv-nnunet)" e "PANORAMA PDAC (.venv 3.12)"
- Trainer installati dentro .venv-nnunet (spariscono se reinstallo nnunetv2):
  - `.../nnUNetTrainer/customTrainerCEcheckpoints.py`
  - `.../nnUNetTrainer/variants/loss/liuPanDxTrainers.py`
- Dati: `imagesTr/` (22 casi, 1,5 G), `cache/`, `panorama_report.xlsx` nella radice
- Symlink in `~/Desktop/PhD_pancreas/dati_tc/{PDAC,health}` (22 casi, 11+11)

## Baseline: FATTO e VERIFICATO
Pipeline confrontata riga per riga con `DIAGNijmegen/PANORAMA_baseline`
(`src/data_utils.py` + `src/process.py`): resample_img, CropPancreasROI,
PostProcessing, GetFullSizDetectionMap tutte identiche o equivalenti.
Spacing (4.5,4.5,9.0), margini [100,50,15] mm, sitkFloat32, 5 fold + TTA. OK.

Risultati sui 22 casi PANORAMA (11 PDAC / 11 controlli):
- AUROC 1.000, AP 1.000 — **gonfiati**: sono casi del training set
- Benchmark out-of-fold: AUROC 1.000 lo stesso (controlli "sure negatives",
  problema più facile del pubblicato 0.915 dove il 41% dei controlli aveva
  altra patologia pancreatica)
- Casi difficili: 101083_00001 → 0.666 (out-of-fold 0.238), 101485_00001 → 0.799
- 16 minuti per 22 casi. File: `baseline_pdac/risultati/punteggi.csv`

## IL PROBLEMA
Lanciando il punto 5 di PanDx (113,9 M parametri, ResidualEncoderUNet — vs
30,7 M del baseline) la sessione grafica muore. Il PC resta acceso.

Log (`journalctl -b -1`, 29 lug 22:59–23:02):
    NVRM: Xid (PCI:0000:6b:00): 8, pid=..., name=Xorg, Channel 00000012
    (EE) NVIDIA(0): The NVIDIA X driver has encountered an error
    (II) NVIDIA(0): Error recovery was successful.
in loop ogni 8 secondi per 3+ minuti. Ultimo Xid colpisce `tokio-runtime-w`
(= rustdesk) → perdo il controllo remoto.

**Xid 8 = "GPU stopped processing", attribuito a Xorg, NON al processo python.**
Nessun OOM nei log. Il baseline non ha mai dato problemi.

Causa: la A6000 fa contemporaneamente display (Xorg, gnome-shell, rustdesk,
Cursor) e calcolo. Sotto PanDx il percorso grafico va in fault, muore la
sessione, e con lei il kernel del notebook.

## COSA NON HO ANCORA PROVATO
1. tmux + script (invece del notebook): il calcolo sopravvive alla morte
   della sessione grafica
       jupyter nbconvert --to script pandx_liu/pandx_inference.ipynb
       tmux new -s pandx   → lancio → Ctrl+B poi D → mi disconnetto
2. `sudo nvidia-smi -pm 1` (persistence mode, era Off)
3. `sudo nvidia-smi -pl 250` (tetto potenza, default 300 W)
4. Girare con Chrome e Cursor CHIUSI e rustdesk disconnesso
5. Aggiornare il driver 535.309.01 (macchina condivisa: da concordare)

## Stato di PanDx
Pesi CANCELLATI per liberare 5,1 GB. Da riscaricare (~7 min a 15 MB/s):
    cd ~/Desktop/PhD_pancreas/panorama-inference/pandx_liu
    python -c "import gdown; gdown.download_folder(
      'https://drive.google.com/drive/folders/1RpbofQDrQNzwfYjFhQYRRWCN8HhIoZQP',
      output='models')"
Struttura attesa: models/workspace/nnUNet_results/{Dataset103_...,
Dataset107_PDAC_Detection/nnUNetTrainerCELossLesionSplit__nnUNetPlans_v3__3d_fullres}
10 file .pth, 5,1 G totali.

Note: per PanDx NON esiste benchmark out-of-fold (Liu non ha pubblicato la
suddivisione in fold). Riferimenti pubblicati da non mescolare:
0.916/0.720 (Lancet Oncol, 1130 casi) e 0.9263/0.7243 (arXiv PanDx, 957 casi).

## Obiettivo finale
Predizioni su casi nuovi (coorte Erasme) con baseline fedele alla letteratura.
Configurazione: FAST_MODE=False, 5 fold, TTA attiva.
Cartella dati separata per coorte (NON riusare dati_tc, contiene training set).
Label: il tumore deve avere valore 1 (convenzione PANORAMA), altrimenti
picai_eval salta il caso.
