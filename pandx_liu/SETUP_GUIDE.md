# Guida di installazione — PanDx (H. Liu), vincitore della PANORAMA challenge

Guida passo passo per far girare `pandx_inference.ipynb` su una macchina nuova.
Pensata anche per essere data in pasto a Cursor: ogni passo è un comando da eseguire,
con la spiegazione del perché.

- **Cosa fa il notebook:** prende TC addominali con contrasto (fase porto-venosa, `.nii.gz`)
  e per ogni caso restituisce un punteggio 0–1 di sospetto PDAC, poi calcola AUROC e AP.
- **Cosa NON è:** un dispositivo medico. È uno strumento di ricerca.

> **Se hai già installato il notebook del baseline** (`baseline_pdac/`), l'ambiente Python
> è **lo stesso**: salta ai punti 3 e 4 (trainer di Liu e pesi). Non serve un secondo venv.

---

## 0 · Il vincolo da capire prima di iniziare

Un solo dettaglio determina tutta l'installazione:

> **PyTorch < 2.3 non funziona con NumPy ≥ 2**, e **PyTorch ≥ 2.6 non riesce a caricare
> questi checkpoint** (dalla 2.6 `torch.load` usa `weights_only=True` e i pesi, che
> contengono oggetti numpy, vengono rifiutati).

Quindi:

| Sistema | Versione di torch | Vincolo su numpy |
|---|---|---|
| **Linux con GPU** | `torch==2.5.1` (comunque **< 2.6**) | nessuno |
| **macOS Intel** | `torch==2.2.2` (è l'ultima disponibile) | **numpy < 2** obbligatorio |

Le sezioni sotto ne tengono già conto.

---

## 1 · Installazione su Linux con GPU (consigliata per l'uso vero)

```bash
# Ubuntu: se manca Python 3.12 ->  sudo add-apt-repository ppa:deadsnakes/ppa
#                              sudo apt install python3.12 python3.12-venv
cd /percorso/del/progetto      # la cartella che contiene pandx_liu
python3.12 -m venv .venv-nnunet
source .venv-nnunet/bin/activate
pip install --upgrade pip

# PyTorch con CUDA — LA VERSIONE VA FISSATA
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121

# tutto il resto
pip install -r pandx_liu/requirements-linux-cuda.txt
```

> ⚠️ **Non scrivere `pip install torch` senza versione.** Gli indici CUDA danno
> versioni troppo recenti (cu124 → 2.6.0, cu118 → 2.7.1) che **non caricano i pesi**.

Verifica che la GPU sia vista:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Registra il kernel Jupyter:

```bash
python -m ipykernel install --user --name panorama-nnunet \
       --display-name "PANORAMA baseline (.venv-nnunet)"
```

---

## 2 · Installazione su macOS Intel (solo per sviluppo e test)

⚠️ Su CPU l'inferenza richiede **da 1 a 4 ore per caso**: la rete di Liu ha 113,9
milioni di parametri (contro i 30,7 del baseline), quindi è sensibilmente più lenta.

```bash
cd /percorso/del/progetto
python3.12 -m venv .venv-nnunet
source .venv-nnunet/bin/activate
pip install --upgrade pip

# ATTENZIONE: i vincoli vanno passati TUTTI INSIEME in un solo comando,
# altrimenti pip installa numpy 2.x e poi torch si rompe.
pip install -r pandx_liu/requirements-macos-intel.txt

python -m ipykernel install --user --name panorama-nnunet \
       --display-name "PANORAMA baseline (.venv-nnunet)"
```

Controllo (non deve comparire nessun avviso su NumPy):

```bash
python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"
```

Atteso: `torch 2.2.2 | numpy 1.26.4`

---

## 3 · Il trainer di Liu (passo obbligatorio)

I pesi dello stadio 2 sono stati addestrati con un trainer personalizzato chiamato
`nnUNetTrainerCELossLesionSplit`. nnU-Net lo cerca **dentro il proprio pacchetto**:
se non lo trova, il caricamento del modello fallisce con

```
RuntimeError: Unable to locate trainer class nnUNetTrainerCELossLesionSplit
```

> ⚠️ È un trainer **diverso** da quello del baseline (`nnUNetTrainer_Loss_CE_checkpoints`).
> Se usi entrambi i notebook, servono **tutti e due** i file: non si sostituiscono.

```bash
# assicurati che il venv sia ATTIVO (source .venv-nnunet/bin/activate)
TDIR=$(python -c "import nnunetv2, os; print(os.path.join(nnunetv2.__path__[0],'training','nnUNetTrainer','variants','loss'))")
curl -fsSL https://raw.githubusercontent.com/han-liu/PDAC_Detection/main/packages/nnunetv2/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerCELoss.py \
     -o "$TDIR/liuPanDxTrainers.py"

# verifica che sia il file giusto e non una pagina di errore
grep -q "class nnUNetTrainerCELossLesionSplit" "$TDIR/liuPanDxTrainers.py" \
  && echo "OK, installato in: $TDIR" \
  || echo "ERRORE: download fallito, il file non contiene la classe attesa"
```

> **Nota:** se reinstalli o aggiorni `nnunetv2`, questo file va rimesso.

---

## 4 · Scarica i pesi (~5,1 GB)

I pesi di Liu stanno su **Google Drive**, non su Zenodo. Servono `gdown`.

> ⚠️ Sono **circa 5,1 GB** (i suoi checkpoint pesano ~912 MB l'uno, contro i 235 MB
> del baseline): metti in conto lo spazio disco e il tempo di scaricamento.

```bash
# con il venv ATTIVO (source .venv-nnunet/bin/activate)
pip install gdown
cd pandx_liu
python -c "import gdown; gdown.download_folder(
    'https://drive.google.com/drive/folders/1RpbofQDrQNzwfYjFhQYRRWCN8HhIoZQP',
    output='models')"
```

> ⚠️ **Google Drive a volte rifiuta i download automatici** (limiti di traffico giornalieri).
> Se `gdown` fallisce, apri il link nel browser, scarica la cartella a mano e mettila in
> `pandx_liu/models/` mantenendo la struttura `workspace/nnUNet_results/Dataset...`.
> Il notebook cerca i pesi sia con quel prefisso sia senza.

Struttura attesa alla fine:

```
pandx_liu/models/workspace/nnUNet_results/
├── Dataset103_PANORAMA_baseline_Pancreas_Segmentation/     ← stadio 1 (del baseline)
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
│       ├── plans.json, dataset.json
│       └── fold_0 … fold_4/checkpoint_final.pth
└── Dataset107_PDAC_Detection/                              ← stadio 2 (di Liu)
    └── nnUNetTrainerCELossLesionSplit__nnUNetPlans_v3__3d_fullres/
        ├── plans.json, dataset.json
        └── fold_0 … fold_4/checkpoint_final.pth
```

> Lo stadio 1 è **lo stesso modello** del baseline: se hai già `baseline_pdac/models`,
> quei pesi sono identici. Sono duplicati qui apposta, per rendere la cartella autonoma.

---

## 5 · Organizza i dati

```
CARTELLA_DATI/
├── PDAC/
│   ├── images/     ← TC dei pazienti con tumore    (.nii.gz)
│   └── labels/     ← segmentazioni manuali del tumore (opzionale, serve per l'AP)
└── health/
    └── images/     ← TC dei controlli               (.nii.gz)
```

Regole:
- devono essere **TC con mezzo di contrasto in fase porto-venosa** — è l'unica fase
  su cui i modelli sono addestrati;
- i nomi devono corrispondere tra `images/` e `labels/`; l'eventuale suffisso `_0000`
  viene tolto in automatico, quindi `001_0000.nii.gz` si accoppia con `001.nii.gz`.

---

## 6 · Esegui il notebook

1. Apri `pandx_inference.ipynb` in Cursor / VS Code.
2. Scegli il kernel **PANORAMA baseline (.venv-nnunet)**.
3. Nella **cella 1** metti i percorsi delle tue cartelle.
4. Esegui le celle **in ordine**. Le prime tre sono controlli: se segnalano un
   problema, fermati e risolvilo lì.
5. Il **punto 5** elabora un caso solo: usalo per misurare il tempo.
6. Il **punto 6** elabora tutto e scrive `punteggi.csv`.
7. Il **punto 7** calcola AUROC e AP con grafici.

### Prima prova consigliata

Nella cella 1 imposta `FAST_MODE = True`: usa 1 fold invece di 5 e disattiva la TTA,
quindi è circa **40 volte più veloce**. I risultati **non** sono quelli ufficiali:
per i numeri veri rimetti `FAST_MODE = False`.

---

## 7 · Problemi frequenti

| Errore | Causa | Rimedio |
|---|---|---|
| `Unable to locate trainer class nnUNetTrainerCELossLesionSplit` | manca il trainer di Liu | rifai il **punto 3** |
| `gdown` scarica 0 file o dà "quota exceeded" | limite di traffico di Google Drive | scarica a mano dal browser (punto 4) |
| `A module compiled using NumPy 1.x cannot be run in NumPy 2.x` | numpy aggiornato per sbaglio | `pip install "numpy==1.26.4"` (solo macOS Intel) |
| `UnpicklingError ... numpy.core.multiarray.scalar` | torch ≥ 2.6 | installa `torch==2.5.1` |
| `Detected old nnU-Net plans format` | **normale**, ignoralo | — |
| Il kernel non compare in Cursor | kernel non registrato | rifai `ipykernel install`, poi ricarica |

---

## 8 · Cosa aspettarsi (e come non ingannarsi)

- **Tempi:** GPU 3–8 min per caso; CPU 1–4 h per caso (più lento del baseline).
- **Non esiste un benchmark out-of-fold per PanDx**: Liu non ha pubblicato la sua
  suddivisione in fold, quindi far girare il modello sui casi pubblici PANORAMA dà
  punteggi **gonfiati** (li ha già visti in addestramento). L'unica valutazione onesta
  è su dati esterni. Il notebook lo spiega al punto 8.
- **Attenzione a quali numeri confronti.** Circolano due coppie di valori per PanDx,
  misurate su coorti diverse:
  - **AUROC 0,916 / AP 0,720** — figura S8 dell'appendice di Alves et al., *Lancet Oncol*
    2026 (coorte di test sequestrata, 1130 casi; il baseline fa 0,915 / 0,634);
  - **AUROC 0,9263 / AP 0,7243** — articolo PanDx (arXiv 2503.10068), test set ufficiale
    della challenge, 957 casi.

  Sono coorti **diverse**: non vanno mescolate. In entrambe il 41% dei controlli aveva
  altre patologie pancreatiche. Se i tuoi controlli sono soggetti sani, il problema è
  più facile e i tuoi numeri **non sono confrontabili** con nessuna delle due.
- **Pochi casi = intervalli larghi:** con qualche decina di esami l'IC 95% sull'AUROC
  è di circa ±0,10. Riportalo sempre.

---

## 9 · Riferimenti

- Codice PanDx: <https://github.com/han-liu/PDAC_Detection>
  (esiste anche <https://github.com/han-liu/PanDx>, stesso contenuto)
- Articolo: Liu H, Gao R, Krieg E, Grbic S. *PanDx: AI-Assisted Early Detection of PDAC
  on Contrast-Enhanced CT*, MICCAI Workshop on Applications of Medical AI, 2025;
  pp. 63–71 ([arXiv 2503.10068](https://arxiv.org/abs/2503.10068))
- Challenge: Alves N, Schuurmans M, Rutkowski D, et al. *Lancet Oncol* 2026;27(1):116–124
- Pesi dello stadio 1 (baseline): <https://zenodo.org/records/11160381> (CC BY-NC 4.0)
- nnU-Net: Isensee F, et al. *Nat Methods* 2021;18:203–211
