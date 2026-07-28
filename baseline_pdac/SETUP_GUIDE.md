# Guida di installazione — PANORAMA baseline (inferenza PDAC)

Guida passo passo per far girare il notebook `panorama_baseline_inference.ipynb`
su una macchina nuova. Pensata anche per essere data in pasto a Cursor: ogni passo
è un comando da eseguire, con la spiegazione del perché.

- **Cosa fa il notebook:** prende TC addominali con contrasto (fase porto-venosa, `.nii.gz`)
  e per ogni caso restituisce un punteggio 0–1 di sospetto PDAC, poi calcola AUROC e AP.
- **Cosa NON è:** un dispositivo medico. È uno strumento di ricerca, licenza dei pesi
  **CC BY-NC 4.0** (solo uso non commerciale).

---

## 0 · Il vincolo da capire prima di iniziare

Un solo dettaglio determina tutta l'installazione:

> **PyTorch < 2.3 non funziona con NumPy ≥ 2.**

Su **Linux con GPU** non è un problema: si usa torch ≥ 2.3 e numpy 2.x, tutto liscio.

Su **macOS Intel** invece l'ultima versione di PyTorch disponibile è **2.2.2**
(dalla 2.3 in poi Apple/PyTorch non pubblicano più wheel per x86_64), quindi lì
**bisogna obbligatoriamente restare a numpy < 2**.

Se ignori questo vincolo ottieni l'errore:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Le due sezioni sotto tengono già conto di tutto questo.

---

## 1 · Installazione su Linux con GPU  (consigliata per l'uso vero)

### 1.1 Crea un ambiente dedicato

> **Perché dedicato e non il venv che usi già?** Perché nnU-Net impone versioni
> precise di numpy/scipy e potrebbe entrare in conflitto con altri pacchetti
> (per esempio TotalSegmentator, che a sua volta dipende da nnunetv2).
> Un venv separato costa 3 GB di disco e ti evita ore di debug.
> Se preferisci comunque usare quello esistente, salta al punto 1.2 e verifica
> a mano che `nnunetv2` non sia già presente con un'altra versione.

```bash
cd ~/percorso/del/progetto
python3.12 -m venv .venv-nnunet
source .venv-nnunet/bin/activate
pip install --upgrade pip
```

### 1.2 Installa i pacchetti

```bash
# PyTorch con CUDA — LA VERSIONE VA FISSATA (vedi avviso sotto)
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121

# tutto il resto
pip install -r requirements-linux-cuda.txt
```

> ⚠️ **Non scrivere `pip install torch` senza versione.** Dalla 2.6 `torch.load` usa
> `weights_only=True` come default e i checkpoint del baseline vengono **rifiutati**
> (`UnpicklingError: ... Unsupported class numpy.core.multiarray.scalar`).
> Gli indici CUDA danno versioni troppo recenti (cu124 → 2.6.0, cu118 → 2.7.1):
> con nnunetv2 2.5.1 serve **torch < 2.6**. Se la tua GPU richiede per forza
> torch ≥ 2.6, devi aggiornare anche nnunetv2.

Verifica che la GPU sia vista:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1.3 Registra il kernel Jupyter

```bash
python -m ipykernel install --user --name panorama-nnunet \
       --display-name "PANORAMA baseline (.venv-nnunet)"
```

Poi vai al **punto 3** (comune a entrambi i sistemi).

---

## 2 · Installazione su macOS Intel  (solo per sviluppo e test)

⚠️ Su CPU l'inferenza richiede **da 30 minuti a 2 ore per caso**. Va bene per
verificare che tutto funzioni, non per elaborare centinaia di esami.

```bash
cd /percorso/del/progetto      # la cartella che contiene baseline_pdac
python3.12 -m venv .venv-nnunet
source .venv-nnunet/bin/activate
pip install --upgrade pip

# ATTENZIONE: i vincoli vanno passati TUTTI INSIEME in un solo comando,
# altrimenti pip installa numpy 2.x e poi torch si rompe.
pip install "numpy==1.26.4" "torch==2.2.2" "scipy==1.13.1" \
            "tifffile==2024.8.30" "imagecodecs==2024.6.1" \
            "nnunetv2==2.5.1" report-guided-annotation picai_eval \
            ipykernel matplotlib pandas SimpleITK nibabel scikit-learn openpyxl

python -m ipykernel install --user --name panorama-nnunet \
       --display-name "PANORAMA baseline (.venv-nnunet)"
```

Controllo che sia andato tutto bene (non deve comparire nessun avviso su NumPy):

```bash
python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"
```

Atteso: `torch 2.2.2 | numpy 1.26.4`

---

## 3 · Il trainer personalizzato  (passo obbligatorio, spesso dimenticato)

I pesi del baseline sono stati addestrati con un trainer personalizzato chiamato
`nnUNetTrainer_Loss_CE_checkpoints`. nnU-Net lo cerca **dentro il proprio pacchetto**:
se non lo trova, il caricamento del modello fallisce con

```
RuntimeError: Unable to locate trainer class nnUNetTrainer_Loss_CE_checkpoints
```

Copialo al posto giusto (comando unico, funziona su Linux e macOS):

```bash
TDIR=$(python -c "import nnunetv2, os; print(os.path.join(nnunetv2.__path__[0],'training','nnUNetTrainer'))")
curl -fsSL https://raw.githubusercontent.com/DIAGNijmegen/PANORAMA_baseline/main/src/customTrainerCEcheckpoints.py \
     -o "$TDIR/customTrainerCEcheckpoints.py"

# verifica che sia il file giusto e non una pagina di errore
grep -q "class nnUNetTrainer_Loss_CE_checkpoints" "$TDIR/customTrainerCEcheckpoints.py" \
  && echo "OK, installato in: $TDIR" \
  || echo "ERRORE: download fallito, il file non contiene la classe attesa"
```

> Il file cambia solo la *loss* usata in addestramento — per l'inferenza è
> irrilevante, ma la classe deve esistere perché nnU-Net possa ricostruire la rete.
>
> **Nota:** se un giorno reinstalli o aggiorni `nnunetv2`, questo file va rimesso.

---

## 4 · Scarica i pesi (1,8 GB)

```bash
cd baseline_pdac          # la cartella di questo progetto
mkdir -p models && cd models

curl -L -o D103.zip "https://zenodo.org/records/11160381/files/Dataset103_PANORAMA_baseline_Pancreas_Segmentation.zip?download=1"
curl -L -o D104.zip "https://zenodo.org/records/11160381/files/Dataset104_PANORAMA_baseline_PDAC_Detection.zip?download=1"

unzip -q D103.zip && unzip -q D104.zip
rm D103.zip D104.zip     # opzionale, libera 1,8 GB
cd ..
```

Struttura attesa alla fine:

```
models/
├── Dataset103_PANORAMA_baseline_Pancreas_Segmentation/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
│       ├── plans.json, dataset.json
│       └── fold_0 … fold_4/checkpoint_final.pth
└── Dataset104_PANORAMA_baseline_PDAC_Detection/
    └── nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres/
        ├── plans.json, dataset.json
        └── fold_0 … fold_4/checkpoint_best_panorama.pth
```

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
- i nomi dei file devono **corrispondere** tra `images/` e `labels/` (es. `001.nii.gz`);
- devono essere **TC con mezzo di contrasto in fase porto-venosa** — è l'unica fase
  su cui i modelli sono addestrati;
- orientamento e spacing qualsiasi: ci pensa il ricampionamento.

---

## 6 · Esegui il notebook

1. Apri `panorama_baseline_inference.ipynb` in Cursor / VS Code.
2. In alto a destra scegli il kernel **PANORAMA baseline (.venv-nnunet)**.
3. Nella **cella 1** metti i percorsi delle tue cartelle.
4. Esegui le celle **in ordine**. Le prime tre sono controlli: se segnalano un
   problema, fermati e risolvilo lì.
5. Il **punto 5** elabora un caso solo: usalo per misurare il tempo e stimare il lotto.
6. Il **punto 6** elabora tutto e scrive `punteggi.csv`.
7. Il **punto 7** calcola AUROC e AP con grafici.

### Prima prova consigliata

Nella cella 1 imposta `FAST_MODE = True`: usa 1 fold invece di 5 e disattiva la TTA,
quindi è circa **40 volte più veloce**. Serve solo a verificare che la catena funzioni.
I risultati **non** sono quelli ufficiali: per i numeri veri rimetti `FAST_MODE = False`.

---

## 7 · Problemi frequenti

| Errore | Causa | Rimedio |
|---|---|---|
| `Unable to locate trainer class nnUNetTrainer_Loss_CE_checkpoints` | manca il trainer personalizzato | rifai il **punto 3** |
| `A module compiled using NumPy 1.x cannot be run in NumPy 2.x` | numpy aggiornato per sbaglio | `pip install "numpy==1.26.4"` (solo su macOS Intel) |
| `A NumPy version >=2.0.0 is required for this version of SciPy` | scipy troppo recente per numpy 1.x | `pip install "scipy==1.13.1"` |
| Il kernel non compare in Cursor | kernel non registrato | rifai `ipykernel install`, poi ricarica la finestra |
| `Lo stadio 1 non ha trovato il pancreas` | immagine senza addome, o non contrastografica | verifica l'esame |
| CUDA out of memory | GPU piena | usa `FOLDS = (0,)`, oppure chiudi altri processi |
| Lentissimo | stai girando su CPU | normale: usa la macchina con GPU |

---

## 8 · Cosa aspettarsi (e come non ingannarsi)

- **Tempi:** GPU 1–3 min per caso; CPU 30 min – 2 h per caso.
- **Distorsione ottimistica:** se valuti sui 2238 casi pubblici di PANORAMA, il modello
  li ha già visti in addestramento → i punteggi sono gonfiati. Per una valutazione onesta
  servono casi nuovi, mai visti.
- **Confronto con l'articolo:** l'AUROC di 0,92 pubblicato è misurato su una coorte in cui
  il **41% dei controlli aveva altre patologie pancreatiche** (IPMN, pancreatite, tumori
  neuroendocrini). Se i tuoi controlli sono soggetti sani, il problema è più facile e il
  tuo AUROC risulterà più alto, ma **non è confrontabile**.
- **Pochi casi = intervalli larghi:** con qualche decina di esami l'IC 95% sull'AUROC è
  di circa ±0,10. Riportalo sempre.

---

## 9 · Riferimenti

- Codice baseline: <https://github.com/DIAGNijmegen/PANORAMA_baseline> (Apache-2.0)
- Pesi: <https://zenodo.org/records/11160381> (CC BY-NC 4.0)
- Articolo: Alves N, Schuurmans M, Rutkowski D, et al. *Lancet Oncol* 2026;27(1):116–124
- nnU-Net: Isensee F, et al. *Nat Methods* 2021;18:203–211
