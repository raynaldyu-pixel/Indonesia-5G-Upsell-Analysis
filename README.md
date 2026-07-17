# 📡 The Booster Signal — Interactive 5G Upsell Analysis

**Live app:** [indonesian-5g-upsell-analysis.streamlit.app](https://indonesian-5g-upsell-analysis.streamlit.app/)

An interactive companion to the *5G Upsell Business Proposal — Indonesian Market
Edition*. Every phase of the pipeline — EDA, Rupiah ROI, decile lift,
e-wallet go-to-market, and campaign export — is a live, adjustable dashboard built
around a single fixed Indonesian subscriber dataset and a model trained in Colab.

There is **no upload UI**. The dataset and model are baked into the deployment —
this app always analyzes the same Indonesian prepaid base with the same
Colab-trained model, so there's nothing to re-attach on every visit.

## What's inside

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `data/Client.csv` | Subscriber profile data |
| `data/Record.csv` | Usage record data |
| `model/model.pkl` | Colab-trained XGBoost model |

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

## Setup — one-time only

The app reads from **fixed paths**, relative to `app.py`:

```
data/Client.csv
data/Record.csv
model/model.pkl
```

1. Create the `data/` and `model/` folders next to `app.py`.
2. Drop your two Indonesian subscriber CSVs into `data/`, merge-able on `Customer_ID`,
   with these columns:

   ```
   Customer_ID, churn, hnd_webcap, totmou, totrev, avgmou, avgrev, rev_Mean,
   ovrmou_Mean, ovrrev_Mean, change_mou, change_rev, eqpdays, months,
   roam_Mean, drop_vce_Mean, blck_vce_Mean, custcare_Mean
   ```

3. Export your trained model from Colab and drop it into `model/model.pkl`:

   ```python
   import joblib
   joblib.dump(xgb_model, "model.pkl")   # sklearn XGBClassifier
   ```

   Native XGBoost format also works — save as `model/model.json` (or `.ubj`) and
   update `MODEL_PATH` in `app.py` to match.

4. If you ever need different filenames or a different folder layout, edit the three
   constants at the top of `app.py`:

   ```python
   CLIENT_CSV_PATH = APP_DIR / "data" / "Client.csv"
   RECORD_CSV_PATH = APP_DIR / "data" / "Record.csv"
   MODEL_PATH      = APP_DIR / "model" / "model.pkl"
   ```

That's it — no in-app training, no file pickers. Restart the app (or just edit the
files on disk) and it picks up changes automatically.

## Deploying to Streamlit Community Cloud

Since there's no upload step, the data and
model files need to actually live in the deployed repo:

1. Commit `app.py`, `requirements.txt`, `data/Client.csv`, `data/Record.csv`, and
   `model/model.pkl` to your GitHub repo.
   - If the data or model is sensitive, use a **private** repo — Streamlit Cloud
     deploys from private repos fine, it's just not public-readable.
   - Large files: GitHub blocks anything over 100 MB per file. If your CSVs or model
     are close to that, use [Git LFS](https://git-lfs.com/) or trim the dataset.
2. Go to [share.streamlit.io](https://share.streamlit.io/), connect the repo, and
   point it at `app.py`.
3. Streamlit Cloud installs from `requirements.txt` and boots the app — no secrets
   or extra config needed since everything is read from local files in the repo.

## What's interactive

- **Sidebar** — Rupiah financial baseline (ARPU 4G/5G, contract horizon, campaign
  cost, opt-out penalty) and the evaluation holdout % / random seed used to score
  the model.
- **Model tab** — scores the loaded model against a fresh slice of the data; shows
  AUC/precision/recall/F1, feature importance, confusion matrix, and full
  classification report. The model itself is never retrained in-app.
- **Financial ROI tab** — drag the classification threshold and campaign-reach
  slider to watch precision/recall, the Rupiah cost matrix, and CapEx-recovery ROI
  recompute instantly.
- **Decile Lift tab** — switch between 5/10/20 bins.
- **Go-to-Market tab** — move the Tier 1/2/3 cutoffs to see campaign mix, e-wallet
  channel assignment, localised Bahasa Indonesia messages, and the A/B test
  simulation shift.
- **Export tab** — download the scored campaign targeting list as CSV.

## Notes

- Charts use the same navy/teal/blue/orange palette as the notebook and
  presentation deck for a consistent portfolio look.
- If either the data or the model file is missing at boot, the app shows exactly
  which path it expected and stops cleanly instead of falling back to anything
  synthetic.
