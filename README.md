# Beyond Goals and Assists – Attacker Scouting App

Streamlit app for the MSc Data Analytics project *"Beyond Goals and Assists: Cross-Tournament
Evaluation of Player Action Value in International Football Using StatsBomb Event Data"*.

- **Goal:** help clubs with smaller budgets find undervalued attacking players.
- **Method:** VAEP (Decroos et al., KDD 2019) with one XGBoost model (two outputs: P(score), P(concede)).
- **Training:** FIFA World Cup 2022. **Unseen tests:** UEFA Euro 2024, Copa América 2024.
- **Pipeline:** Kaggle notebook (Steps 1–8a) produces the files in `app_data/`.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
| File | Content |
|---|---|
| `app.py` | the Streamlit app |
| `app_data/*_attacker_rankings.csv` | attacker rankings per tournament (VAEP/90, 95% ranges, hidden talent) |
| `app_data/*_attacker_actions.csv.gz` | every action of the ranked attackers with model probabilities and VAEP |
| `app_data/xgb_scores.json`, `xgb_concedes.json` | the trained XGBoost model |
| `app_data/test_results.csv`, `calibration_test.png` | out-of-sample metrics (ROC-AUC, PR-AUC, Brier, ECE) |

## Data
StatsBomb Open Data – https://github.com/statsbomb/open-data. Used under the StatsBomb open data
licence; data provided by StatsBomb (Hudl). For research and scouting support only – no betting use.
