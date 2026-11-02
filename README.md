# Neuroninių tinklų modelių palyginimas atpažįstant emocijas iš teksto

VilniusTECH kursinis darbas, dalykas „Neuroniniai tinklai“ (FMITM24102), 2026 m.

**Autoriai:**
- Edgaras Legus
- Teodora Einorytė

Dešimt neuroninių tinklų modelių (logistinė regresija, MLP, Embedding, Conv1D, transformeris, SimpleRNN, LSTM, GRU,
BiLSTM, BiLSTM su dėmesio mechanizmu) palyginami atpažįstant šešias emocijas angliškuose tekstuose (dair-ai/emotion),
su DistilBERT kaip konkurentu, kitame rinkinyje (GoEmotions) ir lietuviškame vertime. Išsamus planas ir kiekvieno
žingsnio būsena – [`PLAN.md`](PLAN.md).

## Diegimas

Python 3.13, skaičiuota be vaizdo plokštės (CPU):

```
python -m venv .venv
.venv\Scripts\activate          # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Windows'e TensorFlow gali nepavykti įdiegti, jei aplinkos kelias per ilgas (260 simbolių riba); tada aplinką
kurkite trumpame kelyje, pvz. `python -m venv C:\venvs\nt`.

Duomenys (`data/*.csv`) yra repozitorijoje. GloVe vektorius (S12, ~330 MB) atsisiunčia `python download_glove.py`.

## Emocijų atpažinimo įrankis

```
python app.py                                   # naršyklėje http://localhost:7860
python predict.py "I feel so alone tonight"     # komandinė eilutė; --lang lt – lietuviškas modelis
```

Įrankio modeliai (angliškas BiLSTM su dėmesio mechanizmu, lietuviškas Conv1D) yra `results/models/app/`.

## Eksperimentai ir rezultatai

| Komanda | Ką daro |
|---|---|
| `python train.py --model gru --seed 1 --dev` | vienas greitas bandomasis paleidimas |
| `python run_study.py S1` | visi vieno tyrimo paleidimai (S1–S5, S9a, S9–S12, S15–S17, T0a, T0b) |
| `python competitor.py --seed 1` | DistilBERT (S14; `--study S14b` – daugiakalbis) |
| `python compare.py` | lentelės ir paveikslai ataskaitos 4.2–4.7, 4.10 skyriams |
| `python generalization.py` | lentelės ir paveikslai 4.8 ir 4.9 skyriams |
| `python timing.py` | prognozavimo trukmė (S8) |

Visi rezultatai – `results/`: `results.csv` (eilutė kiekvienam paleidimui), `predictions/` (kiekvieno paleidimo
prognozės), `tables/` ir `figures/` (ataskaitos lentelės ir paveikslai). Išmokyti modeliai, išskyrus įrankio, į
repozitoriją nekeliami; juos atkuria `run_study.py`.
