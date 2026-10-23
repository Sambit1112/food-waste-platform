# SurplusLink — AI-Powered Food Waste Management Platform

Predicts surplus food from donors and optimally matches it to nearby
recipients before it spoils, using ML forecasting + combinatorial
optimization instead of manual coordination.

## What's inside

| Module | Approach | File |
|---|---|---|
| Surplus prediction | RandomForestRegressor on historical sales (day-of-week, category, event flag) | `models/predictor.py` |
| Image classification (CNN) | Multi-task PyTorch CNN: food category + portion tier from a donor photo | `models/food_classifier.py` |
| Redistribution optimization | Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on a cost matrix combining distance, urgency, and capacity fit | `models/matcher.py` |
| Backend / API | Flask + SQLite | `app.py`, `database.py` |
| Frontend | Live map (Leaflet), photo-upload form, listing feed, impact dashboard (Chart.js) | `templates/`, `static/` |

## The CNN module, honestly

`models/food_classifier.py` is a real, trained multi-task CNN (shared conv backbone, two heads) — not a mock. It predicts:

1. **Food category** (cooked / produce / bakery / dairy / packaged) — **94% validation accuracy**
2. **Portion tier** (small / medium / large) — **~40-45% validation accuracy**, mapped to a suggested kg range the donor confirms/edits rather than a forced exact number

Two things worth stating plainly in your report/viva:

- **Training data is synthetic, not real food photos.** This sandboxed environment has no internet access to a dataset like Food-101 and no camera to collect real ones, so `models/generate_food_images.py` generates a proxy dataset — each category gets a distinct color/texture signature, each image also encodes a portion tier as pixel coverage. This proves the full pipeline (upload → tensor → CNN → category + portion → autofill) genuinely works end to end, the same way `seed_data.py` stands in for a real POS export. **Before using this for anything beyond a demo, retrain on real donor photos**: replace the generator's output with a real `ImageFolder`-style directory (`data/food_images/<category>/*.jpg`) and rerun `python models/food_classifier.py` — the architecture, training loop, and API don't change.
- **Portion tier accuracy is deliberately modest, and that's the honest result.** Estimating quantity/weight from a single 2D photo with no depth cue or reference object is a genuinely hard, active CV research problem — that's why the model outputs a *tier* + range for the donor to adjust, not a claimed exact kg. If your report needs stronger portion accuracy, the real fix is a reference-object-in-frame technique or a depth-aware model, not more synthetic data.

## Why these choices (for your report/viva)

- **RandomForest over deep learning for prediction**: the feature set (donor, day-of-week, category, event flag) is small and tabular — a tree ensemble outperforms a neural net here with far less data and is easy to explain via feature importance.
- **Hungarian algorithm over greedy nearest-neighbor for matching**: greedy matching can strand a donor with the closest recipient even if that recipient was a better fit for a *more urgent* listing elsewhere. The Hungarian algorithm solves the assignment globally and optimally in polynomial time (O(n³)), which is easy to defend as "provably optimal" rather than "a heuristic that usually works."
- **Cost function, not pure distance**: cost = distance − (urgency × weight) + capacity_penalty. This means a listing about to expire will "outbid" a farther recipient over a nearer one that's a worse fit, which is the actual real-world priority (don't let food expire waiting for the closest match).

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or plain pip install in a venv
python database.py             # creates the schema
python seed_data.py             # loads 12 donors, 8 recipients, 180 days of sales history, 10 live listings
python models/predictor.py       # trains the surplus prediction model (~2s)
python models/generate_food_images.py  # builds the synthetic photo training set (~5s)
python models/food_classifier.py       # trains the CNN (~1-2 min on CPU)
python app.py                          # starts the server at http://localhost:5000
```

Open `http://localhost:5000` for the dispatch view (map, live listings, post surplus, run optimizer).
Open `http://localhost:5000/dashboard` for the impact view (kg diverted, meals saved, CO2 offset).

## API reference

- `GET/POST /api/donors` — list or register donors
- `GET/POST /api/recipients` — list or register recipients
- `GET/POST /api/listings` — list available surplus, or post new surplus (auto-runs prediction on POST)
- `POST /api/classify` — multipart image upload → `{food_category, category_confidence, portion_tier, portion_confidence, suggested_kg_range}`
- `GET /api/predict?donor_id=&food_category=&day_of_week=` — standalone prediction query
- `POST /api/match` — runs the Hungarian-algorithm optimizer over all available listings and recipients, persists matches
- `GET /api/matches` — match history
- `GET /api/stats` — aggregate impact stats for the dashboard

## Extending this for the capstone report

- **Dataset**: swap `seed_data.py`'s synthetic generator for a real POS/inventory export if your college dining hall or a partner restaurant will share data — the prediction pipeline doesn't change, just the source table.
- **Real food photos for the CNN**: replace `generate_food_images.py`'s output with real donor photos (Food-101 is a good public starting point) in the same folder layout, then retrain — see "The CNN module, honestly" above.
- **Routing beyond assignment**: current matching is 1-to-1 per optimization run. A v2 could batch multiple listings per pickup trip using a vehicle routing solver (e.g., OR-Tools) if you want a stronger "logistics optimization" claim in the report.
- **Real-Time SMS & Push Notification Engine**:
  - **In-Browser Simulated Phone (Default)**: Click the **"📱 SMS Phone"** button in the top navigation bar to open the virtual smartphone mockup. All coordination SMS messages, volunteer driver mission alerts, and 4-digit handover OTPs appear here live in real-time.
  - **Physical Mobile Handset Delivery**: To send actual SMS messages to physical mobile phones (e.g. `+91 8249756350`), create/edit your `.env` file in the project root:
    - **Option A (India +91 - Fast2SMS)**: Sign up for free at [Fast2SMS](https://www.fast2sms.com) and add `FAST2SMS_API_KEY=your_key_here`.
    - **Option B (Global - Twilio)**: Add `TWILIO_ACCOUNT_SID=...`, `TWILIO_AUTH_TOKEN=...`, and `TWILIO_PHONE_NUMBER=...`.
    - Restart `python app.py`. Messages will deliver over cellular networks to your phone!

## Project structure

```
food-waste-platform/
├── app.py                  # Flask app + API routes
├── database.py              # SQLite schema
├── seed_data.py              # synthetic demo data generator
├── models/
│   ├── predictor.py            # surplus prediction (RandomForest)
│   ├── matcher.py               # matching optimization (Hungarian algorithm)
│   ├── generate_food_images.py  # synthetic training-photo generator
│   ├── food_classifier.py        # CNN: category + portion-tier classification
│   └── food_cnn.pt                # trained CNN weights (created on train)
├── templates/
│   ├── index.html             # dispatch/operations view (incl. photo upload)
│   └── dashboard.html          # impact dashboard
├── static/
│   ├── css/style.css
│   └── js/{app.js, dashboard.js}
├── data/
│   ├── foodwaste.db              # created on init
│   └── synthetic_food_images/     # created by generate_food_images.py
└── requirements.txt
```
