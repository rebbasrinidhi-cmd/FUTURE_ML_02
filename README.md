# FUTURE_ML_02: Support Ticket Classification

Future Interns · Machine Learning Track · Task 2

A system that reads a customer support ticket and returns its **category** (which team should handle it) and its **priority** (High / Medium / Low), tuned so urgent tickets are rarely missed.

## The business problem

A support team gets hundreds of tickets a day. Agents lose time reading each one, deciding who should handle it, and guessing how urgent it is. This project automates that first step and shows how much of it can safely run without a human.

## What is in this repo

| File | Purpose |
|---|---|
| `Support_Ticket_Classification.ipynb` | Full workflow: data, cleaning, EDA, category models, priority logic, error analysis, insights, triage function |
| `images/` | Charts used in the notebook (mix, volume, model comparison, confusion matrices, priority policies, confidence, top words) |
| `data/support_tickets_simulated.csv` | The simulated raw ticket history (messy on purpose) |
| `outputs/test_predictions.csv` | Test-set predictions with category confidence and P(High) |
| `outputs/category_model_comparison.csv` | Cross-validated scores for all five category models |
| `outputs/priority_policy_comparison.csv` | Four priority policies compared (cross-validation and test) |
| `outputs/*_classification_report.csv` | Per-class precision, recall and F1 |
| `outputs/confidence_routing_table.csv` | Auto-routing coverage vs accuracy at each confidence level |
| `outputs/routing_and_response_targets.csv` | Suggested team and response-time setup |
| `outputs/triage_demo.csv` | Output of `triage_ticket()` on six example tickets |
| `requirements.txt` | Python dependencies |

## Approach

1. **Data cleaning**: duplicates, empty bodies, HTML leftovers, email signatures and inconsistent category spellings are found, counted and fixed (6,039 raw rows become 5,873 usable tickets).
2. **Text preparation**: subject and body are joined, lowercased, stripped of boilerplate, tokenized, stop-words removed (negations like "not" are kept) and lightly stemmed. Money amounts become a single token.
3. **Features**: TF-IDF on single words and word pairs, fitted inside each cross-validation fold to avoid leakage.
4. **Category models compared**: majority-class baseline, Complement Naive Bayes, Logistic Regression, Linear SVM, Random Forest. The best is tuned with a small grid, then scored once on the held-out test set.
5. **Priority policies compared**: keyword rules, ML, ML plus critical-keyword override, and ML plus a High-recall threshold chosen on cross-validation (catch at least 95% of High tickets).
6. **Error analysis**: top category mix-ups, accuracy by ticket length, most confidently wrong tickets.
7. **Confidence routing**: how many tickets can be routed automatically at a chosen accuracy.
8. **`triage_ticket(subject, body)`**: returns category, confidence, priority, team, response target and a "needs human check" flag.

## Results (simulated data, held-out test set of 1,175 tickets)

**Category (5-fold cross-validation on training data)**

| Model | Accuracy % | Macro-F1 % |
|---|---|---|
| Linear SVM (best) | 95.8 | 95.6 |
| Logistic Regression | 94.9 | 94.7 |
| Complement Naive Bayes | 94.6 | 94.3 |
| Random Forest | 93.4 | 93.1 |
| Majority class (baseline) | 22.2 | 6.1 |

On the test set the tuned **Linear SVM** reaches **96.4% accuracy** and **96.1% macro-F1**.

**Priority (test set)**

| Policy | Accuracy % | Macro-F1 % | High precision % | High recall % |
|---|---|---|---|---|
| A. Keyword rules | 75.7 | 75.5 | 79.3 | 85.3 |
| B. ML (argmax) | 88.5 | 88.5 | 90.8 | 94.6 |
| C. ML + critical-keyword override | 87.3 | 87.3 | 86.8 | 95.2 |
| D. ML + High-recall threshold | 88.3 | 88.2 | 90.5 | 94.3 |

The keyword rules alone catch 85% of High tickets at 79% precision; the ML policies improve both by a wide margin. Plain ML (policy B) and the threshold policy (D) perform almost the same here. The threshold's value is that it is an adjustable dial between catching more urgent tickets and raising fewer false alarms.

**Auto-routing:** at model confidence of 0.80 or higher, 90% of tickets can be routed automatically at 97.5% accuracy. The remaining 10% go to a quick human check.

**Business insights from the data**

- Technical Issue is the largest queue and also the most urgent (about 4 in 10 tickets are High priority). General Inquiry is the calmest (about 4% High).
- The outage week (week ending 2025-09-21) brought 243 tickets against a typical 109, mostly High-priority Technical tickets. A surge of High Technical tickets is a strong signal to page the on-call engineer.
- The most common mix-ups are General Inquiry tickets read as Technical, and Returns tickets read as Shipping or Billing, because those categories share vocabulary.

## Important note on the data

The notebook runs on a **simulated** ticket history built from templates, so it works with no downloads. Because of that, absolute scores are optimistic. About 3% of category labels were deliberately flipped, which caps category accuracy near 97%, and priority labels come from the simulation rather than from real agents. On real tickets expect lower scores and re-check the model leaderboard. Team names and response times in the routing table are illustrative suggestions.

## Using a real dataset

Set `DATA_PATH` and the column names in the configuration cell, for example:

- Kaggle *Customer Support Ticket Dataset*
- Any helpdesk export with a subject, a body, a category and a priority

The file needs a category column and a priority column (mapped to High / Medium / Low through `PRIORITY_MAP`). A `created_at` column is optional but needed for the weekly-volume chart. Everything downstream runs unchanged.

## Notes on tools

Text cleaning, stop-word removal and stemming are implemented directly with `re` and scikit-learn, so no NLTK or spaCy downloads are needed. Swapping in an NLTK or spaCy lemmatizer is a one-function change (`clean_text`).

## Run it

```
pip install -r requirements.txt
jupyter notebook Support_Ticket_Classification.ipynb
```
