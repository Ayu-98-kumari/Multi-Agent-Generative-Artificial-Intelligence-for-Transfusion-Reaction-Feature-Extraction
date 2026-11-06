# input_output

The working folder for the three-model comparison
(`model_evaluation_pipeline/run_models.sh`). Put **one** input CSV here, run one
command, and every result is written back into this same folder.

## 1. What the input file must look like

- **Format:** a `.csv` file (UTF-8), any name, e.g. `my_notes.csv`.
- **Header:** `Clinical Text` in **column A** (capitalisation and surrounding
  spaces don't matter).
- **One clinical note per row**, in column A, below the header. A note can be
  long and contain line breaks -- it just has to sit inside one quoted CSV cell
  (Excel, Google Sheets, or `pandas.to_csv` all do this for you).
- **No blank rows in between.** Reading stops at the first empty cell in
  column A, so any note below a blank row is skipped.
- Any other columns are ignored.
- **Only one input CSV at a time.** If there are several, the script lists them
  and asks you to pass the one you want.

Minimal example (`my_notes.csv`):

```
Clinical Text
"Patient developed a fever of 39.1 C and rigors 30 minutes into a red cell transfusion..."
"Shortly after a platelet transfusion the patient reported hives and itching..."
```

## 2. Run it

From the repository root (dependencies installed and API keys in `.env` -- see
the main README):

```bash
bash model_evaluation_pipeline/run_models.sh
```

## 3. What appears here

Named after your input file (here `my_notes`):

| file | contents |
|---|---|
| `my_notes_output_qwen.csv` | qwen3.6-27b results |
| `my_notes_output_gpt120b.csv` | gpt-oss-120b results |
| `my_notes_output_kimi.csv` | Kimi K3 results |
| `my_notes_model_comparison.csv` | all three side by side, one row per note |
| `my_notes.dropped.log`, `.absences.log`, `.schema.log` | diagnostic logs (from the last model run) |
| `.cache_qwen/`, `.cache_gpt120b/`, `.cache_kimi/` | hidden per-model caches (see below) |

Each per-model CSV has four columns: `Clinical Text`, then `Extracted Features`,
`Treatments` and `Past Medical History` as JSON.

## 4. If a run stops part-way

Run the same command again. Notes that already finished are read from the
`.cache_*` folders instead of being sent to the models again, so you only pay for
what's left. To ignore the caches and start completely fresh:

```bash
bash model_evaluation_pipeline/run_models.sh --fresh
```

To check your setup on a single note first (cheap), add `-n 2` -- it scores only
spreadsheet row 2, the first note under the header.

## Privacy

Nothing you put in this folder is ever committed to git -- your input, the
outputs and the caches stay on your machine (see `.gitignore`). Only this README
and the `.gitignore` are tracked, so the folder ships empty.
