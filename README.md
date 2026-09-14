# iqra-ar-letters

An interactive **Streamlit** app that helps kids learn to write Arabic letters step by step: the child sees a simple illustration and a word with one or more missing letters, draws the missing letter(s) by hand on a canvas, and gets instant feedback from a handwriting-classification model trained on Arabic letters.

---

## Overview

| | |
|---|---|
| **Target audience** | Kids learning the Arabic alphabet |
| **Core idea** | Complete a word by drawing the missing letter instead of typing it |
| **Levels** | 3 progressive difficulty levels |
| **Stack** | Streamlit + TensorFlow/Keras + interactive canvas |
| **Interface language** | Fully Arabic UI (RTL) |

---

## Gameplay

The child sees a small illustration (drawn as inline SVG in the code itself — no internet dependency, no external image library) representing the word's meaning, followed by the word displayed as boxes: a filled box for each existing letter, and a dashed placeholder box for each missing letter. The child draws the missing letter on a dedicated canvas and taps **"تحقق" (Check)** to get instant, per-letter feedback.

### Levels

| Level | Word length | Missing letters |
|---|---|---|
| 1 | 3 letters | 1 |
| 2 | 4 letters | 2 |
| 3 | 5–6 letters | 3 |

Level progression happens **automatically** after a set number of fully-correct words (default: 3, configurable via `WORDS_TO_ADVANCE`).

---

## Features

- A structured word bank per level, easy to extend with new words.
- Inline SVG illustrations for every word (no emojis, no external image loading).
- Per-letter feedback instead of a single pass/fail result.
- Persistent progress bar, score, and level badges in the header.
- Fully custom, professional visual design (Tajawal font, consistent palette, correct RTL layout).
- Graceful handling when the model file is missing — the app never crashes.
- One-click full game reset from the UI.

---

## Project structure

```
.
|__ notebooks
├── app.py                      # Full application code
├── requirements.txt            # Python dependencies
├── arabic_letters_model.keras  # Model file (added manually, not included)
└── README.md
```

### Key parts of `app.py`

| Component | Purpose |
|---|---|
| `WORD_BANK` | Word list grouped by level |
| `WORD_ICONS` | Maps each word to its illustrative SVG |
| `load_model()` | Loads the Keras model with caching (`cache_resource`) |
| `new_round()` | Picks a random word and selects the missing letters |
| `predict_letter()` | Preprocesses the drawn image and runs it through the model |
| `register_result()` | Updates score and handles level-progression logic |
| `render_*()` | UI rendering functions (header, hint image, word, canvases, feedback) |

---

## Model requirements

The app expects a Keras model file named **`arabic_letters_model.keras`** with the following spec:

- **Input:** grayscale image, `32×32×1`, values normalized between 0 and 1, white strokes on a black background.
- **Output:** a softmax probability distribution over **28 classes**, in the same order as the `LETTERS` array in `app.py`:

  ```
  ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي
  ```

> If your model was trained with a different class order, update the `LETTERS` array to match your training order exactly — otherwise predictions will be wrong even when the drawn letter is correct.

---

## Running locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place `arabic_letters_model.keras` in the same folder as `app.py`.

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open the URL printed in the terminal (usually `http://localhost:8501`).

---

## Deployment

The project runs out of the box on **Streamlit Community Cloud** or any Streamlit-compatible platform (Render, Hugging Face Spaces, etc.):

1. Push `app.py`, `requirements.txt`, and `arabic_letters_model.keras` to the same repository.
2. Make sure the model file size fits your platform's limits.
3. Connect the repository to the platform and deploy — no extra configuration needed.

---

## Customization

- **Add new words:** add the word to `WORD_BANK` under the right level, and add a matching SVG in `WORD_ICONS` (same style as the rest: `#123B4F` strokes, `#D9A441` accents).
- **Change the words-to-advance threshold:** edit `WORDS_TO_ADVANCE`.
- **Add a 4th level:** extend `WORD_BANK`, `LEVEL_LABELS`, and `MAX_LEVEL`, and add longer words.
- **Change the visual identity:** all colors and fonts are defined centrally in `inject_css()`.

---

## Troubleshooting

| Issue | Likely cause | Fix |
|---|---|---|
| `File does not exist: app.py` | Running the command from the wrong folder | `cd` into the project folder before `streamlit run` |
| `ModuleNotFoundError` | Missing dependency | `pip install -r requirements.txt` |
| `image_data was not requested` | Newer version of `streamlit-drawable-canvas` | Make sure `return_image_data=True` is passed to `st_canvas` |
| All answers come back wrong | Canvas stroke/background colors don't match the model's training data, or `LETTERS` order doesn't match the model's class order | Make sure strokes are white on a black background, and that `LETTERS` matches your training order |

---

## License

This project is free to use and modify within your own work.
