# iqra-ar-letters

🔗 **جرّب التطبيق مباشرة:** [تعلم كتابة الحروف العربية · Streamlit](https://iqra-ar-letters.streamlit.app/)

An interactive Streamlit app that helps kids learn to write Arabic letters step by step: the child sees a simple illustration and a word with one or more missing letters, draws the missing letter(s) by hand on a canvas, and gets instant feedback from a handwriting-classification model trained on Arabic letters.

## Overview

| | |
|---|---|
| **Target audience** | Kids learning the Arabic alphabet |
| **Core idea** | Complete a word by drawing the missing letter instead of typing it |
| **Levels** | 3 progressive difficulty levels |
| **Stack** | Streamlit + TensorFlow/Keras + interactive canvas |
| **Interface language** | Fully Arabic UI (RTL) |

## Gameplay

The child sees a small illustration (drawn as inline SVG in the code itself — no internet dependency, no external image library) representing the word's meaning, followed by the word displayed as boxes: a filled box for each existing letter, and a dashed placeholder box for each missing letter. The child draws the missing letter on a dedicated canvas and taps **"تحقق"** (Check) to get instant, per-letter feedback.

## Levels

| Level | Word length | Missing letters |
|---|---|---|
| 1 | 3 letters | 1 |
| 2 | 4 letters | 2 |
| 3 | 5–6 letters | 3 |

Level progression happens automatically after a set number of fully-correct words (default: 3, configurable via `WORDS_TO_ADVANCE`).

## Features

- A structured word bank per level, easy to extend with new words.
- Inline SVG illustrations for every word (no emojis, no external image loading).
- Per-letter feedback instead of a single pass/fail result.
- Persistent progress bar, score, and level badges in the header.
- Fully custom, professional visual design (Tajawal font, consistent palette, correct RTL layout).
- Graceful handling when the model file is missing — the app never crashes.
- One-click full game reset from the UI.

## Project structure
