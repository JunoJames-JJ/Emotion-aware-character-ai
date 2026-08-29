# Real-Time Multimodal Emotion Model

A small AI system that listens to a line of dialogue (text + audio), figures out the emotion behind it, and replies in a way that matches that emotion.

Built for the **ML Challenge: Real-Time Multimodal ML Challenge**, using the **MELD dataset** (Multimodal EmotionLines Dataset).

---

## What this project does, in plain words

Imagine a character robot listening to someone talk. It doesn't just hear the *words* — it also picks up on *how* something was said. "I'm fine." said flatly vs. said sharply means two different things.

This system:
1. Takes one line of speech at a time (the text of what was said + the audio of how it was said)
2. Guesses the emotion behind it (one of 7: neutral, joy, sadness, anger, surprise, fear, disgust)
3. Writes a short, natural reply that reacts to that emotion
4. Returns everything as clean, structured data — ready for a robot, app, or any system to use

---

## Track chosen: Text + Audio

MELD provides text, audio, and video. We chose **Text + Audio** over Text + Vision because MELD already provides audio per line in an easy-to-use form, while video would require extra work (extracting faces frame-by-frame) that wasn't worth the time trade-off for this challenge's timebox.

---

## How it works (architecture)

![Architecture diagram](architecture.png)

**Text version of the same diagram, for quick reference:**

```
                 ┌─────────────────────────┐
                 │   One line of dialogue    │
                 │   (text + audio)          │
                 │   fed in, one at a time    │
                 └────────────┬───────────────┘
                              │
              ┌───────────────┴────────────────┐
              ▼                                  ▼
   ┌────────────────────┐            ┌────────────────────────┐
   │   Text encoder        │            │   Audio encoder          │
   │   DistilBERT           │            │   Wav2Vec2-base           │
   │   (reads the words)    │            │   (reads the sound)       │
   └───────────┬────────────┘            └────────────┬─────────────┘
              │                                        │
              └───────────────┬────────────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │   Fusion + classifier      │
                 │   (our own small model)     │
                 │   Guesses the emotion       │
                 └────────────┬───────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │   Response generator        │
                 │   Qwen2.5-1.5B-Instruct     │
                 │   Writes a short reply       │
                 └────────────┬───────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │   Structured JSON output    │
                 │   emotion, response,        │
                 │   confidence, latency        │
                 └─────────────────────────┘
```

**In simple terms:** the text and audio each get "read" separately by their own AI model, their understanding is combined into one summary, a small classifier we built and trained guesses the emotion from that summary, and then a small language model writes a reply based on the emotion it detected.

---

## Tech stack

| Purpose | Tool used | Why |
|---|---|---|
| Reading text | DistilBERT | Small, fast, well-tested text model |
| Reading audio | Wav2Vec2-base | Standard model for understanding speech |
| Guessing emotion | Our own small neural network | Combines text + audio understanding into one decision |
| Writing replies | Qwen2.5-1.5B-Instruct | Small language model, good at short natural replies |
| Dataset | MELD (via Hugging Face) | Required by the challenge; real TV dialogue with emotion labels |
| Where we built it | Google Colab (free GPU) | Fast enough to train without needing a personal GPU |
| Programming | Python, PyTorch, Hugging Face `transformers` | Standard tools for this kind of AI work |

---

## Defining "real-time" for this project

MELD is pre-recorded dialogue, not a live microphone feed. So we defined "real-time" as:

> **Feeding one utterance at a time, like a live conversation, and measuring how long each one takes to fully process — from raw input to final response.**

**Result: about 1 second per line on average** (1,060 ms), tested on a Tesla T4 GPU. This is a reasonable pace for turn-based conversation — like two people pausing briefly between talking, not a fast back-and-forth debate. Response time varied from about 0.5 to 2.2 seconds depending on how long the generated reply was.

---

## How well does it work? (honest results)

We tested the trained classifier on 522 examples it had never seen before.

**Overall accuracy: 56.5%** (random guessing among 7 emotions would only get ~14%, so the model is clearly learning real patterns)

**But accuracy isn't even across emotions** — here's the honest breakdown:

| Emotion | Accuracy | Why |
|---|---|---|
| Neutral | 93% | Very common in the data (1,012 training examples) |
| Anger | 59% | Reasonably common, fairly distinct |
| Joy | 16% | Fewer examples, easy to confuse with neutral |
| Surprise | 28% | Fewer examples |
| Sadness | 2% | Very few examples (166), often subtle in tone |
| Fear | 0% | Extremely rare (39 examples) — not enough to learn from |
| Disgust | 0% | Extremely rare (53 examples) — not enough to learn from |

**We noticed this imbalance and tried to fix it.** We attempted two versions of "weighted loss" (a technique that tells the model to pay more attention to rare emotions). Neither fully solved the problem — fear and disgust stayed at 0% no matter what weighting we tried. This told us the real issue isn't the training method, it's that there simply isn't enough example data for those two emotions for a small model to learn reliable patterns from. We're reporting this honestly rather than hiding it — it's a genuine, evidence-backed limitation of using a small model on limited rare-class data within a short timebox.

---

## Resource requirements

**Total parameters across the entire system: 1.705 billion**
(The challenge allows up to 6 billion — we're using about 28% of the allowed budget.)

| Component | Parameters |
|---|---|
| Text encoder (DistilBERT) | 66,362,880 |
| Audio encoder (Wav2Vec2-base) | 94,371,712 |
| Our fusion + classifier | 395,271 |
| Response generator (Qwen2.5-1.5B) | 1,543,714,304 |
| **Total** | **1,704,844,167** |

**Hardware used:** Tesla T4 GPU (available free on Google Colab)
**Peak GPU memory used:** 10.68 GB (out of 15 GB available on a T4)

---

## Example output

Here's one real input traveling through the full system, start to finish:

```json
{
  "text": "Chandler what do you say?",
  "true_emotion": "neutral",
  "predicted_emotion": "neutral",
  "confidence": 0.869,
  "response_text": "Oh, that's Chandler? Sounds friendly enough! What's up?",
  "latency_ms": 1109.1
}
```

More examples are saved in `outputs/sample_run.json`.

---

## What we completed

- Full end-to-end pipeline: text + audio in → detected emotion + natural reply out
- Trained and evaluated an emotion classifier on real MELD data
- Investigated and documented a real class-imbalance problem, including two honest attempts to fix it
- Defined and measured real-time behavior with actual latency numbers
- Verified the system stays well within the 6 billion parameter limit
- Reported real hardware and memory usage

## What we intentionally left out

- **The optional standout extension** (combining vision + voice + text together, or using reinforcement learning) — we chose to fully finish and document the required core task well rather than stretch into unfinished extra scope, in line with the challenge's own guidance to prefer a small, correct, well-explained prototype.
- **True continuous audio streaming** — we process one complete utterance at a time rather than a continuous live audio stream, which is a simpler and more honest scope given the timebox.
- **Fixing the fear/disgust 0% accuracy** — we tried two approaches and documented why neither fully worked; a real fix would likely require more training examples for these rare emotions, which is outside what we could gather in this timebox.

---

## How to run this yourself

1. Install requirements: `pip install -r requirements.txt`
2. Make sure you have a Hugging Face account and access token (needed to download the dataset and models)
3. Run `scripts/run_demo.py` to see example utterances flow through the full pipeline
4. See `notebooks/training.ipynb` for the full training process, kept for transparency
