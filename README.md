# CraftGPT

A character-level GPT-style Transformer language model built from scratch in PyTorch, trained on the Tiny Shakespeare dataset.

## Overview

CraftGPT is a minimal decoder-only Transformer implementation designed to generate Shakespearean text at the character level. The model demonstrates the core principles of GPT architecture through a clean, educational codebase built without high-level abstractions.

## Architecture

The model implements a standard Transformer decoder with the following specifications:

- **6 Transformer layers** with residual connections
- **6 attention heads** per layer
- **384-dimensional token embeddings**
- **Character-level tokenization** (vocabulary derived from input corpus)
- **Causal self-attention** with triangular masking
- **Position-wise feed-forward networks** (4× embedding dimension)
- **Layer normalization** (pre-norm configuration)
- **Residual connections** around attention and feed-forward blocks

**Total Parameters:** ~10M

## Dataset

The model is trained on the **Tiny Shakespeare** dataset, a concatenation of Shakespeare's plays totaling approximately 1.1 MB of text (~1 million characters). The dataset provides a compact yet linguistically rich corpus suitable for character-level language modeling, capturing Early Modern English vocabulary, dramatic structure, and poetic patterns.

## Training

### Configuration

| Hyperparameter | Value |
|----------------|-------|
| Batch Size | 128 |
| Context Window (Block Size) | 256 tokens |
| Max Iterations | 5,000 |
| Initial Learning Rate | 3e-4 |
| Minimum Learning Rate | 3e-5 |
| Embedding Dimension | 384 |
| Attention Heads | 6 |
| Transformer Layers | 6 |
| Dropout | 0.25 |
| Gradient Clipping | 1.0 |
| Early Stopping Patience | 3 evaluations |

### Training Features

- **GPU Acceleration:** Automatic CUDA detection and device placement
- **Learning Rate Scheduling:** Linear warmup (100 iterations) followed by cosine annealing decay
- **Gradient Clipping:** Norm-based clipping at 1.0 to prevent exploding gradients
- **Checkpointing:** Regular checkpoints saved every 500 iterations
- **Best Model Tracking:** Separate checkpoint for the model with lowest validation loss
- **Early Stopping:** Training halts after 3 consecutive evaluations without validation improvement
- **Loss Monitoring:** Train/validation loss computed every 500 iterations over 200 batches

### Data Split

- Training: 90% of corpus
- Validation: 10% of corpus

## Results

Training stopped at **3,500 iterations** due to early stopping after validation loss failed to improve for 3 consecutive evaluations.

### Loss Metrics

| Iteration | Train Loss | Validation Loss |
|-----------|------------|-----------------|
| 0 | 4.2221 | 4.2303 |
| 500 | 1.6276 | 1.8050 |
| 1000 | 1.3422 | 1.5742 |
| 1500 | 1.2219 | 1.5039 |
| 2000 | 1.1416 | **1.4804** (best) |
| 2500 | 1.0816 | 1.4805 |
| 3000 | 1.0261 | 1.4852 |
| 3500 | 0.9797 | 1.4993 |

**Best validation loss:** 1.4804 at iteration 2000

The model demonstrates clear convergence with training loss decreasing from 4.22 to 0.98. Validation loss plateaued around 1.48 after iteration 2000, triggering early stopping at iteration 3500. The gap between training and validation loss indicates some overfitting, which is expected given the small dataset size.

### Training Curve

```
Train Loss:  4.22 → 0.98 (77% reduction)
Val Loss:    4.23 → 1.48 (65% reduction)
```

## Generated Output

After training, the model generates coherent Shakespearean-style dialogue with character names, dramatic structure, and period-appropriate vocabulary. Sample output from `more.txt`:

```
KING HEDWARD:
What was it doth, and falls our want I show on.

YORK:
O, how he told Gloucester, our force your gave toward:
Now, if all green, all visits him be heap,
Your orages and trings and sweetching us quiet-shrift.

EDWARD:
Where he stuff war sengethers his head:
Are I yet wepardon'd o't sin some of Exeter,
Let's patiently a better lips to hirst;
Did sever afterness and bitter hundred
Than gentle have made you discord again:
My passe that I shall, therefore I be key'd:
In he shall put my Romeaning head of.

Nurse:
No more is no day; but never for a weeds!
One old subdue day, take that yet let my behold
For what he should do the discopes.
```

The model successfully learns:
- Character attribution and dialogue structure
- Shakespearean character names (YORK, EDWARD, Nurse, JULIET, ROMEO)
- Dramatic formatting conventions
- Thematic vocabulary (Gloucester, Exeter, king, prince)
- Poetic cadence and rhythm

Character-level modeling produces occasional spelling inconsistencies and grammatical drift, which are expected artifacts of the approach.

## Project Structure

```
.
├── CraftGPT.py          # Main training script and model implementation
├── input.txt              # Tiny Shakespeare dataset (1.1 MB)
├── loss_history.txt       # Training and Validation loss
├── more.txt               # Generated text samples (10,000 character)
└── README.md              # This file
```

### Key Files

- **CraftGPT.py:** Complete implementation including model architecture, training loop, checkpointing system, and text generation
- **loss_history.txt:** file containing training and validation loss at each evaluation interval
- **more.txt:** Sample text generated by the trained model (10,000 tokens)

## Installation & Usage

### Prerequisites

- Python 3.x
- PyTorch with CUDA support (recommended)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akansh-j/CraftGPT.git
   cd CraftGPT
   ```

2. **Install PyTorch:**
   
   Visit [pytorch.org](https://pytorch.org/get-started/locally/) and install the appropriate build for your system.

3. **Run training:**
   ```bash
   python CraftGPT.py
   ```
   
   The script will:
   - Load the Tiny Shakespeare dataset from `input.txt`
   - Train the model with automatic checkpointing
   - Generate sample text upon completion
   - Save training metrics to `loss_history.txt`
   - Save generated samples to `more.txt`
   
   Training automatically resumes from `checkpoint.pt` if it exists. To start fresh, delete existing checkpoint files.

## Hardware

This project was developed and trained on an **NVIDIA RTX 3050 Laptop GPU** with **6 GB VRAM**.

**Memory usage:** ~5-6 GB VRAM during training with batch size 128.

The model can also run on CPU, though training will be significantly slower.

## Learning & Motivation

CraftGPT was built to understand the Transformer and GPT architecture from first principles. By implementing multi-head attention, positional embeddings, layer normalization, and the training loop without relying on high-level libraries, the project provides hands-on insight into how modern language models function at a fundamental level.

## License

License to be determined. This project is currently provided for educational and research purposes.

---

**Built with PyTorch** | Character-level modeling | Decoder-only Transformer
