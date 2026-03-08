# Arithmetic Reasoning with Tiny Transformer and Qwen LoRA

This repository explores **character-level arithmetic reasoning** using two contrasting language-modeling approaches:

1. a **custom Tiny Transformer** trained from scratch, and  
2. **Qwen2.5-0.5B-Instruct** adapted with **LoRA / QLoRA-style fine-tuning**.

The project focuses on controlled generalization for simple symbolic math, using synthetic addition and subtraction expressions with carefully designed train, validation, reinforcement-learning prompt, and test splits.

## Project Overview

The core task is to predict the correct numeric answer for arithmetic prompts such as:

```text
103 - 88 =
57 + 21 =
```

Rather than treating this as a standard classification problem, the notebook formulates it as a **causal language modeling task** at the **character level**, where the model generates the final answer digits autoregressively.

The project compares:
- a lightweight Transformer implemented manually in PyTorch,
- a modern instruction-tuned large language model adapted with parameter-efficient fine-tuning,
- performance across multiple distribution settings:
  - **seen examples**
  - **unseen operands**
  - **unseen prompt formats**

## Main Features

- Synthetic dataset generation for arithmetic expressions
- Controlled holdout of operand sets for out-of-distribution testing
- Evaluation on formatting shifts such as spacing variations
- Character-level vocabulary and masked prompt-to-answer training
- Custom implementation of:
  - positional encoding
  - multi-head attention
  - transformer blocks
  - greedy decoding
- Fine-tuning pipeline for **Qwen2.5-0.5B-Instruct**
- Support for **LoRA** and optional **4-bit quantization**
- Exact-match accuracy reporting by split, operator type, and number length

## Repository Structure

```text
arithmetic-reasoning-transformers/
├── code.ipynb
├── README.md
├── requirements.txt
├── data_task1/              # generated CSV splits
├── checkpoints/             # Tiny Transformer checkpoints
└── checkpoints_qwen_lora/   # Qwen LoRA checkpoints
```

## Methodology

### 1. Data Generation
The notebook creates synthetic arithmetic samples using:
- variable digit lengths,
- balanced addition and subtraction,
- operand holdout sets for true unseen-number evaluation,
- formatting perturbations for robustness testing.

Generated splits include:
- pretraining train set,
- validation set,
- RL prompt set,
- seen test set,
- unseen-number test set,
- unseen-format test set.

### 2. Tiny Transformer from Scratch
A compact Transformer language model is implemented directly in PyTorch with:
- learned embeddings,
- sinusoidal positional encodings,
- masked self-attention,
- feed-forward layers,
- causal next-token prediction.

The loss is masked over the prompt so the model is only trained to generate the answer.

### 3. Qwen + LoRA / QLoRA
The project also adapts **Qwen2.5-0.5B-Instruct** using:
- Hugging Face Transformers,
- PEFT LoRA adapters,
- optional bitsandbytes quantization,
- prompt formatting with system and user messages.

This allows comparison between a small custom model and a pretrained instruction model under parameter-efficient fine-tuning.

## Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

For notebook use:

```bash
jupyter notebook
```

## Usage

Open the notebook and run it in order:

```bash
jupyter notebook code.ipynb
```

The workflow includes:
1. generating arithmetic datasets,
2. building the character-level vocabulary,
3. training the Tiny Transformer,
4. evaluating exact-match accuracy,
5. loading Qwen,
6. attaching LoRA adapters,
7. fine-tuning and evaluating the Qwen-based model.

## Evaluation

The notebook measures:
- exact-match accuracy,
- operator-specific performance,
- accuracy by operand length,
- robustness to unseen numbers,
- robustness to unseen prompt formatting.

This makes the project useful for studying **algorithmic generalization** and **reasoning robustness** in language models.

## Dependencies

Core libraries used in the notebook include:
- PyTorch
- pandas
- scikit-learn
- transformers
- accelerate
- peft
- bitsandbytes
- matplotlib
- jupyter

## Notes

- The Qwen/LoRA section is most suitable for a CUDA-enabled environment.
- `bitsandbytes` may require extra system compatibility depending on platform.
- Generated datasets, checkpoints, and cached outputs do not need to be committed unless you want to publish trained artifacts.
- For a clean GitHub repo, it is best to keep large checkpoints and generated CSV files out of version control.

## Suggested Improvements

- move training code from notebook cells into Python modules,
- add CLI scripts for training and evaluation,
- log experiments with TensorBoard or Weights & Biases,
- benchmark more arithmetic operators and larger number ranges,
- compare chain-of-thought prompting versus direct answer generation.

## License

This repository is shared as part of a personal machine learning and language modeling portfolio.
