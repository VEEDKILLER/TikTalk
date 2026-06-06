This project fine-tunes the Whisper model using LoRA.
Dataset: a preprocessed child speech conversation dataset from TalkBank.

Dataset structure and paths:
Metadata (containing audio file paths and transcription text): dataset/metadata.csv
Audio files: dataset/audio/xxx.wav
(Currently, only 5 samples have been selected for code training and testing due to the large audio dataset size.)

Note: The dataset has not been split into train/validation/test sets. The split must be performed at training time using random seed 42, to allow fair comparison between the baseline model and the fine-tuned model.

Fine-tuning details:
1. Baseline model: the plain Whisper model without any fine-tuning
2. Fine-tuned model: Whisper large-v3 fine-tuned with LoRA
3. Early stopping is required: stop training when the WER on the validation set does not decrease for 3 consecutive epochs
4. Compare the two models on the test set to demonstrate the effect of fine-tuning

This project uses uv for environment management (all dependencies except bitsandbytes are installed).
Code testing environment: Mac, M4 chip (current)
Actual training environment: 3080Ti GPU, 16GB VRAM, Ubuntu on WSL2 (future — the same code should transfer directly by detecting the CUDA GPU)

Required dependencies:
Hugging Face transformers, peft, datasets, torch, soundfile, jiwer+evaluate (for WER/SER/CER metrics), tqdm
accelerate
tensorboard (for visualizing the training process)
bitsandbytes (for loading the large-v3 model at reduced precision; note: only enabled on CUDA, not activated on Mac; not yet installed via uv)

Training stack: Seq2SeqTrainer + peft + Accelerate + bitsandbytes

LoRA (PEFT) configuration
* **Target Modules**: `["q_proj", "v_proj"]` (focused on the core attention layers)
* **Rank (r)**: 32
* **Alpha**: 64
* **Dropout**: 0.05

Training hyperparameters (Training Arguments)
* **Memory optimization**: `per_device_train_batch_size=2` combined with `gradient_accumulation_steps=8` achieves an effective batch size of 16.
* **Precision**: FP16 enabled (`fp16=True`).
* **Learning rate**: `1e-4` (with AdamW optimizer and linear warmup schedule).
* **Early Stopping**: monitors validation `wer` with `patience=3` (stops if no improvement for 3 consecutive epochs).

* **Dynamic Padding**: uses custom `DataCollatorSpeechSeq2SeqWithPadding` to pad audio input features and replace label padding tokens with `-100` to mask the loss.
* **Evaluation**: `predict_with_generate=True` enables autoregressive text generation during evaluation.
* **Text Normalization**: before computing WER/CER, applies the Whisper English Normalizer (removes punctuation, lowercases) to both predictions and references, ensuring metrics accurately reflect speech recognition performance.

Code should use modern configuration patterns.

First test on Mac to see how long a single batch takes.
