Background: The Whisper large-v3 model has been fine-tuned with LoRA and the fine-tuned files are now available.
Folder path: best_adapter/
This folder contains:
adapter_config.json, adapter_model.safetensors, processor_config.json, tokenizer.json, tokenizer_config.json, and a README file.
These are the outputs produced by fine-tuning.
The fine-tuning process is documented in FineTuneInstruction.md, which covers the methods and techniques used, such as Rank, Alpha, Dropout, and FP16.

After training, a quick evaluation was run on the machine. The team determined that the WER decreased noticeably after fine-tuning and the results are good enough.

Task: The model now needs to be deployed on a cloud server. After evaluation, Modal has been selected as the service provider for ASR model inference, using a T4 GPU.

Steps required:
1. Merge weights: merge the Whisper large-v3 base model with the final trained LoRA fine-tuning parameters.
2. Use the faster-whisper framework and ensure fast inference speed.

Python environment: uv is used to manage the virtual environment. `uv init` has already been run, but no dependencies have been installed yet. Modal is not installed and Modal setup has not been performed.

Data transfer: In the final project, file transfers will occur — likely MP3 or WAV files — and the service should return the transcribed text.

Testing: Postman will be used to send data and verify that the service works correctly.

After completing the code, a document is needed that covers:
1. Deployment walkthrough: step-by-step instructions for deploying to Modal via the command line
2. Code structure: what the code does and what hyperparameters are used
