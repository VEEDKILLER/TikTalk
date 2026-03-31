这是一个使用whisper模型并使用LoRA进行微调的项目
项目的数据集：使用已经预处理好的来自TalkBank的儿童语音对话数据集。

数据集结构和路径：
元数据，包含语音文件路径和转录文本信息：dataset/metadata.csv
语音文件：dataset/audio/xxx.wav
（目前由于audio数据庞大，只挑选了5个样本作为代码训练测试）

注意：数据集未进行训练集、验证集、测试集的划分，需要在训练时候进行划分，指定随机种子42，便于使用baseline模型和微调模型进行对比实验。

项目的微调信息：
1. baseline模型：使用未被微调的简单whisper模型
2. 微调模型：使用whisper large-v3模型，并使用LoRA进行微调
3. 需要实现早停机制，当验证集上的wer连续3个epoch没有下降时，停止训练
4. 对比两个模型的在使用测试集上的差异，表现出微调的效果

本项目使用uv进行环境管理（除了bitsandbytes其他全都安装完成）
代码测试训练环境：mac，m4芯片（本次）
实际训练环境：3080ti显卡，16g显存，wsl2的ubuntu系统（未来，需要做到可以直接同代码迁移，通过检测设备显卡cuda）
需要用到的依赖：
Huggingface的transformers，peft，datasets，torch，soundfile，jiwer+evaluate（用于计算wer、ser、cer指标），tqdm
accelerate
tensorboard（用于可视化训练过程）
bitsandbytes（因为使用large-v3模型，以较低精度加载模型，注意，需要在cuda上时候，当在mac环境中不启动，目前未使用uv安装）

使用Seq2SeqTrainer + peft + Accelerate + bitsandbytes组合去微调训练

LoRA (PEFT) 配置参数
* **Target Modules**: `["q_proj", "v_proj"]` (专注微调注意力机制核心层)。
* **Rank (r)**: 32
* **Alpha**: 64
* **Dropout**: 0.05

训练超参数 (Training Arguments)
* **显存优化**: `per_device_train_batch_size=2`，配合 `gradient_accumulation_steps=8` 达到等效 Batch Size 16。
* **精度控制**: 启用 FP16 (`fp16=True`)。
* **学习率**: `1e-4` (配合 AdamW 优化器与线性学习率预热 warmup)。
* **早停机制 (Early Stopping)**: 监控验证集 `wer`，`patience=3` (连续 3 个 epoch 未下降即停止)。

* **动态 Padding**: 使用自定义 `DataCollatorSpeechSeq2SeqWithPadding`，将音频输入特征补齐，并将标签序列的 padding 标记替换为 `-100` 以屏蔽 Loss 计算。
* **评估方式**: 在评估阶段开启 `predict_with_generate=True` 让模型自回归生成文本。
* **文本标准化 (Normalization)**: 在计算 WER/CER 前，对预测文本和标签文本统一应用 Whisper English Normalizer（移除标点、转小写），确保指标真实反映语音识别能力。

需要使用现代化参数配置的方法去编写代码。

现在先在mac上测试，看跑一个batch具体需要花费多长的时间