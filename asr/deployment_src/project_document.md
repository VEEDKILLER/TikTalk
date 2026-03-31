背景：通过LoRA微调Whipser large-v3模型，现在已经获得了微调后的文件
文件夹路径：best_adapter/
这个文件夹中包含了
adapter_config.json, adapter_model.safetensors,processsor_config.json,tokenizer.json, tokenizer_config.json还有一个README文件。
这些是微调之后的产出。
微调过程见FineTuneInstructions.md文件，这个文件包含了微调过程中使用的方法和技术比如Rank，Alpha，Dropout，使用了fp16

在训练结束后，在机器进行简单的评估，小组认为微调后wer下降明显，效果足够好

任务：目前需要对其进行部署在云端的服务器中。经过挑选，决定使用Modal作为服务商，进行ASR模型的推理，并使用T4的GPU

需要做的步骤：
1. 合并权重：需要将Whisper large v3模型和最终训练的LoRA微调参数进行合并。
2. 使用faster-whisper框架，并且需要做到推理的速度要快，

Python环境：使用uv来管理虚拟环境，目前已进行`uv init`操作。但是未安装任何依赖。目前未安装Modal，未进行Modal设置

传输的数据：在最终的项目，将会进行文件传输，有可能传输mp3文件或者wav文件，接收转录后的文本。

小幅测试：会使用postman进行发送数据看是否能够正常运作
完成代码后需要做一个文档，内容为：
1. 流程说明，我知道如何利用命令行一步一步部署在Modal上
2. 代码结构是如何，里面用了什么超参数？