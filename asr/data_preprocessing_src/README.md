这是一个对数据集进行预处理的项目
项目的最终目标：为 Whisper 模型的 LoRA 微调提供干净、标准化的训练数据。重点优化模型对儿童语言（发音特征、特殊停顿等）的语音识别能力。
此项目的目标：预处理数据集达到微调可用

目前的数据集路径框架：
* **音频文件夹**: `dataset_orig/4`, `5`, `6`, `7`, `8`, `9` (按组别存放)
    * *示例*: `dataset_orig/4/4001.mp3`
    * *参数*: 大小 20-40MB，采样率 44.1 kHz，时长 ~20 分钟/个。
* **标签文本文件夹**: `dataset_orig/OCSC`
    * *示例*: `OCSC/4/4001.cha` (与音频文件一一对应)
    * *包含角色*: `CHI` (Target_Child，目标儿童)，`EXP` (Investigator，老师/调查员)。
    * *其他文件*: 包含 `0types.txt`, `0metadata.cdc` 等 TalkBank 附带的元数据文件（暂不处理）。


音频文件不符合whisper微调的格式要求，需要进行预处理，且cha文件中无时间戳信息

通过文本编辑打开cha文件，预览信息：
‘’‘
@UTF8
@PID:	11312/a-00075513-1
@Begin
@Languages:	eng
@Participants:	EXP Investigator, CHI Target_Child, OTH Other, FAT Father, EX2 Investigator
@ID:	eng|OCSC|EXP|||||Investigator||KM|
@ID:	eng|OCSC|CHI|6;09.13|male|||Target_Child||Cillian|
@ID:	eng|OCSC|OTH|||||Other||robot|
@ID:	eng|OCSC|FAT|||||Father|||
@ID:	eng|OCSC|EX2|||||Investigator||KS|
@Media:	6023, audio, unlinked
@Types:	cross, interview, TD
*EXP:	one more minute .
%mor:	num|one adj|more-Cmp-S1 noun|minute .
%gra:	1|3|NUMMOD 2|3|AMOD 3|0|ROOT 4|3|PUNCT
*CHI:	okay .
%mor:	intj|okay .
%gra:	1|0|ROOT 2|1|PUNCT
*EXP:	alrighty +/.
%mor:	intj|alrighty +/.
%gra:	1|0|ROOT 2|1|PUNCT
*CHI:	xxx I opened it .
%mor:	pron|I-Prs-Nom-S1 verb|open-Fin-Ind-Past-S1 pron|it-Prs-Acc-S3 .
%gra:	1|2|NSUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT
*CHI:	and I sawed , so +...
%mor:	cconj|and pron|I-Prs-Nom-S1 verb|saw-Fin-Ind-Past-S1 cm|cm adv|so +...
%gra:	1|3|CC 2|3|NSUBJ 3|0|ROOT 4|5|PUNCT 5|3|ADVMOD 6|3|PUNCT
*EXP:	okay .
%mor:	intj|okay .
%gra:	1|0|ROOT 2|1|PUNCT
*EXP:	yeah , we'll see in a minute .
%com:	start tasks .
%mor:	intj|yeah cm|cm intj|well verb|see-Fin-Imp-S adp|in det|a-Ind-Art noun|minute .
%gra:	1|4|DISCOURSE 2|1|PUNCT 3|4|DISCOURSE 4|0|ROOT 5|7|CASE 6|7|DET 7|4|OBL 8|4|PUNCT
*EXP:	so this is Cillian six zero two three .
%mor:	adv|so pron|this-Dem-S1 aux|be-Fin-Ind-Pres-S3 propn|Cillian num|six num|zero num|two num|three .
%gra:	1|4|DISCOURSE 2|4|NSUBJ 3|4|COP 4|0|ROOT 5|4|NUMMOD 6|4|NUMMOD 7|8|NUMMOD 8|4|NMOD-UNMARKED 9|4|PUNCT
*EXP:	so , I'm gonna first introduce you to our friend here .
%mor:	adv|so cm|cm intj|im adv|gonna adv|first verb|introduce-Inf-S pron|you-Prs-Acc-S2 adp|to pron|our-Prs-Gen-P1 noun|friend adv|here .
%gra:	1|6|DISCOURSE 2|1|PUNCT 3|6|DISCOURSE 4|6|ADVMOD 5|6|ADVMOD 6|0|ROOT 7|6|OBJ 8|10|CASE 9|10|NMOD-POSS 10|6|OBL 11|6|ADVMOD 12|6|PUNCT
*EXP:	let's let them introduce themselves .
’‘’

需要进行的步骤：
1. 本数据集来源于TalkBank网站，需要进行预处理
2. 根据其转录信息删除特殊符号等内容：
3. 使用stable-ts进行大模型对齐，将cha文件中和原始音频进行对齐，得到时间戳。使用的模型框架：faster-whisper，并且使用medium模型。将清洗后的文本与长音频进行毫秒级对齐，获取每句话的精确时间戳。
4. 为了符合whisper微调的格式，需要进行转化，变成一个结构极其简单、跨平台通用的 Hugging Face AudioFolder 格式。（16000Hz, 单声道，且切片时长控制在 30 秒以内）
5. 最后的输出在clean_dataset文件夹中,并且不再区分4，5，6，7，8，9文件夹，统一在metadata.csv中显示file_name（写相对路径）和transcription，clean_dataset/audio/中存储音频文件

环境依赖管理：使用uv进行管理，ffmpeg已经安装在此mac设备的brew中
可能涉及到的依赖已经成功通过uv安装：
pylangacq pydub librosa soundfile pandas tqdm stable-ts faster-whisper


数据集网站上其他存在的信息：
Within each transcript, tasks are marked with Gem codes (@G) as specified below. In addition, when social chit-chat at the end of the session was included, it is marked with the code @G: EndTasks. In addition, we have included on this site three cut documents for special words found in our transcripts: (1) OCSC_lofreq includes low frequency words that were used by children in this task; (2) OCSC_wugs includes the nonsense words used in the Wug task; (3) OCSC_comm includes our specific “communicator” conventions, used for transcribing children sounding out letters in the alphabet task.

Alphabet (@G: Alphabet)：Children were shown 26 cards (in alphabetical order), each containing a capital letter, a word starting with that letter and a picture of the word. Children were asked to name the letter and the picture, and to think of another word that started with the same letter. Children who were having difficulties with letter naming were not pressured to name the picture or provide another word that started with the same letter.
Numbers (@G: Numbers)
Wug Task (@G: Wug)：Children were run through a version of the classic Wug task (Berko, 1958) focusing exclusively on plural morphology. Children were asked to produce the plural forms for 10 common words.
Experimental Pictures (@G: ExpPictures)
Reading Passage (@G: Reading)
How To Task (@G: Howto)
Descriptive Pictures (@G: DescriptivePictures)