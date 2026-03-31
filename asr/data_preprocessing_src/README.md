This project preprocesses the dataset.
Final goal: provide clean, standardized training data for LoRA fine-tuning of the Whisper model, with a focus on improving speech recognition of child language (pronunciation patterns, special pauses, etc.).
Objective of this project: preprocess the dataset to a state ready for fine-tuning.

Current dataset directory structure:
* **Audio folder**: `dataset_orig/4`, `5`, `6`, `7`, `8`, `9` (organized by group)
    * *Example*: `dataset_orig/4/4001.mp3`
    * *Properties*: size 20–40 MB, sample rate 44.1 kHz, duration ~20 minutes each.
* **Transcript label folder**: `dataset_orig/OCSC`
    * *Example*: `OCSC/4/4001.cha` (one-to-one correspondence with audio files)
    * *Speakers*: `CHI` (Target_Child), `EXP` (Investigator/teacher).
    * *Other files*: includes TalkBank metadata files such as `0types.txt`, `0metadata.cdc` (not processed for now).


The audio files do not meet Whisper fine-tuning format requirements and must be preprocessed. The .cha files also contain no timestamp information.

Opening a .cha file in a text editor shows:
'''
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
'''

Required steps:
1. This dataset originates from the TalkBank website and must be preprocessed.
2. Remove special symbols and other artifacts based on the transcript content.
3. Use stable-ts for large-model alignment: align the cleaned text from the .cha files with the original audio at millisecond precision to obtain timestamps. Framework used: faster-whisper with the medium model. Each cleaned transcript is aligned against the full-length audio to get precise per-utterance timestamps.
4. To comply with Whisper fine-tuning format requirements, convert to the simple, cross-platform HuggingFace AudioFolder format (16000 Hz, mono, segments no longer than 30 seconds).
5. Final output goes into the `clean_dataset/` folder. The group subdirectories (4, 5, 6, 7, 8, 9) are no longer used; everything is unified in `metadata.csv` with `file_name` (relative path) and `transcription` columns. Audio files are stored in `clean_dataset/audio/`.

Environment management: uv is used; ffmpeg is already installed via Homebrew on this Mac.
Dependencies successfully installed via uv:
pylangacq pydub librosa soundfile pandas tqdm stable-ts faster-whisper


Additional information from the dataset website:
Within each transcript, tasks are marked with Gem codes (@G) as specified below. In addition, when social chit-chat at the end of the session was included, it is marked with the code @G: EndTasks. In addition, we have included on this site three cut documents for special words found in our transcripts: (1) OCSC_lofreq includes low frequency words that were used by children in this task; (2) OCSC_wugs includes the nonsense words used in the Wug task; (3) OCSC_comm includes our specific "communicator" conventions, used for transcribing children sounding out letters in the alphabet task.

Alphabet (@G: Alphabet): Children were shown 26 cards (in alphabetical order), each containing a capital letter, a word starting with that letter and a picture of the word. Children were asked to name the letter and the picture, and to think of another word that started with the same letter. Children who were having difficulties with letter naming were not pressured to name the picture or provide another word that started with the same letter.
Numbers (@G: Numbers)
Wug Task (@G: Wug): Children were run through a version of the classic Wug task (Berko, 1958) focusing exclusively on plural morphology. Children were asked to produce the plural forms for 10 common words.
Experimental Pictures (@G: ExpPictures)
Reading Passage (@G: Reading)
How To Task (@G: Howto)
Descriptive Pictures (@G: DescriptivePictures)
