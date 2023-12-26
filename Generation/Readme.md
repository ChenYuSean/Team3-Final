## Text Generation

This part is for training LoRA in the text generation model. 
* [Taiwan-LLM: Language Models for Taiwanese Culture](https://github.com/MiuLab/Taiwan-LLM): The pretrained model we used for generation

### data_process.py
This file is used to process and select data from dataset.
The hierarchy of folder is shown belowed.
```
Root Dir
|___Channel1
|       |___Video1.json
|       |___Video2.json
|       |___Video3.json
|       ....
|___Channel2
```
Each subdirectory would been seen as different channel, and the files under it would been seen as different videos.  
`prepare_dataset()` would return the training dataset for train.py. If `select = True`, each video would pick one data for each emotions(Total 7).

### train.py
This file is used to train the Taiwan Llama model. Below are some args that used in training.
* --train_file \<path>: Requried. The path to train files root directory. 
* --model_name_or_path \<path>: Required. The path to the pretrained model
* --eval_file \<path> : The path to the validation files. If None split the data by the size setting below.
* --output_dir: the path to output directory.
* --train_size: train size. If None, select all.
* --test_size: test size. If None, select all if eval_file has set path, or else complementary of train size.
* --num_video_per_channel: Number of video that choose per channel. If None, choos all videos.
* --batch_size: Batch size.
* --gradient_accumulation_steps: Gradient accumulation steps.
* --learning_rate:Learing rate.

