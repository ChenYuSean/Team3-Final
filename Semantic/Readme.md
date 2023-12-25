# Semantic
This part is to extract the semantic tag of the youtube comments we wet.

We use two different pre-trained BERT model from huggingface to get the semantic tag.

1. [techthiyanes/chinese_sentiment](https://huggingface.co/techthiyanes/chinese_sentiment): This model will classify a sentence to 5 categories: star1 ~ star5. Sentences of star1 is kind of the most negative sentences. Sentences of star5 is kind of the most positive sentences.
2. [touch20032003/xuyuan-trial-sentiment-bert-chinese](https://huggingface.co/touch20032003/xuyuan-trial-sentiment-bert-chinese): This model will classify a sentence to 8 categories of mood, which are `none`, `disgust`, `happiness`, `like`, `fear`, `sadness`, `anger`, and `surprise`.

## run.py
This file could read the output of scraper, predict two type of semantic tags, and write to another file.

* We delete those comments which is under another comment.
* This program will add `video_title` and `video_description` from `video-meta.json` to each json dictionary.
* The input data format should be like:

```
{
    "cid": "Ugy4_FkR2wVzzlPAa-l4AaABAg",
    "text": "爛爛 偷褲 大缸開唱打👋鎗 是黏清人的襊愛",
    "time": "2 年前",
    "author": "@chl5125",
    "channel": "UC94iuAHGFEKkZWCLJMbY0_A",
    "votes": "168",
    "photo": "https://yt3.ggpht.com/ytc/APkrFKa8d4UWzHItasvpVHD4pveaCK-WYRnxp5V9vw=s176-c-k-c0x00ffffff-no-rj",
    "heart": false,
    "reply": false,
    "time_parsed": 1638972781.945735
},
```

* The output format should be like:

```
{
    "video_id": "_-X-DjZ9TG0",
    "video_title": "【呱吉】呱張新聞EP18：你以為的法院認證不是你以為的",
    "video_description": "你以為喔？\n\n--\n📌加入呱吉頻道會員：https://bit.ly/3361X61\n📌呱吉的Podcast（每週更新）：https://apple.co/2GKc3Rp\n📌呱吉FB粉絲團：https://www.facebook.com/froggych...",
    "cid": "Ugy4_FkR2wVzzlPAa-l4AaABAg",
    "comment_text": "爛爛 偷褲 大缸開唱打👋鎗 是黏清人的襊愛",
    "votes": 168,
    "time": 1638972781.945735,
    "star_num": "star 5",
    "mood": "anger"
}
```


* After editting the `base_input_folder_path` and  `base_output_folder_path` in the beginning of `run.py`, you could execute the program by directly run `python run.py`(under this folder). And then the program will write output json files to the folder you assigned.