"""
Only extract those reply==false, which is directly responsed to the video, not to a comment.

Could use time_parsed to filter time range 
"""
import os
from os.path import isfile, isdir
import json
# import datasets

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_1_name = "techthiyanes/chinese_sentiment" # star 1-5
model_2_name = "touch20032003/xuyuan-trial-sentiment-bert-chinese" # happiness

model_1 = AutoModelForSequenceClassification.from_pretrained(model_1_name)
model_2 = AutoModelForSequenceClassification.from_pretrained(model_2_name)
tokenizer_1 = AutoTokenizer.from_pretrained(model_1_name)
tokenizer_2 = AutoTokenizer.from_pretrained(model_2_name)

base_input_folder_path = "/home/vr/disk/r12922047/2023adl-final/dataset/"
base_output_folder_path = "/home/vr/disk/r12922047/2023adl-final/train_data"


files = os.listdir(base_input_folder_path)

for input_folder_name in files:
    input_folder_path = os.path.join(base_input_folder_path, input_folder_name)
    if not isdir(input_folder_path):
          continue
    
    try:
        os.mkdir(os.path.join(base_output_folder_path, input_folder_name))
    except:
        pass


    # input_folder_name = "UCB9ryAh6vhavNxALJUJT6-Q"
    # input_folder_path = os.path.join(base_input_folder_path, input_folder_name)
    try:
        with open(os.path.join(input_folder_path, "video_metadata.json"), 'r') as f:
            meta_data = json.load(f)
    except:
        continue
    output_folder_path = os.path.join(base_output_folder_path, input_folder_name)
    output_folder_file_list = os.listdir(output_folder_path)

    channel_id = meta_data["channel_id"]
    channel_name = meta_data["channel_name"]

    index = 0
    output_data = []
    star_record = {}
    with torch.no_grad():
        for key in meta_data:
            output_data = []
            if key=="channel_id" or key=="channel_name":
                continue

            # if index >= 1:
            #     break

            index += 1
            video_id = key
            video_file_name = key + ".json"
            video_title = meta_data[key]["title"]
            video_description = meta_data[key]["description"]

            if video_file_name in output_folder_file_list:
                continue

            with open(os.path.join(input_folder_path, video_file_name), 'r')as f:
                video_data = json.load(f)
            for comment in video_data:
                if comment["reply"]:
                    continue
                cid = comment["cid"]
                text = comment["text"]
                votes = comment["votes"]
                try:
                    if "萬" in votes:
                        votes = votes.replace("萬", "")
                        votes = int(float(votes)*10000)
                    else:
                        votes = int(votes)
                except:
                    print("not int format votes: "+votes)
                
                time_record = comment["time_parsed"]

                try:
                    inputs = tokenizer_1(text, truncation=True, return_tensors="pt")
                    logits = model_1(**inputs).logits
                    predicted_class_id = logits.argmax().item()
                    star_num = model_1.config.id2label[predicted_class_id]

                    inputs = tokenizer_2(text, truncation=True, return_tensors="pt")
                    logits = model_2(**inputs).logits
                    predicted_class_id = logits.argmax().item()
                    mood = model_2.config.id2label[predicted_class_id]
                except Exception as e:
                    pass
                    # print(e)
                    # print(text)
                    # print(" ")
                    # print(" ")

                # for testing
                if star_num not in star_record:
                    star_record[star_num] = {"comment_num": 0, "acc_votes": 0}
                if type(votes) == int:
                    star_record[star_num]["comment_num"] += 1
                    star_record[star_num]["acc_votes"] += votes
                else:
                    print("votes:" +votes)

                json_data = {
                    "video_id": video_id,
                    "video_title": video_title,
                    "video_description": video_description,
                    "cid": cid,
                    "comment_text": text,
                    "votes": votes,
                    "time": time_record,
                    "star_num": star_num,
                    "mood": mood,
                }
                output_data.append(json_data)

                if star_num == "star 1":
                    if type(votes) == int:
                        if votes < 10:
                            continue
                    print("start 1 data={}, votes={}".format(text, str(votes)))

            with open(os.path.join(output_folder_path, video_file_name), "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            
            

    with open(os.path.join(output_folder_path, "star_record.json"), "w", encoding="utf-8") as f:
        json.dump(star_record, f, ensure_ascii=False, indent=4)
    print(star_record)