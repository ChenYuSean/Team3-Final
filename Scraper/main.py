from itertools import islice
from youtube_comment_downloader import *
import scrapetube
import ipdb
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--channel_list_file', type=str, default='./channel_list.json')
args = parser.parse_args()

channel_list = json.load(open(args.channel_list_file , 'r'))

for index, channel_key in enumerate(channel_list.keys()):
    print(f'{index+1}/{len(channel_list.keys())}', channel_key)
    videos = scrapetube.get_channel(
        channel_id=channel_list[channel_key],
        sort_by="newest",
        limit=50,
    )

    video_json = dict()
    video_json['channel_id'] = channel_list[channel_key]
    video_json['channel_name'] = channel_key
    for video in videos:
        video_json[video['videoId']] = {}
        video_json[video['videoId']]['title'] = video['title']['runs'][0]['text']
        video_json[video['videoId']]['description'] = video['descriptionSnippet']['runs'][0]['text'] if 'descriptionSnippet' in video else ''

        downloader = YoutubeCommentDownloader()

        comments = downloader.get_comments_from_url(
            f'https://www.youtube.com/watch?v={video["videoId"]}', 
            sort_by=SORT_BY_POPULAR,
            )
            
        if not os.path.exists("./data/" + channel_list[channel_key]):
            os.makedirs("./data/" + channel_list[channel_key])
        with open(f"./data/{channel_list[channel_key]}/{video['videoId']}.json", "w") as f:
            json.dump(list(comments), f, indent=4, ensure_ascii=False)

    with open(f"./data/{channel_list[channel_key]}/video_metadata.json", "w") as f:
        json.dump(video_json, f, indent=4, ensure_ascii=False)
