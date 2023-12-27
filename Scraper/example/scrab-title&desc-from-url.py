from pytube import YouTube
from json import loads

def get_description(video: YouTube) -> str:
    i: int = video.watch_html.find('"shortDescription":"')
    desc: str = '"'
    i += 20 
    while True:
        letter = video.watch_html[i]
        desc += letter  # letter can be added in any case
        i += 1
        if letter == '\\':
            desc += video.watch_html[i]
            i += 1
        elif letter == '"':
            break
    return loads(desc)

def scrape_video_info(url):
    try:
        video = YouTube(url)
        title = video.title
        description = get_description(video)
        return title, description
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    title, description = scrape_video_info(video_url)
    print(f"Title: {title}")
    print(f"Description: {description}")
