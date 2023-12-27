from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
from linebot.models import MessageEvent, TextMessage, TextSendMessage, TemplateSendMessage, ButtonsTemplate, PostbackAction

import tempfile, os
import datetime
import time
import traceback
from typing import Optional, Dict, Sequence
from os.path import exists, join, isdir

from scrab_title_descp import scrape_video_info

import torch
import transformers

import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    pipeline
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer

app = Flask(__name__)
static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')

# CHANNEL_ACCESS_TOKEN
line_bot_api = LineBotApi('CHANNEL_ACCESS_TOKEN')

# CHANNEL_SECRET
handler = WebhookHandler('CHANNEL_SECRET')

#USER STATE
user_states={}


"""
==== Get accelerate model and load model/tokenizer =====

"""
#get accelerate model
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    
max_memory = f'{80000}MB'
max_memory = {i: max_memory for i in range(n_gpus)}
device_map = "auto"

# if we are in a distributed setting, we need to set the device map and max memory per device
if os.environ.get('LOCAL_RANK') is not None:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    max_memory = {'': max_memory[local_rank]}

DEFAULT_PAD_TOKEN = "[PAD]"
# load model
model_name="model/Taiwan-LLM-7B-v2.0-chat"
checkpoint_dir="model/checkpoint-1350"

print(f'loading base model {model_name}...')
compute_dtype = torch.float16        #(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit= 4,
    load_in_8bit=8,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=4,
        load_in_8bit=8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype= torch.float32 ,     #(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
)
model.config.torch_dtype=torch.float32  #model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

# Tokenizer
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    tokenizer_type='llama' if 'llama' in model_name else None, # Needed for HF name change
)
if tokenizer._pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )
if 'llama' in model_name or isinstance(tokenizer, LlamaTokenizer):
    tokenizer.add_special_tokens({
                    "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                    "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                    "unk_token": tokenizer.convert_ids_to_tokens(
                        model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                    ),
            })

# Loading checkpoint 
print("Loading adapters from checkpoint.")
model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)


"""
==== robot response =====

"""
def get_prompt(title:str, description:str, mode:str) -> str:
    '''Format the instruction as a prompt for LLM.'''

    # comment_type = '正面評論' if mood in ['like','happiness'] else '負面評論' if mood in ['sadness','anger','fear'] else '中立評論'
    # moods = ['like','sadness','anger','fear']
    # ch_moods = ['喜歡','難過','生氣','害怕']
    # if mood in moods:
    #     mood = ch_moods[moods.index(mood)]
    comment_type = '正面評論' if mode=='positive' else '負面評論' if mode=='negative' else '中立評論'
    
    return f"請幫這部影片生出對應需求的{comment_type}。影片標題:[{title}]。影片敘述:[{description}]。\
ASSISTANT:"

def response(text):
    
    pipe = pipeline("text-generation", model=model,tokenizer=tokenizer, torch_dtype=torch.float32, device_map="auto")

    outputs = pipe(text, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    answer=outputs[0]["generated_text"].split('ASSISTANT:', 1)[-1].strip()

    return answer


# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


# 處理訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    if user_id not in user_states:
        user_states[user_id]={}     # if user states not exist, initiate the dictionary

    if 'mode' not in user_states[user_id]: # User selecting the mode 'positive/negative'
        if event.message.text.lower() == 'mode':
            buttons_template=ButtonsTemplate(
                title='Choose Mode',
                text='請選擇模式: ',
                actions=[
                    PostbackAction(label='正面評論', data='positive'),
                    PostbackAction(label='負面評論', data='negative')
                ]
            )
            template_message = TemplateSendMessage(alt_text='Choose Mode', template=buttons_template)
            line_bot_api.reply_message(event.reply_token, template_message)
            
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text='Send "mode" to choose a mode.'))
    # elif 'mood' not in user_states[user_id]:
    #     moods = ['like','sadness','anger','fear']
    #     if event.message.text.lower() == 'mood':
    #         buttons_template=ButtonsTemplate(
    #             title='Choose Mood',
    #             text='請選擇情感: ',
    #             actions=[
    #                 PostbackAction(label='喜歡', data='like'),
    #                 PostbackAction(label='難過', data='sadness'),
    #                 PostbackAction(label='生氣', data='anger'),
    #                 PostbackAction(label='害怕', data='fear')
    #             ]
    #         )
    #         template_message = TemplateSendMessage(alt_text='Choose Mood', template=buttons_template)
    #         line_bot_api.reply_message(event.reply_token, template_message)
    #     else:
    #         line_bot_api.reply_message(event.reply_token, TextSendMessage(text='Send "mood" to choose a mood.'))
    else:   # User already chose  mode and mood
        user_input = event.message.text  # get the yt url
        user_mode = user_states[user_id]['mode']
        #user_mood = user_states[user_id]['mood']

        # scrap tile and description od video
        title, description=scrape_video_info(user_input)

        model_input=get_prompt(title,description,user_mode)
        try:
            answer = response(model_input)
            print(answer)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(answer))
        except:
            print(traceback.format_exc())
            line_bot_api.reply_message(event.reply_token, TextSendMessage('Error occured, please wait...'))

    

@handler.add(PostbackEvent)
def handle_message(event):
    print(event.postback.data)
    user_id = event.source.user_id
    data = event.postback.data

    # 在這裡根據按鈕回傳的資料進行處理
    if data == 'positive':
        # 如果是正面評論，進行相應處理
        user_states[user_id]['mode'] = 'positive'
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text='你選擇了正面評論模式'))
    elif data == 'negative':
        # 如果是負面評論，進行相應處理
        user_states[user_id]['mode'] = 'negative'
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text='你選擇了負面評論模式'))
    else:
        # 其他情況
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text='Unknown selection.'))


@handler.add(MemberJoinedEvent)
def welcome(event):
    uid = event.joined.members[0].user_id
    gid = event.source.group_id
    profile = line_bot_api.get_group_member_profile(gid, uid)
    name = profile.display_name
    message = TextSendMessage(text=f'{name}歡迎加入')
    line_bot_api.reply_message(event.reply_token, message)
        
        
import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port)
