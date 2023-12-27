# YouTube Comment Crawler

## Environment Setup

```bash
pip install -r requirement.txt
```

## Usage

```bash
python main.py --channel_list_file <file_path>
```

## channel_list_file format

```json
{
    "channel_id": "channel_name"
    // ...
}
```

## Collected Data Description

There are multiple directory whose name corresponds to the `channel_id`.

Within the directory, a file named `video_metadata.json` houses essential information, including the `channel_id`, `channel_name` of the channel, as well as the `title` and `description` of each video.
Other files named with `video_id` contain the comments associated with the video.

This is one sample comment  

```json
    "cid": "UgzYaJQfRHsmFXeJBsp4AaABAg",
    "text": "一開始王世堅還有點客客氣氣的 沒有很放得開 提到柯文哲整個精神都來了  感謝他們重新定義了什麼叫靈魂伴侶👨‍❤️‍👨",
    "time": "11 個月前 (已編輯)",
    "author": "@Han-uf6hj",
    "channel": "UCt_mYeMOpo7FPNx4feSz6cg",
    "votes": "3502",
    "photo": "https://yt3.ggpht.com/GG9Vrbaxa76gNAvc9LcggCyFkZgm3yOgezVWJaXm9cjhZwLknr8bMB8YplqErb2nef_U0ef89Q=s176-c-k-c0x00ffffff-no-rj",
    "heart": true,
    "reply": false,
    "time_parsed": 1673186467.126761
```

## Channel List (total 100 channels)

**12/07 update (6 channels)**

- '@STRNetworkasia': 'UCDrswN-SqWh7Kii62h9aXGA',
- '@joeman': 'UCPRWWKG0VkBA0Pqa4Jr5j0Q',
- '@Toyz69': 'UC2h6FK5xqOIqMM9MIYOdFzg',
- '@Notorious_3cm': 'UCi2GvcaxZCN-61a0co8Smnw',
- '@ebcapocalypse': 'UCqdR2j_W3QRyw5nc-0Ku-0A',
- '@rayduenglish': 'UCeo3JwE3HezUWFdVcehQk9Q',

**12/08 update (13 channels)**

- '@DcardTaiwan': 'UCaaZYXjaVXrxegTZHCAxDmQ',
- '@helloiamhook': 'UCbIJeyl_va8MG2xx0q4Uobg',
- '@aliceinsiliconwonderland': 'UCB9ryAh6vhavNxALJUJT6-Q',
- '@xilanceylan': 'UCVjmBo22Mimyc4NR4Nz89MA',
- '@bluepigeon0810': 'UCUn77_F5A65HViL9OEvIpLw',
- '@TheDoDoMen': 'UCfq75-6J5seC82CmtLSFxXw',
- '@amogoodlife': 'UCWI5T5awmyoUBw72aNcqadQ',
- '@beautywu': 'UCAfcy122TZHqDQMMAwfbvBQ',
- '@Zoebitalk': 'UCs4jRrfq8fWWwMj6-Xj_wHw',
- '@BIGSNAKEBALL': 'UCQd3V4gv_lcLX9SpIVW3wmw',
- '@MariaAbe': 'UC2tRcusVoXSGqUcSu1GO4Ng',
- '@bailingguo': 'UCD2KoUc0f4Bv2Bz0mbOah8g',
- '@shasha77': 'UCiWXd0nmBjlKROwzMyPV-Nw'

**12/09 update (16 channels)**

- '@nsfwstudio': 'UCj_z-Zeqk8LfwVxx0MUdL-Q', 
- '@jam_steak': 'UCZJ6_ybUlrz2M-QrstqG4dg',
- '@amazingtalkershow': 'UCFpIb7s-_kx3mbxTIGMWvPw',
- '@FroggyChiu': 'UC_XRq7JriAORvDe1lI1RAsA',
- '@CtiNews': 'UCpu3bemTQwAU8PqM4kJdoEQ',
- '@funnynoproblem': 'UC6Unc9BmBvkWI_YPjGUyoIw',
- '@LadyFlavor': 'UCOz7W0VH--2WlqAE_3jtL4w',
- '@ttshowtw': 'UCIECJyQ6meDyN-UnVHgXAMA',
- '@PanScitw': 'UCuHHKbwC0TWjeqxbqdO-N_g',
- '@laogao': 'UCMUnInmOkrWN4gof9KlhNmQ',
- 'vsmediatw': 'UCiYZw0h6hA5ENlPhTZFTHTA', 
- '@lynnwu0219': 'UCiewBSUlxrhoyn6oTNt0ilw',
- '@Hahatai': 'UC9g4w3QvOCTM_3ok6WlH43Q',
- '@teeprbala': 'UCzAOdjLlfyW19t8PtG1f7MA',
- '@OMIOBEN': 'UChNqKKo9tAbj3v2hCkcPFOQ',
- '@HighlightSpace': 'UCJjoK1VO-NFfeGA7s9GMVLA',

**12/10 update (19 channels)**

- '@chillseph': 'UCBY6NwU6OpYQiPYR1urdF0g',
- '@Muyao4': 'UCLW_SzI9txZvtOFTPDswxqg',
- '@594blackdragon': 'UChC0xsoHg-yBsFF2P844bkA',
- '@tessereq': 'UC0Q-fBheHysYWz9ObSEzMdA',
- '@hanhanpovideo': 'UCEBaVKCwLP3UOdSq3LNNqmw',
- '@BaxuanMei': 'UC2SmF-JiJfPbod2MuW-Drcg',
- '@What-AllBlow-Mean': 'UCrXFZWUCMD7RoROtZOajYRQ',
- '@big_star_ken': 'UCZVCbj9weVNAWqXS9gnfm5A',
- '@Chienseating': 'UC9i2Qgd5lizhVgJrdnxunKw',
- '@user-huangbrothers': 'UCV_S2S-Zs8LeuJxK-T3RQQg',
- '@namewee': 'UCFUtqTcgJgRnmZ3tMU6P74Q',
- '@NANACIAOCIAO': 'UCRm_PQqRwiwA7g8bqCoPH9A',
- '@onion_man': 'UCzxN4G3s9uR9ao5_O5DoXmA',
- '@guy1224': 'UCpGGLFkG4heKm6hnJYn5HxA',
- '@McJengSu': 'UCuTYW0IJthW5oztb-j77HBQ',
- '@ruge1222': 'UCBCGkO6uBZdrNbuP4qrm5CA',
- '@DeeGirlsTalk': 'UC-H0vfujGN6LDf8VW9YY2Eg',
- '@nerdzun': 'UC-ujeda5rDgCe-910J5keTg',
- '@huzi1989': 'UC9YOQFPfEUXbulKDtxeqqBA'

**12/11 update (12 channels)**

- '@vw1229': 'UClRL6qxRQRsjBQ0b5wUjjDA',
- '@haileymocaixi': 'UCZqsvi6uhFrbsJcAmcZudcQ',
- '@DaChien': 'UCgDQKFV2rMNzTE8Y3rHMVbQ',
- '@maze0517': 'UCmUumXHn322rcoo9ulg_jvw',
- '@huangjung': 'UCOGk7wgE-xXPMgyw0S7STBA',
- '@hellohorlung': 'UCnXLslDRBPExnUBunhM918Q',
- '@cheapaoe': 'UCGGrblndNzi86WY5lJkQJiA',
- '@loserzun': 'UC6VKHP606ee6ffKwKmBHSig',
- '@user-vx5bd2fn6y': 'UCzjNxGvrqfxL9KGkObbzrmg',
- '@thisgroupofpeople': 'UC6FcYHEm7SO1jpu5TKjNXEA',
- '@WACKYBOYS520': 'UCEfetJrzg6OcXWWuX8uhdhw',
- '@Tw-Universe': 'UCO83NhweFeyT1qotDKS7sWg'

**12/13 update (10 channels)**

- '@TsaiAGaLife': 'UCtcaZ5FUqaNXGX6xhpiGPQA',
- '@three_muggles': 'UC-Es7ozDeMMPy9_jH6uL5TA',
- '@ELTASPORTSHD': 'UCCQvP4hsRW9emj0meGk15jg',
- '@haleyAip4ever': 'UC0NFqwYXVztcVxWqYcfQ4SQ',
- '@coolmantsai': 'UCCSZ1P-Gas8IIgzXAwV0YFA',
- '@WINNIBao': 'UCngJawDcrvMgdiyCe8y7dBg',
- '@63OFFICIAL': 'UCVsm3D-t1w0amKNb_FurUzA',
- '@beryl_lulu': 'UC3o2KSr5rbnttYMkUOowEFA',
- '@cbotaku': 'UCjVVRPc0WRcjO9ARP0mZ6FQ',
- '@godtone777': 'UCmJwfvC-1omeZ_31Cb6vawA',

**12/15 update (11 channels)**

- '@k3okii': 'UCSXb506zUhJOS2r41_gTTAQ',
- '@alizabethlin': 'UC3_8r_y60jCpnBm8JubdEwg',
- '@WuQingFeng': 'UCDBKa4DLm_t3MjUgr_E-RDA',
- '@xcsfwr': 'UCGccecSe71noam2_jKmfQaw',
- '@tainanjosh': 'UCWAgfxHuoVWm4rmqYwoKvlw',
- '@jessetang1113': 'UCK7LdglLCApOTaylxX8hW2Q',
- '@fanamericantime': 'UC2VKL-DkRvXtWkfjMzkYvmw'
- '@goldfishbrain': 'UCTT5gtQU5rX8sUQnZaBqiVw',
- '@Empty1207': 'UCiiS3Rp6cLZs4qmM3lb8H2A',
- '@user-mq3ov4iy1e': 'UCeeZZSeqKz3ZDrO1oxQkxfQ',
- '@atotheda': 'UCQ6dAn-1lkPAnYmP4fnRNAg',

**12/16 update (13 channels)**

- '@liketaitai': 'UCHfY_EOzB1i57hYLSw_rYMg',
- '@SanyuanJAPAN2015': 'UCCBq7s8VOCyek275uvq5lYQ',
- '@papayaclass': 'UCdEpz2A4DzV__4C1x2quKLw',
- '@alisasa_official': 'UCAKJ0tmI_RMXqTgxL_OMfIg',
- '@aottergirls': 'UCAr4MVsPBKjhg5eLDDpbDFg',
- '@huan2322': 'UCpmx8TiMv9yR1ncyldGyyVA' ,
- '@psyman4835': 'UCETKJquzRBMqvRPJru5_SHw',
- '@57History': 'UCHTiZqszKobT-oenbQzKDBg',
- '@user-jc6jo4mw8c': 'UC8_o9aFpknEMck7D43E5Zww',
- '@GQTaiwan': 'UCI1zO6-A3h7DHg-R_x34vLg',
- '@by_ellllllllla': 'UCEbGJMxfB-qMRTdIjUXhw4w',
- '@RockRecordsTaipei': '@RockRecordsTaipei',
- '@imyeahhi': 'UCkA28YTskVsTzh6S5X6As_Q'