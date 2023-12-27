NTU ADL Final Project-Line Robot
==
# Environment
## Requirements
```
pip install -r requirements.txt
```

# Run robot
- Open a terminal and run `app.py`, which is the linebot source code
```
python3 app.py
```

## ngrok
1. Download ngrok
2. Run `ngrok.exe` and input
    ```
    ngrok http 5000
    ```
    - This command tells ngrok to create a tunnel on local port 5000. If you're using a different port, replace 5000 with your actual port number.

3. Get the public URL: Once ngrok is running, it will display a public URL like `https://******.ngrok-free.app`.
![image](https://hackmd.io/_uploads/BkdU7QPw6.png)


4. Configure Line Bot Webhook URL: In the Line Developer Console, set the Webhook URL to the ngrok public address, including the /callback path (e.g., **https://******.ngrok-free.app/callback**). This allows the Line platform to forward user messages to your local server.

## Paste Webhook
- set the forwarding url in webhook
![image](https://hackmd.io/_uploads/rJ0hfmPwT.png)

