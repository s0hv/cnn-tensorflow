import json
import requests
import os


with open('emotes.json', 'r', encoding='utf-8') as f:
    em = json.load(f)

data = 'data'
url = 'https://cdn.discordapp.com/emojis/%s.png?v=1'
for emotes in em.values():
    for emote, emote_id in emotes['emotes']:
        r = requests.get(url % emote_id, stream=True)
        folder = os.path.join(data, emote_id)
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(os.path.join(folder, emote + '.png'), 'wb') as f:
            for chunk in r.iter_content(2048):
                f.write(chunk)
