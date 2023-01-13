import nltk
import json
import nextcord

import pandas as pd

from nextcord.ext import commands


nltk.download('stopwords')

dataset = pd.read_json(r'./dataset/dataset.json')

config = json.load(open('config.json', 'rb'))

bot = commands.Bot()

