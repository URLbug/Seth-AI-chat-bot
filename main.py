import nextcord
import re
import asyncio

from practic import seth
from __init__ import bot, config


@bot.event
async def on_ready():
    print(f'Seth AI - READY')

    stat = nextcord.Game(
        name=f"Hello! I Seth!",
        status=nextcord.Status.idle
        )
    
    await bot.change_presence(status=nextcord.Status.idle, activity=stat)

@bot.slash_command()
async def say_he(interaction: nextcord.Interaction, say_he: str):
    await interaction.response.defer()
    
    try:
        say_he = re.sub(r'[^\w\s]','', say_he)
        say_he = seth(say_he)

        await asyncio.sleep(5)
        
        await interaction.followup.send(say_he)
    except:
        await interaction.followup.send("Sorry, I don't understend u...")


if __name__ == '__main__':
    bot.run(config['TOKEN'])