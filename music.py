import spotipy
import spotipy.util as util
import os
from random import randint
import json
import webbrowser
#variables for music player
username="dludfugf17jl42f9qvlbkjbnc"
scope = 'user-library-read'
client_id="Enter your own client id"
client_secret="Enter your own client secret "
redirect_uri="https://google.com"
happy=["alt-rock","happy","hard-rock"]
sad=["sad","death-metal","black-metal"]
neutral=["dancehall","holidays","honky-tonk"]
dictSongsGenre={'Happy':happy,'Sad':sad,'Neutral':neutral}

def playMusic(song_type):
    try:
        token=util.prompt_for_user_token(username,scope,client_id=client_id,client_secret=client_secret,redirect_uri=redirect_uri)
    except:
        os.remove(".cache-{username}")
        token=util.prompt_for_user_token(username,scope,client_id=client_id,client_secret=client_secret,redirect_uri=redirect_uri)
    sp = spotipy.Spotify(auth=token)
    #genre=dictSongsGenre[song_type][randint(0,len(dictSongsGenre[song_type]))]
    songList=sp.recommendations(seed_genres=dictSongsGenre[song_type])
   # print(sp.devices()) requires premium membership
    webbrowser.open(songList['tracks'][randint(0,10)]['external_urls']['spotify'])
    #print(len(songList['tracks']))
    print(json.dumps(songList['tracks'][randint(0,len(songList['tracks']))]['external_urls']['spotify'],sort_keys=True,indent=4))

#playMusic(input("enter the genre:"))


