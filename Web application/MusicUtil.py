import pandas as pd

df = pd.read_csv('data/data_moods.csv')


def getMusicByMood(mood):

    if(mood == "Neutral" or mood =="Fear") :
        return df[df['mood'] == 'Calm']
    elif(mood == "Angry") :
        return df[df['mood'] == 'Energetic']
    else :
        return df[df['mood'] == mood]
        

