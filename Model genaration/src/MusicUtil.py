import pandas as pd

df = pd.read_csv('data-clean/data_moods.csv')


def getMusicByMood(mood):

    if(mood == "Neutral" or mood =="Fear") :
        return df[df['mood'] == 'Calm']['name'].to_list()
    elif(mood == "Angry") :
        return df[df['mood'] == 'Energetic']['name'].to_list()
    else :
        return df[df['mood'] == mood]['name'].to_list()
        

