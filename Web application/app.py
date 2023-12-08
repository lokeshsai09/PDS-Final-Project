import flask
from flask import Flask,Response,render_template
from StreamService import Stream,getSongs
#from CK48StreamService import Stream,getSongs
from MusicUtil import getMusicByMood
import cv2

#from StreamService import Stream
app = Flask(__name__)
songs = getMusicByMood("Angry")
@app.route("/")
def home():
    return render_template("index.html",data = songs)

@app.route("/video")
def video():
    return Response(Stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/songs")
def getSongstoUI():
    songs = getSongs()
    songs = songs.head(15)
    return songs.to_json(orient='records')
if __name__ == '__main__':
    app.run(host='0.0.0.0')
    app.static_folder = 'static'
