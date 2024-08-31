from flask import Flask,request,jsonify
from youtube import summarise_youtube
from pine import store,similarity
from  voice import summarise_voice
from quiz import quizmaker
from chatgpt import rungpt
from chatPdf import make_an_store
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})



@app.route('/youtube',methods=["POST"])
def youtube_sumarizer():
    data = request.json
    c = data.get('classId')
    k = data.get('url')
    res = summarise_youtube(k)
    store([res],str(c),{"url":k})
    return jsonify({
        "res":res
    })

@app.route('/voice',methods=["POST"])
def voice_sumarizer():
    data = request.json
    c = data.get('classId')
    k = data.get('text')
    res = summarise_voice(k)
    store([res],str(c),{})
    return jsonify({
        "res":res
    })

@app.route('/search',methods=["POST"])
def finder():
    data = request.json
    c = data.get('classId')
    k = data.get('query')
    res = similarity(text=k,namespace=str(c))
    return jsonify({
        "res":res
    })


@app.route('/quiz',methods=["POST"])
def quiz():
    data = request.json
    c = data.get('classId')
    k = data.get('topic')
    d = data.get('difficulty')
    res = quizmaker(d,c,topic=k)
    return res

@app.route('/chatgpt',methods=["POST"])
def chats():
    data = request.json
    c = data.get('class')
    k = data.get('question')
    res = rungpt(k,c)
    return jsonify({
        "res":res
    })


@app.route('/pdfvector',methods=["POST"])
def pdfv():
    data = request.json
    c = data.get('class')
    k = data.get('url')
    make_an_store(k,c)
    print('Done')
    return jsonify({
      
        "res" :"Done"
    })

@app.route('/')
def print1():
    return "Hello, world!!"

if __name__ == '__main__':
    app.run(debug=True)
   
