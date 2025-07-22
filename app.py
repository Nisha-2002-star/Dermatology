import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from flask import Flask, render_template, flash, request, session, send_file, jsonify
from flask import render_template, redirect, url_for, request
import mysql.connector

# chat initialization
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/AdminLogin")
def AdminLogin():
    return render_template('AdminLogin.html')


@app.route('/NewUser')
def NewUser():
    return render_template('NewUser.html')


@app.route('/UserLogin')
def UserLogin():
    return render_template('UserLogin.html')

@app.route('/Chat')
def Chat():
    return render_template('chat.html')


@app.route('/Predict')
def Predict():
    return render_template('Predict.html')

@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        if request.form['uname'] == 'admin' and request.form['password'] == 'admin':
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='3medicalchatskinnewdb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb")
            data = cur.fetchall()

            return render_template('AdminHome.html', data=data)
        else:
            flash("UserName or Password Incorrect!")

            return render_template('AdminLogin.html')

@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3medicalchatskinnewdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb")
    data = cur.fetchall()
    return render_template('AdminHome.html', data=data)


@app.route("/newuser", methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':
        name = request.form['uname']
        mobile = request.form['mobile']
        email = request.form['email']
        address = request.form['address']
        username = request.form['username']
        password = request.form['password']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='3medicalchatskinnewdb')
        cursor = conn.cursor()
        cursor.execute(
            "insert into regtb values('','" + name + "','" + mobile + "','" + email + "','"+ address +"','" + username + "','" + password + "')")
        conn.commit()
        conn.close()
        flash("Record Saved!")
    return render_template('UserLogin.html')


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['uname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='3medicalchatskinnewdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and password='" + password + "'")
        data = cursor.fetchone()
        if data is None:

            flash("UserName Or Password Incorrect..!")
            return render_template('UserLogin.html', data=data)
        else:
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='3medicalchatskinnewdb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where username='" + username + "' and password='" + password + "'")
            data = cur.fetchall()

            flash("you are successfully logged in")
            return render_template('UserHome.html', data=data)


@app.route("/UserHome")
def UserHome():
    uname = session['uname']
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='3medicalchatskinnewdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb where username='" + uname + "' ")
    data = cur.fetchall()
    return render_template('UserHome.html', data=data)

@app.route("/ask",  methods=['GET', 'POST'])
def ask():

    message = str(request.form['messageText'])

    #msg = request.form["msg"]
    msg = message
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    #return res
    return jsonify({'status':'OK','answer':res})



#@app.route("/get", methods=["POST"])
@app.route("/get")
def chatbot_response():
    msg = request.args.get('msg')
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    return res



# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


'''def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result'''


import random

def getResponse(ints, intents_json):
    # Check if the 'ints' list is empty
    if not ints:
        return "I'm sorry, I didn't understand that."

    # Extract the predicted tag from the first element
    tag = ints[0].get("intent")

    # Extract the list of intents from the JSON data
    list_of_intents = intents_json.get("intents", [])

    # Search for the matching intent and return a random response
    for intent in list_of_intents:
        if intent["tag"] == tag:
            response = intent["responses"]
            print(response)

            precaution = intent.get("Precaution", "No precaution available")
            print(precaution)
            out = response +'<br>'+ precaution
            return out

    return "I'm sorry, I don't have a response for that."


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        file = request.files['file']
        file.save('static/Out/Test.jpg')
        org = 'static/Out/Test.jpg'

        import warnings
        warnings.filterwarnings('ignore')

        import tensorflow as tf
        import numpy as np
        import os
        from keras.preprocessing import image


        classifierLoad = tf.keras.models.load_model('Model/skinmodel.h5')
        test_image = image.load_img('static/Out/Test.jpg', target_size=(150, 150))
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)
        print(result)
        res=''
        out=''
        if result[0][0] == 1:
            print("Acne Skin")
            out = "MildDemented"

            res = "To care for Acne-Prone Skin:<br>1. Wash your face twice daily with a mild cleanser.<br>2. Use oil-free and non-comedogenic skincare and makeup products.<br>3. Avoid excessive scrubbing or harsh exfoliation.<br>4. Maintain a healthy diet and stay hydrated."

        elif result[0][1] == 1:
            print("dry Skin")
            out = "dry Skin"
            res = "To care for Dry Skin:\n1. Use a moisturizer with hyaluronic acid or ceramides.\n2. Avoid alcohol-based skincare products.\n3. Protect your skin from harsh weather by using sunscreen.\n4. Consider using a humidifier to add moisture to the air."

        elif result[0][2] == 1:
            print("normal Skin")
            out = "normal Skin"
            res = "To normal skin:\n1. you don’t have to worry about excess oil or breakouts. \n2.Just keep your routine simple: use a gentle cleanser, a lightweight moisturizer\n 3. And a broad-spectrum sunscreen daily to maintain your skin's healthy\n4. Balanced appearance."
        elif result[0][3] == 1:
            print("oily Skin")
            out = "oily Skin"
            res = "To care for Oily Skin:\n1. Avoid heavy, greasy skincare products.\n2. Use clay masks to help absorb excess oil.\n3. Don't over-wash your face, as it can increase oil production.\n4. Stick to an oil-free, balanced skincare routine."
        return render_template('Predict.html', pre=out, result=res, org=org)




if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

