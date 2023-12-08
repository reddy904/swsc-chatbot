import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import re
# from tensorflow.keras.models import load_model
from keras.models import load_model
import json
import random
import mysql.connector
from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

# Load the pre-trained model and other necessary files
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
bot_name = "SWSC_Bot"
lemmatizer = WordNetLemmatizer()

def cleaning(sentence):
    sentence= sentence.lower()
    sentence = re.sub(r"i'm","i am",sentence)
    sentence = re.sub(r"he's","he is",sentence)	
    sentence = re.sub(r"she's","she is",sentence)	
    sentence = re.sub(r"that's","that is",sentence)
    sentence = re.sub(r"what's","what is",sentence)	
    sentence = re.sub(r"where's","where is",sentence)		
    sentence = re.sub(r"\'ll","will",sentence)	
    sentence = re.sub(r"\'ve","have",sentence)	
    sentence = re.sub(r"\'re","are",sentence)	
    sentence = re.sub(r"\'d","will",sentence)	
    sentence = re.sub(r"won't","will not",sentence)	 
    sentence = re.sub(r"can't","cannot",sentence)	
    sentence = re.sub(r"[-()\"#/@;:<>=|.?,]","",sentence)
    sentence_words = nltk.word_tokenize(sentence)

    filter_word = list(filter(lambda x: x in classes or words, sentence_words))
    print("###########_______###############---------------"+str(filter_word)+"______________##########################")
    return filter_word
#--------

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words=cleaning(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"+str(res)+"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # ERROR_THRESHOLD = 0.20

    results = [[i,r] for i,r in enumerate(res)]     #=> results=[[i,r],[i,r]....]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)  # x=[i,r]

    print("========="+str(results[0])+"==============================================")
    return_list = []
    # print("****-----------------------"+classes[86]+"-----------------------********")
    return_list.append({"intent": classes[results[0][0]], "probability": str(results[0][1])})
    print("++++++++++++++++++++"+str(return_list)+"++++++++++++++++++++++++++") 
    # for r in results[0]:
    #     return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # print("++++++++++++++++++++"+str(return_list)+"++++++++++++++++++++++++++")    
    return return_list

def getResponse(ints, intents_json,tagging=False):
    if tagging == True:
        tag = ints
    else:
        tag = ints[0]['intent']
    
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(text):  
    if text in classes:
        res = getResponse(text,intents,tagging=True)
        print("This is my response==================>"+str(res))
        return res
    else:    
        filter_word = cleaning(text)
        for word in filter_word:
            if lemmatizer.lemmatize(word) in classes: 
                word = lemmatizer.lemmatize(word)
            if (word in classes) :
                print("This is my response==================>"+str(word))
                res = getResponse(word,intents,tagging=True)
                return res

    ints = predict_class(text, model)  #ints=[{'tags':"greeting",'probability':"ihidhi"}]
    # ints=>highest probability ==>tags,probability
    # print("-------------------------------------"+ str(type(ints[0]['probability']))+"-----------------------------------------")
    # print(ints[0])  #==>dicionary of pattern class with high probability
    prob=float(ints[0]['probability']) #filtering the highest
    print(type(prob))
    if prob > 0.75:
        res = getResponse(ints, intents) 
    else:
        res="Hey, I'm only a bot, I need things simple.Could you please place query more detailly or Exclude slang words?,Thank you"  

        mydb = mysql.connector.connect(host="localhost", user="root", passwd="",database="chatbot_query")

        mycursor = mydb.cursor()

        query = f"INSERT INTO new_query (Query)  VALUES ('{text}')"

        mycursor.execute(query)
        mydb.commit()
        mydb.close()    
    return res

from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=['POST'])
def get_bot_response():
    data = request.data.decode('utf-8')
    # intents = predict_class(data, model)
    response = getResponse(intents, intents, tagging=True)
    res = chatbot_response(data)
    print(res)

    return f"<i>Intents:<i>{intents}</i>\n\n\t <b> Response: {response} ress:{res}</b>"

if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0')