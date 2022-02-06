from flask import Flask, render_template, request
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification,pipeline
app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():
    if request.method=='POST':
        text = request.form["text_field"]

        test_model = DistilBertForSequenceClassification.from_pretrained("./model_files") ## give path to model files directory
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        nlp = pipeline('sentiment-analysis', model=test_model, tokenizer=tokenizer)

        results = nlp([text])

        pred = None

        if results[0]['label']=="LABEL_1":
            pred = "Positive"
        else:
            pred = "Negative"
        
        return render_template('index.html',show=True,pred=pred) 


    else:
        return render_template('index.html',show=False)


if __name__=="__main__":
    app.run()