from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch

application = Flask(__name__)
app = application

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def huggingFace(text, summary_type):
    # Load the text summarization pipeline
    #summarizer = pipeline("summarization", device=device)
    model_name = "sshleifer/distilbart-cnn-12-6"
    summarizer = pipeline("summarization", model=model_name, revision="a4f8f3e")

    # Split the text into chunks of maximum 600 words
    max_words = 600
    chunks = [text[i:i+max_words] for i in range(0, len(text), max_words)]

    summaries = []

    # Generate the summary based on the specified type

    if summary_type == "short":
        
        # Process each chunk and generate a summary
        for chunk in chunks:
            summary = summarizer(chunk, max_length=20, min_length=10, num_beams=2, length_penalty = 0.8, do_sample=False)[0]["summary_text"]
            summaries.append(summary)

        # Combine the summaries into a single string
        summary = " ".join(summaries)
    elif summary_type == "long":
       
        # Process each chunk and generate a summary
        for chunk in chunks:
            summary = summarizer(chunk, max_length=40, min_length=10, num_beams=2, do_sample=False)[0]["summary_text"]
            summaries.append(summary)

        # Combine the summaries into a single string
        summary = " ".join(summaries)

    return summary

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/summary', methods=['POST'])
def summary():
    if request.method == 'POST':
        article_text = request.json['articleText']
        summary_type = request.json['summaryType']
        
        if summary_type == "short" or summary_type == "long":
            summarized_text = huggingFace(article_text, summary_type)
            return jsonify({'summarized_text': summarized_text})
        else:
            return jsonify({'error': 'Invalid summary type'})

if __name__ == '__main__':
    app.run(debug=True)


