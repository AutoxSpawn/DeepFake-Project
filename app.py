from flask import Flask, request, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import openai
import os
from dotenv import load_dotenv
from newspaper import Article  # Extracts news from URLs

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key securely
openai_api_key = os.getenv("OPENAI_API_KEY")

# 🔹 Check if API Key is Missing (Throw Error)
if not openai_api_key:
    raise ValueError("⚠️ OpenAI API Key is missing! Add it to your .env file.")

# Set API key
openai.api_key = openai_api_key

# Initialize Flask App
app = Flask(__name__)

# Load Pretrained Fake News Classification Model
model_name = "lvwerra/bert-imdb"  # Pretrained model for text classification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def extract_text_from_url(url):
    """Extract the main article content from a news URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return None  # If extraction fails, return None

def gpt_explain_fake_news(text, initial_prediction):
    """Use GPT-4o to verify claims, consider future events, and override classification if needed."""
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are an AI fact-checker with knowledge up to March 2025. "
                    "Read the article carefully and fact-check its claims against known political and historical events. "
                    "⚠️ If the article discusses FUTURE POSSIBILITIES, POLITICAL SPECULATION, OR ANALYSIS, do NOT assume it is fake. "
                    "⚠️ If new political figures are mentioned, assume they may have been elected after October 2023. "
                    "If after verification, you determine the claims are not supported by credible sources, classify the article as 'Fake News'."
                )},
                {"role": "user", "content": f"Fact-check and explain this news article: {text}"}
            ]
        )
        
        explanation = response.choices[0].message.content

        # 🔹 Correct the classification based on new criteria
        if "Fake News" in explanation or "Not Real News" in explanation:
            corrected_prediction = "Fake News"
        else:
            corrected_prediction = "Real News"

        return corrected_prediction, explanation
    except openai.OpenAIError as e:
        return initial_prediction, f"Error: {str(e)}"


def detect_fake_news(text):
    """Classify text as Real or Fake News and get an explanation from GPT-4o."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    labels = ["Fake News", "Real News"]
    confidence = probs.max().item() * 100
    initial_prediction = labels[probs.argmax()]

    # Get fact-checking explanation & allow GPT-4o to override the prediction
    corrected_prediction, gpt_explanation = gpt_explain_fake_news(text, initial_prediction)

    return {
        "Prediction": corrected_prediction,  # Now uses GPT-4o's corrected decision
        "Confidence": f"{confidence:.2f}",
        "Explanation": gpt_explanation
    }

#############################################################################################################
app = Flask(__name__, template_folder="Website/templates", static_folder="Website/static")

@app.route("/", methods=["GET", "POST"])
def home():
    """Webpage for Fake News Detection (Supports Text and URLs)."""
    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        url = request.form.get("news_url", "").strip()

        if url:  # If a URL is provided, extract its text
            extracted_text = extract_text_from_url(url)
            if extracted_text:
                result = detect_fake_news(extracted_text)
                return render_template("home.html", prediction=result["Prediction"], confidence=result["Confidence"], text=extracted_text, explanation=result["Explanation"], url=url)
            else:
                return render_template("home.html", error="Could not extract text from the URL.", url=url)

        if text:  # If text is provided, analyze it
            result = detect_fake_news(text)
            return render_template("home.html", prediction=result["Prediction"], confidence=result["Confidence"], text=text, explanation=result["Explanation"])

        return render_template("home.html", error="Please enter news text or a URL.")

    return render_template("home.html", prediction=None)

@app.route("/loading")
def loading():
    return render_template("loading.html")

if __name__ == "__main__":
    app.run(debug=True)











