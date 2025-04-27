import json
import nltk
import os

from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from openai import OpenAI

load_dotenv()

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
domain_stopwords = {"model", "study", "effect", "analysis", "data", "paper", "university", "introduction", "test", "warsaw", "research", "article", "country", "result"}
stop_words.update(domain_stopwords)
lemmatizer = WordNetLemmatizer()

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI"))

def preprocess_all_papers(author_dir='preprocessed_data'):
    save_dir = 'preprocessed_data_test'
    os.makedirs(save_dir, exist_ok=True)

    for author_file in os.listdir(author_dir):
        data = []
        data_processed = []
        with open(os.path.join(author_dir, author_file), "r", encoding="utf-8") as file_src:
            if not os.path.exists(os.path.join(save_dir, author_file)):
                data = json.load(file_src)
                for paper in data:
                    if paper['title']:
                        paper['title'] = preprocess_text(paper['title'])
                    if paper['abstract']:
                        paper['abstract'] = preprocess_text(paper['abstract'])
                        if len(paper['abstract'].split(" ")) <= 15:
                            print(paper['abstract'])
                            continue
                    if paper['keywords']:
                        paper['keywords'] = preprocess_text(paper['keywords'])
                    data_processed.append(paper)
                with open(os.path.join(save_dir, author_file), "w", encoding="utf-8") as file_dest:
                    json.dump(data_processed, file_dest, indent=4, ensure_ascii=False)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    processed_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalnum() and not token.isdigit() and token not in stop_words
    ]
    return ' '.join(processed_tokens)

def translate_text(text, source_language="Polish", target_language="English"):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator specializing in translating {source_language} to {target_language}. If text is already in {target_language}, please return unchanged text with no other comments or remarks."
                },
                {
                    "role": "user",
                    "content": f"Translate the following text from {source_language} to {target_language}:\n```\n{text}\n```"
                },
            ],
        )
        translation = response.choices[0].message.content
        return translation.strip()
    except Exception as e:
        return f"Error: {e}"

