import os
import string
import time
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import PyPDF2
import re

# Download NLTK data
nltk.download('punkt')

# 🔑 Authenticate Google Drive
def authenticate_google_drive():
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile('client_secrets.json')  # Update with your credentials
    if not gauth.credentials or gauth.access_token_expired:
        gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

# 📥 Download PDFs from Drive
def download_pdfs(drive, folder_id, processed_files):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and mimeType='application/pdf'"}).GetList()
    new_files = []
    for file in file_list:
        if file['id'] not in processed_files:
            file.GetContentFile(file['title'])
            new_files.append(file['title'])
            processed_files.add(file['id'])
    return new_files

# 📄 Extract sections from PDFs (with section extraction)
def extract_sections_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = [page.extract_text() for page in reader.pages]
        full_text = "\n".join(text)
        
    # Define regex patterns for common section titles
    sections = {
        "abstract": r"(?<=\babstract\b)(.*?)(?=\b(?:introduction|methodology|results|conclusion)\b|$)",
        "introduction": r"(?<=\bintroduction\b)(.*?)(?=\b(?:methodology|results|conclusion)\b|$)",
        "methodology": r"(?<=\bmethodology\b)(.*?)(?=\b(?:results|conclusion)\b|$)",
        "results": r"(?<=\bresults\b)(.*?)(?=\b(?:discussion|conclusion)\b|$)",
        "discussion": r"(?<=\bdiscussion\b)(.*?)(?=\b(?:conclusion)\b|$)",
        "conclusion": r"(?<=\bconclusion\b)(.*)"
    }
    
    extracted_sections = {}
    for section, pattern in sections.items():
        match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_sections[section] = match.group(1).strip()
        else:
            extracted_sections[section] = ""
    
    return extracted_sections

# 🧹 Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return " ".join(nltk.word_tokenize(text))

# 🔍 Embed text using Sentence-BERT
def embed_text(text, model):
    return model.encode(text)

# 📦 Process and embed reference papers
def process_reference_papers(drive, publishable_folder_id, model):
    conference_embeddings = {}
    folder_list = drive.ListFile({'q': f"'{publishable_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"}).GetList()
    
    for folder in folder_list:
        conference_name = folder['title']
        conference_embeddings[conference_name] = []
        
        sub_files = drive.ListFile({'q': f"'{folder['id']}' in parents and mimeType='application/pdf'"}).GetList()
        for file in sub_files:
            file.GetContentFile(file['title'])
            sections = extract_sections_from_pdf(file['title'])
            text = " ".join([sections.get('abstract', ''), sections.get('methodology', '')])  # Example: abstract + methodology
            text = preprocess_text(text)
            embedding = embed_text(text, model)
            conference_embeddings[conference_name].append(embedding)
            os.remove(file['title'])
    
    return conference_embeddings

# 🔎 Find most similar conference
def find_best_conference(paper_embedding, conference_embeddings):
    best_conf, best_score = None, -1
    for conference, embeddings in conference_embeddings.items():
        for ref_embedding in embeddings:
            similarity = cosine_similarity([paper_embedding], [ref_embedding])[0][0]
            if similarity > best_score:
                best_conf, best_score = conference, similarity
    return best_conf, best_score

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_justification(text, best_conference):
    # Define max tokens allowed for the model (e.g., 1024 tokens for BART model)
    max_tokens = 1024
    input_length = len(text.split())

    # If the text is too long, truncate it to the max tokens
    if input_length > max_tokens:
        text = ' '.join(text.split()[:max_tokens])

    # Prepare the prompt for summarization
    prompt = f"The paper discusses the following: {text}. Generate a rationale for why the paper belongs to {best_conference} based on the following content:"
    
    try:
        # Generate the justification summary
        summary = summarizer(prompt, max_length=133, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error in summarization: {e}")
        return "Error generating justification."
    
    
# 🔄 Process new papers and classify
def process_new_papers(drive, papers_folder_id, conference_embeddings, model):
    processed_files = set()
    while True:
        new_files = download_pdfs(drive, papers_folder_id, processed_files)
        for file in new_files:
            print(f"\n📄 Processing: {file}")

            # Extract sections from the paper
            sections = extract_sections_from_pdf(file)
            text = " ".join([sections.get('abstract', ''), sections.get('methodology', '')])  # Example: abstract + methodology
            text = preprocess_text(text)
            paper_embedding = embed_text(text, model)

            best_conference, similarity = find_best_conference(paper_embedding, conference_embeddings)
            justification = generate_justification(text, best_conference)
            
            print(f"✅ Recommended Conference: {best_conference} (Similarity: {similarity:.2f})")
            print(f"📝 Justification: {justification}\n")
            
            os.remove(file)  # Clean up
        time.sleep(60)

# 🚀 Main Function
if _name_ == "_main_":
    # Authenticate Google Drive
    drive = authenticate_google_drive()

    # Set Google Drive Folder IDs
    publishable_folder_id = "1RKyDkAyW7cf09THED7Ms4LNBdeTDWnlK"       # Folder ID for 'publishable'
    papers_folder_id = "1Y2Y0EsMalo26KcJiPYcAXh6UzgMNjh4u"             # Folder ID for 'papers'

    # Load Embedding Model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Process and Embed Reference Papers
    print("🔄 Embedding reference papers...")
    conference_embeddings = process_reference_papers(drive, publishable_folder_id, embedding_model)

    # Start Monitoring for New Papers
    print("🚀 Starting real-time paper classification...")
    process_new_papers(drive, papers_folder_id, conference_embeddings, embedding_model)