import os
import re
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import textstat
import pdfplumber
from transformers import BertTokenizer, BertModel
from PyPDF2 import PdfReader

# Section synonyms
SECTION_SYNONYMS = {
    "abstract": ["abstract", "summary"],
    "methodology": ["methodology", "approach", "methods", "experimental setup", "methodologies", "Method", "introduction"],
    "results": ["results", "findings", "outcome", "conclusion", "main result"]
}

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = [page.extract_text() for page in reader.pages]
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# Function to calculate relevance score
def calculate_relevance(abstract, theme_description):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([abstract, theme_description], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

# Function to count equations in PDF
def count_equations(pdf_path):
    equation_count = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                equation_patterns = [
                    r'\$.?\$', r'\\begin\{equation\}.?\\end\{equation\}', r'\\[a-zA-Z]+\{.*?\}',
                    r'[a-zA-Z]+\s*=\s*[^\s]+', r'[^a-zA-Z]\d+\s*=\s*\d+'
                ]
                for pattern in equation_patterns:
                    equation_count += len(re.findall(pattern, text))
    except Exception as e:
        print(f"Error counting equations: {e}")
    return equation_count

# Function to extract sections from text
def extract_sections(text):
    sections = {}
    lines = text.split("\n")
    current_section = None
    section_content = []

    for line in lines:
        for section, synonyms in SECTION_SYNONYMS.items():
            if any(synonym.lower() in line.lower() for synonym in synonyms):
                if current_section:
                    sections[current_section] = " ".join(section_content).strip()
                current_section = section
                section_content = []
                break
        else:
            if current_section:
                section_content.append(line)
    if current_section:
        sections[current_section] = " ".join(section_content).strip()
    return sections

# Function to get BERT embeddings
def get_bert_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Function to compute similarity with BERT
def compute_similarity_with_bert(section1, section2):
    emb1 = get_bert_embeddings(section1)
    emb2 = get_bert_embeddings(section2)
    return cosine_similarity(emb1, emb2)[0][0]

# Function to compute TF-IDF similarity
def compute_tfidf_similarity(section1, section2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([section1, section2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Function to calculate readability score
def calculate_readability_score(text):
    raw_score = textstat.flesch_reading_ease(text)
    normalized_score = (raw_score - 0) / (121 - 0)  # Flesch Reading Ease scores range from 0 to 121
    return max(0, min(1, normalized_score))  # Ensures score is between 0 and 1

# Function to normalize equation count
def normalize_equation_count(equation_count, max_equation_count):
    """Normalizes equation count to a value between 0 and 1."""
    return min(equation_count / max_equation_count, 1.0) if max_equation_count > 0 else 0

# Function to count paragraphs in a section
def count_paragraphs(section_text):
    paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
    return len(paragraphs)

# Main function to compute all feature scores
def evaluate_pdf(pdf_path, max_equation_count=100):
    results = {}
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("No text could be extracted.")
        return

    # Extract sections from the PDF
    sections = extract_sections(text)
    abstract = sections.get("abstract", "")
    methodology = sections.get("methodology", "")
    results_section = sections.get("results", "")

    # Equation count (normalized)
    equation_count = count_equations(pdf_path)
    results["equation_count"] = normalize_equation_count(equation_count, max_equation_count)

    # Section similarity (BERT)
    if abstract and methodology:
        results["abstract_methodology_similarity_bert"] = compute_similarity_with_bert(abstract, methodology)
    if abstract and results_section:
        results["abstract_results_similarity_bert"] = compute_similarity_with_bert(abstract, results_section)
    if methodology and results_section:
        results["methodology_results_similarity_bert"] = compute_similarity_with_bert(methodology, results_section)

    # Section similarity (TF-IDF)
    if abstract and methodology:
        results["abstract_methodology_similarity_tfidf"] = compute_tfidf_similarity(abstract, methodology)
    if abstract and results_section:
        results["abstract_results_similarity_tfidf"] = compute_tfidf_similarity(abstract, results_section)
    if methodology and results_section:
        results["methodology_results_similarity_tfidf"] = compute_tfidf_similarity(methodology, results_section)

    # Readability score
    results["readability_score"] = calculate_readability_score(text)

    # Word count of individual sections
    results["abstract_word_count"] = len(abstract.split())
    results["methodology_word_count"] = len(methodology.split())
    results["results_word_count"] = len(results_section.split())

    # Paragraph count under methodology
    results["methodology_paragraph_count"] = count_paragraphs(methodology)

    return results

# Function to process PDFs and extract features
def process_pdfs(folder_path, label=None, max_equation_count=100):
    """Extract features from all PDFs in the specified folder."""
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            features = evaluate_pdf(pdf_path, max_equation_count)
            if features:
                features["paper_name"] = os.path.splitext(file_name)[0]
                if label is not None:
                    features["label"] = label
                data.append(features)
    return data

# Paths
train_publishable_path = r"C:\Users\Radhika\OneDrive\Desktop\Kshitij\train data\Publishable"
train_non_publishable_path = r"C:\Users\Radhika\OneDrive\Desktop\Kshitij\train data\Non-Publishable"
test_data_path = r"C:\Users\Radhika\OneDrive\Desktop\Kshitij\test data"

# Max equation count for normalization
max_equation_count = 100

# Process training data
train_publishable = process_pdfs(train_publishable_path, label=1, max_equation_count=max_equation_count)
train_non_publishable = process_pdfs(train_non_publishable_path, label=0, max_equation_count=max_equation_count)
train_data = train_publishable + train_non_publishable

# Convert to DataFrame
train_df = pd.DataFrame(train_data)

# Prepare features and labels
X_train = train_df[["equation_count", "abstract_methodology_similarity_bert", "abstract_results_similarity_bert",
                    "methodology_results_similarity_bert", "abstract_methodology_similarity_tfidf",
                    "abstract_results_similarity_tfidf", "methodology_results_similarity_tfidf", "readability_score",
                    "abstract_word_count", "methodology_word_count", "results_word_count", "methodology_paragraph_count"]]
y_train = train_df["label"]

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Impute missing values in X_train_scaled using the mean strategy
imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Process test data
test_data = process_pdfs(test_data_path, label=None, max_equation_count=max_equation_count)
test_df = pd.DataFrame(test_data)

# Prepare features for testing
X_test = test_df[["equation_count", "abstract_methodology_similarity_bert", "abstract_results_similarity_bert",
                  "methodology_results_similarity_bert", "abstract_methodology_similarity_tfidf",
                  "abstract_results_similarity_tfidf", "methodology_results_similarity_tfidf", "readability_score",
                  "abstract_word_count", "methodology_word_count", "results_word_count", "methodology_paragraph_count"]]
X_test_scaled = scaler.transform(X_test)

# Impute missing values in X_test_scaled using the mean strategy
X_test_scaled = imputer.transform(X_test_scaled)

# Predict labels for test data
test_df["predicted_label"] = model.predict(X_test_scaled)

# Save results to CSV
output_file = "evaluation_results3.csv"
test_df[["paper_name", "predicted_label"]].to_csv(output_file, index=False)

print(f"Results saved to '{output_file}'.")
