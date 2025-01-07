import re
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
import textstat
import pdfplumber
from transformers import BertTokenizer, BertModel
from PyPDF2 import PdfReader

# Conference themes for relevance evaluation
conference_themes = {
    "CVPR": "Computer Vision, image processing, deep learning for vision, object detection, segmentation",
    "EMNLP": "Natural language processing, semantic understanding, machine translation, text generation",
    "KDD": "Data mining, knowledge discovery, big data, data science, analytics",
    "NeurIPS": "Machine learning, deep learning, reinforcement learning, AI, optimization, theory",
    "TMLR": "Theoretical machine learning, foundational ML research, statistical learning"
}

# Section synonyms
SECTION_SYNONYMS = {
    "abstract": ["abstract", "summary"],
    "methodology": ["methodology", "approach", "methods", "experimental setup", "methodologies", "Method"],
    "results": ["results", "findings", "outcome"]
}

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = [page.extract_text() for page in reader.pages]
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def calculate_relevance(abstract, theme_description):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([abstract, theme_description], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

def evaluate_relevance_all_conferences(abstract):
    relevance_scores = {}
    for conference, theme_description in conference_themes.items():
        relevance_scores[conference] = calculate_relevance(abstract, theme_description)
    return relevance_scores

def normalize_scores(scores):
    scaler = MinMaxScaler()
    values = list(scores.values())
    reshaped_values = [[v] for v in values]
    normalized = scaler.fit_transform(reshaped_values)
    return {k: normalized[i][0] for i, k in enumerate(scores.keys())}

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

def get_bert_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def compute_similarity_with_bert(section1, section2):
    emb1 = get_bert_embeddings(section1)
    emb2 = get_bert_embeddings(section2)
    return cosine_similarity(emb1, emb2)[0][0]

def calculate_readability_score(text):
    raw_score = textstat.flesch_reading_ease(text)
    normalized_score = (raw_score - 0) / (121 - 0)  # Flesch Reading Ease scores range from 0 to 121
    return max(0, min(1, normalized_score))  # Ensures score is between 0 and 1

# Main function to compute all feature scores
def normalize_equation_count(equation_count, max_equation_count):
    """Normalizes equation count to a value between 0 and 1."""
    return min(equation_count / max_equation_count, 1.0) if max_equation_count > 0 else 0

def evaluate_pdf(pdf_path, max_equation_count=100):
    results = {}
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("No text could be extracted.")
        return

    # Relevance scores
    sections = extract_sections(text)
    abstract = sections.get("abstract", "")
    # if abstract:
    #     raw_relevance_scores = evaluate_relevance_all_conferences(abstract)
    #     results["relevance_scores"] = normalize_scores(raw_relevance_scores)

    # Equation count (normalized)
    equation_count = count_equations(pdf_path)
    results["equation_count"] = normalize_equation_count(equation_count, max_equation_count)

    # Section similarity
    methodology = sections.get("methodology", "")
    results_section = sections.get("results", "")
    if abstract and methodology:
        results["abstract_methodology_similarity"] = compute_similarity_with_bert(abstract, methodology)
    if abstract and results_section:
        results["abstract_results_similarity"] = compute_similarity_with_bert(abstract, results_section)
    if methodology and results_section:
        results["methodology_results_similarity"] = compute_similarity_with_bert(methodology, results_section)

    # Readability score
    results["readability_score"] = calculate_readability_score(text)

    return results


# Example usage
pdf_path = r"C:\Users\Radhika\Downloads\labelled papers\R002.pdf"
evaluation_results = evaluate_pdf(pdf_path)
for feature, score in evaluation_results.items():
    print(f"{feature}: {score}")
