import os
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pdfplumber

# Authenticate Google Drive
def authenticate_google_drive():
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile('client_secrets.json')  # Update with your credentials
    if not gauth.credentials or gauth.access_token_expired:
        gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

# Download PDFs from a folder and subfolders
def download_pdfs(drive, folder_id, processed_files):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents"}).GetList()
    pdf_files = []

    for file in file_list:
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            # Recursively fetch PDFs from subfolders
            pdf_files += download_pdfs(drive, file['id'], processed_files)
        elif file['mimeType'] == 'application/pdf' and file['id'] not in processed_files:
            file.GetContentFile(file['title'])
            pdf_files.append(file['title'])
            processed_files.add(file['id'])
    
    return pdf_files

# Extract text from a PDF
def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

# Count equations in the PDF
def count_equations(text):
    equation_patterns = [
        r'\$.+?\$', r'\\begin\{equation\}.*?\\end\{equation\}', r'\\[a-zA-Z]+\{.*?\}',
        r'[a-zA-Z]+\s*=\s*[^\\\s]+', r'[^a-zA-Z]\d+\s*=\s*\d+'
    ]
    return sum(len(re.findall(pattern, text)) for pattern in equation_patterns)

# Calculate readability score
def calculate_readability_score(text):
    try:
        score = flesch_reading_ease(text)
        normalized = max(0, min(1, (score - 0) / (121 - 0)))
        return normalized
    except Exception as e:
        print(f"Error calculating readability score: {e}")
        return 0

# Calculate feature scores for a PDF
def calculate_features(pdf_path, max_equation_count=100):
    text = extract_text_from_pdf(pdf_path)
    equation_count = count_equations(text)
    readability = calculate_readability_score(text)
    normalized_equation_count = min(equation_count / max_equation_count, 1.0) if max_equation_count > 0 else 0
    return [normalized_equation_count, readability]

# Train logistic regression model
def train_model(features, labels):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    model = LogisticRegression()
    model.fit(features, labels)
    return model, scaler

# Predict labels for unlabeled data
def predict_labels(model, scaler, features):
    features = scaler.transform(features)
    return model.predict(features)

# Main workflow
def main():
    drive = authenticate_google_drive()
    folder_id = '1Z8z4craj36ighb8hzUzeM76OOgpUdsKr'  # Root folder ID
    
    processed_files = set()

    # Fetch labeled papers
    publishable_folder_id = 'your_publishable_folder_id'  # Replace with actual folder ID
    non_publishable_folder_id = 'your_non_publishable_folder_id'  # Replace with actual folder ID

    labeled_features = []
    labels = []

    # Process Publishable folders
    publishable_files = download_pdfs(drive, publishable_folder_id, processed_files)
    for file in publishable_files:
        features = calculate_features(file)
        labeled_features.append(features)
        labels.append(1)  # Label for publishable

    # Process Non-Publishable folder
    non_publishable_files = download_pdfs(drive, non_publishable_folder_id, processed_files)
    for file in non_publishable_files:
        features = calculate_features(file)
        labeled_features.append(features)
        labels.append(0)  # Label for non-publishable

    # Train logistic regression model
    model, scaler = train_model(np.array(labeled_features), np.array(labels))

    # Fetch unlabeled papers
    papers_folder_id = 'your_papers_folder_id'  # Replace with actual folder ID
    unlabeled_files = download_pdfs(drive, papers_folder_id, processed_files)
    
    unlabeled_features = []
    paper_names = []

    for file in unlabeled_files:
        features = calculate_features(file)
        unlabeled_features.append(features)
        paper_names.append(os.path.basename(file))

    # Predict labels for unlabeled papers
    predictions = predict_labels(model, scaler, np.array(unlabeled_features))

    # Save results to CSV
    results_df = pd.DataFrame({'Paper Name': paper_names, 'Label': predictions})
    results_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()
