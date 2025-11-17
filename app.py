import streamlit as st
import pickle
import docx
import PyPDF2
import re
import os

# Get the folder of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ML model and encoder
svc_model = pickle.load(open(os.path.join(BASE_DIR, 'clf.pkl'), 'rb'))
tfidf = pickle.load(open(os.path.join(BASE_DIR, 'tfidf.pkl'), 'rb'))
le = pickle.load(open(os.path.join(BASE_DIR, 'encoder.pkl'), 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Handle uploaded file
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

# Predict resume category
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# Streamlit app
def main():
    st.set_page_config(page_title="Resume Category Prediction - Week 2", layout="wide")
    st.title("Resume Category Prediction - Week 2")
    st.markdown("Upload a resume (PDF, DOCX, TXT) to predict the job category using ML.")

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf","docx","txt"])
    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("Text successfully extracted from resume!")
            
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"**{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
