
# 🌾 Agricultural PDF Processor

An AI-powered web application for extracting, analyzing, and searching agricultural policy or research PDFs using LLMs (GPT-4o), sentence embeddings, and semantic search.

## 🚀 Features

- 📄 **PDF Processing**: Extracts meaningful text and embedded images from uploaded PDF files.
- 🖼️ **Image Extraction**: Saves images from PDF pages for visual context.
- 🔍 **Smart Semantic Search**: Uses Sentence-BERT and GPT-enhanced queries to retrieve the most relevant content based on meaning, not just keywords.
- 💬 **Text Beautification**: Refines and clarifies text using Azure OpenAI’s GPT-4o.
- 🧠 **Theme Extraction**: Identifies key themes and subthemes from agricultural content and exports them as a Word document.
- 📊 **Search Result Insights**: Shows similarity scores, original vs enhanced text, and related images.
- 💾 **Smart Caching**: Avoids reprocessing PDFs by hashing and storing status in a local SQLite DB.

---

## 🧑‍💻 Tech Stack

| Tool | Purpose |
|------|---------|
| **Streamlit** | Web interface |
| **PyMuPDF (fitz)** | PDF parsing |
| **Azure OpenAI (GPT-4o)** | Text enhancement and theme extraction |
| **Sentence-Transformers** | Embedding generation |
| **scikit-learn** | Cosine similarity for search |
| **SQLite** | Local data storage |
| **python-docx** | Exporting Word documents |
| **Pillow / TQDM / Regex** | Utility functions |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/agricultural-pdf-processor.git
cd agricultural-pdf-processor
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Running Locally

```bash
streamlit run exp.py
```

Visit: [http://localhost:8501](http://localhost:8501)

---

## 🌐 Public Access with Ngrok (Optional)

1. Install `ngrok` and authenticate:
```bash
pip install pyngrok
ngrok config add-authtoken <your_token>
```

2. Run app and start tunnel:
```bash
streamlit run exp.py
ngrok http 8501
```

Copy the public `https://....ngrok.io` URL and share it.

---

## 🔐 Configuration

Add your Azure OpenAI credentials either via Streamlit sidebar or using a secrets file:

### `.streamlit/secrets.toml`
```toml
azure_api_key = "your_azure_api_key"
azure_endpoint = "https://your-resource-name.openai.azure.com/"
```

---

## 🧠 Use Cases

- Agricultural policy research
- Government documentation analysis
- Academic studies in agri-economics
- Semantic retrieval in document archives

---

## 📂 Project Structure

```
.
├── exp.py                      # Main Streamlit app
├── requirements.txt            # Python dependencies
├── extracted_images/           # Auto-created, stores images
├── agricultural_ml.db          # SQLite DB (auto-created)
├── themes_and_subthemes.docx   # Word doc output (per PDF)
└── .streamlit/
    └── secrets.toml            # Azure credentials (optional)
```

---

## 📜 License

MIT License. Free to use, modify, and distribute.

---

## 🤝 Contributions

Pull requests, ideas, and suggestions are welcome!

---

## ✨ Credits

Built by [Your Name] using OpenAI, Streamlit, and modern NLP tools to support better understanding of agricultural documents.


## ✨ WorkFlow

![Image](https://github.com/user-attachments/assets/b7f0b894-1bf5-4f0d-a3eb-2969edf3ea62)
