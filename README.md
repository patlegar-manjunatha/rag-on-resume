# Resume RAG Chatbot 
Live App : [Open Streamlit App](https://rag-on-resume.streamlit.app/)


A simple RAG-based chatbot that answers questions using resume data.

## Features

- Resume PDF text extraction
- Chunking structured resume sections
- OpenAI embeddings generation
- FAISS vector search
- LLM-based answer generation
- Streamlit chat interface

---

## Tech Stack

- Python
- Streamlit
- FAISS
- OpenAI Embeddings
- Groq API
- PyPDF

---

## Project Structure

```bash
.
├── data/
│   └── resume.pdf
├── prompts/
│   └── system_prompt.txt
├── main.py
├── streamlit_app.py
├── requirements.txt
└── .env.example
````

---

## How It Works

1. Resume PDF is read and converted to text
2. Text is split into chunks
3. Embeddings are generated for each chunk
4. Embeddings are stored in FAISS
5. Relevant chunks are retrieved for user queries
6. LLM generates answers using retrieved context

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add API keys

Create a `.env` file:

```env
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
```

### Run the app

```bash
streamlit run streamlit_app.py
```

---

## Sample Queries

* Tell me about your projects
* What are your skills?
* What certifications do you have?
* What are you currently working on?

---

## Notes

* Simple and lightweight RAG pipeline
* Uses FAISS for retrieval
* Answers are generated only from resume context
