import os
import faiss
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

# loading env things
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


# openai client for embeddings
try:
    embedding_client = OpenAI(api_key=openai_api_key)
except Exception as e:
    print("embedding client issue : ", e)
    exit()

# groq client for chatting
try:
    groq_client = Groq(api_key=groq_api_key)
except Exception as e:
    print("groq client issue : ", e)
    exit()


# loading prompt
try:
    with open("prompts/system_prompt.txt", "r", encoding="utf-8") as p:
        SYSTEM_PROMPT = p.read()
except Exception as e:
    print("prompt loading issue : ", e)
    SYSTEM_PROMPT = """
    You are a helpful resume chatbot.
    """


# reading pdf
def load_resume_pdf(pdf_path):
    text = ""
    
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print("pdf reading error : ", e)
    return text

# making chunks 
def make_chunks(text):
    headings = [
        "THE JOURNEY & PROFESSIONAL PHILOSOPHY",
        "PROFESSIONAL STRENGTHS",
        "AREAS FOR GROWTH & CONTINUOUS LEARNING",
        "ACADEMIC INTERNSHIPS",
        "PROJECT PORTFOLIO",
        "CURRENT FOCUS & ACTIV E BUILDS",
        "EDUCATION",
        "CERTIFICATIONS"
    ]
    
    chunks = []
    chunk = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line in headings:
            if chunk:
                chunks.append(chunk.strip())
            chunk = line
        else:
            chunk += "\n" + line
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Making embeddings
def make_embeddings(chunks):
    all_embeddings = []
    
    for chunk in chunks:
        try:
            response = embedding_client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embedding = response.data[0].embedding
            all_embeddings.append(embedding)
            
        except Exception as e:
            print("embedding failed : ", e)
    return all_embeddings


# storing to faiss
def create_vector_store(embeddings):
    embedding_array = np.array(embeddings).astype("float32")
    dimension = embedding_array.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_array)
    return index


# retrieving similar chunks
def retrieve_chunks(user_query, index, chunks, top_k=3):
    try:
        query_embedding = embedding_client.embeddings.create(
            model="text-embedding-3-small",
            input=user_query
        )

        query_vector = np.array(
            [query_embedding.data[0].embedding]
        ).astype("float32")

        distances, indices = index.search(query_vector,top_k)
        retrieved_chunks = []
        for idx in indices[0]:
            retrieved_chunks.append(chunks[idx])
        return retrieved_chunks

    except Exception as e:
        print("retrieval issue : ", e)
        return []

# rag response functio
def generate_chat_response(user_input, chat_history, index, chunks):
    try:
        retrieved_chunks = retrieve_chunks(user_input, index, chunks)
        context = "\n\n".join(retrieved_chunks)

        messages = [
            {
                "role": "system",
                "content": f"""
{SYSTEM_PROMPT}

Resume Context:
{context}
"""
            }
        ]

        # history
        for msg in chat_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        messages.append({
            "role": "user",
            "content": user_input
        })

        # groq response
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4
        )

        final_answer = response.choices[0].message.content
        return final_answer

    except Exception as e:
        print("error generating chat response : ", e)
        return "some error happened while generating answer"


# loading everything 
resume_text = load_resume_pdf("data/resume.pdf")
resume_chunks = make_chunks(resume_text)
chunk_embeddings = make_embeddings(resume_chunks)
vector_index = create_vector_store(chunk_embeddings)