import base64
import requests
import speech_recognition as sr
from groq import Groq
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# API keys and Tokens
GROQ_API_KEY = "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"
GITHUB_TOKEN = "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"
YOUTUBE_API_KEY = "AIzaSyCsDPZGaNlmvP9vzNTdJ8w6x2_D8pgFirA"

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to search YouTube videos
def search_youtube(query, max_results=10):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
        "order": "relevance"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        video_links = []
        for item in data['items']:
            video_id = item['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_links.append(video_url)
        return video_links
    else:
        print(f"Failed to retrieve videos. Status code: {response.status_code}")
        return None

# Function to search GitHub repositories
def search_github_repos(query, max_results=5):
    url = "https://api.github.com/search/repositories"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_results}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()['items']
    else:
        print(f"GitHub API request failed. Status code: {response.status_code}")
        return []

# Function to process PDF
def load_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# Chunking function for document text
def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

# Embedding and vector store creation
def create_vector_store(splits):
    model_name = "BAAI/bge-small-en"
    hf_embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    vectorstore = FAISS.from_documents(documents=splits, embedding=hf_embeddings)
    return vectorstore.as_retriever()

# Formatting function for docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to initialize the RAG chain
def initialize_rag_chain(retriever):
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# Function to handle speech input using SpeechRecognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for speech input...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized speech: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand the audio")
        except sr.RequestError as e:
            print(f"Error with the speech recognition service: {e}")
        return None

# Multimodal Assistant Function
def multimodal_assistant(user_input=None, image_path=None):
    # Case 1: Only Text Input or Speech Input
    if user_input is None:
        user_input = recognize_speech()

    if not image_path:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="llama-3.2-11b-vision-preview"
        )
        return chat_completion.choices[0].message.content
    
    # Case 2: Text with Image Input
    else:
        base64_image = encode_image(image_path)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            model="llama-3.2-11b-vision-preview"
        )
        return chat_completion.choices[0].message.content

# Running the Assistant
if __name__ == "__main__":
    input_method = input("Choose input method - 'text', 'speech', or 'image': ").strip().lower()

    if input_method == "text":
        user_input = input("Enter your message or search query: ")
        image_path = input("Enter the image path (if any, leave blank if none): ")

        # If there's an image, pass both inputs to the assistant
        if image_path:
            result = multimodal_assistant(user_input, image_path)
        else:
            result = multimodal_assistant(user_input)
    elif input_method == "speech":
        result = multimodal_assistant()
    elif input_method == "image":
        user_input = input("Enter any additional text (optional, press Enter to skip): ")
        image_path = input("Enter the image path: ")
        result = multimodal_assistant(user_input, image_path)
    else:
        print("Invalid input method. Please choose 'text', 'speech', or 'image'.")

    print("Assistant Response:", result)
