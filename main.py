import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

file_path = "doc.txt"
if file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
else:
    loader = TextLoader(file_path, encoding="utf-8")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.from_documents(texts, embedding_model)

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACEHUB_API_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HUGGINGFACEHUB_API_TOKEN)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    max_new_tokens=256
)

llm_pipeline = HuggingFacePipeline(pipeline=generator)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_pipeline,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

print("Document Helper Ready! Type 'exit' to quit.\n")

while True:
    query = input("Enter your query: ").strip()
    if query.lower() in ["exit", "quit"]:
        break
    try:
        answer = qa_chain.invoke({"query": query})
        print("\nAnswer:\n", answer["result"], "\n")
    except Exception as e:
        print("Error:", e)
