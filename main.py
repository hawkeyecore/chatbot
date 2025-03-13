from fastapi import FastAPI, UploadFile, File
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PDFToTextConverter, BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import shutil

app = FastAPI()

# Initialize Haystack components
document_store = InMemoryDocumentStore()
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
pipe = ExtractiveQAPipeline(reader, retriever)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pdf_converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
    docs = pdf_converter.convert(file_path=file_path)
    document_store.write_documents(docs)
    return {"message": "PDF uploaded and indexed successfully"}

@app.get("/ask/")
async def ask_question(query: str):
    prediction = pipe.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 3}})
    return {"answer": prediction["answers"][0].answer}
