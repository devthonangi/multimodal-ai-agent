from fastapi import FastAPI, UploadFile, Form
from inference.multimodal_agent import MultimodalAgent
import shutil, os, uuid

app = FastAPI(title="Multimodal Vision-Language Agent")
agent = MultimodalAgent()

@app.post("/query")
async def query_image(image: UploadFile, question: str = Form(...)):
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    answer = agent.process_query(temp_path, question)
    os.remove(temp_path)
    return {"response": answer}

@app.get("/")
async def root():
    return {"message": "Multimodal Agent is running!"}
