# server.py
from fastapi import FastAPI, UploadFile, Form, Body, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from rag_engine import (
    answer_query,
    read_file,
    list_user_files,
    delete_user_file_embeddings,
)
import os
import asyncio
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

app = FastAPI()

# Optional: CORS if you're connecting from a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
async def ask(
    query: str = Form(...), file: UploadFile = None, user_id: str = Form(...)
):
    user_file_text = None
    print(f"📝 Received query: {query}, 👤 user_id: {user_id} 🚀")
    if file:
        contents = await file.read()
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        user_file_text = read_file(temp_path)
        os.remove(temp_path)

    async def stream_generator():
        filename = file.filename if file else None
        async for chunk in answer_query(query, user_file_text, user_id, filename):
            yield chunk

    return StreamingResponse(stream_generator(), media_type="text/plain")


@app.get("/")
async def hello():
    print("Received a request to the root endpoint.")
    return {"message": "Hello from the RAG server!\n"}


@app.delete("/delete_file")
async def delete_file(user_id: str = Body(...), file_hash: str = Body(...)):
    result = delete_user_file_embeddings(user_id, file_hash)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result


@app.get("/list_files")
async def get_user_files(user_id: str = Query(...)):
    result = list_user_files(user_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result
