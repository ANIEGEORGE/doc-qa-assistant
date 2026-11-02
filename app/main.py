from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.chunking import split_text
from app.db import engine, get_db, init_db
from app.models import Chunk, Document
from app.embeddings import embed_texts

UPLOAD_DIR = Path("/code/uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    UPLOAD_DIR.mkdir(exist_ok=True)
    yield


app = FastAPI(title="Doc QA Assistant", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": "0.2.0"}


@app.get("/db-health")
def db_health():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"database": "ok"}


@app.post("/documents")
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        reader = PdfReader(file.file)
        pages = [
            (number, page.extract_text() or "")
            for number, page in enumerate(reader.pages, start=1)
        ]
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this PDF")

    pieces = []
    for page_number, text in pages:
        for piece in split_text(text):
            pieces.append((page_number, piece))

    if not pieces:
        raise HTTPException(
            status_code=400,
            detail="No readable text found (is this a scanned PDF?)",
        )

    try:
        vectors = embed_texts([c for _, c in pieces], "RETRIEVAL_DOCUMENT")
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Embedding service failed: {type(exc).__name__}"
        )

    doc = Document(filename=file.filename, page_count=len(pages))
    doc.chunks = [
        Chunk(page_number=p, content=c, embedding=v)
        for (p, c), v in zip(pieces, vectors)
    ]
    db.add(doc)
    db.commit()
    db.refresh(doc)

    file.file.seek(0)
    (UPLOAD_DIR / f"{doc.id}.pdf").write_bytes(file.file.read())

    return {
        "id": doc.id,
        "filename": doc.filename,
        "page_count": doc.page_count,
        "chunk_count": len(pieces),
    }



@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.id).all()
    return [
        {"id": d.id, "filename": d.filename, "page_count": d.page_count}
        for d in docs
    ]


@app.get("/documents/{document_id}/chunks")
def list_chunks(document_id: int, limit: int = 5, db: Session = Depends(get_db)):
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.id)
        .limit(limit)
        .all()
    )
    return [
        {"id": r.id, "page_number": r.page_number, "content": r.content}
        for r in rows
    ]


@app.get("/embed-test")
def embed_test(text: str = "The maximum age is 25 years"):
    vector = embed_texts([text], "RETRIEVAL_DOCUMENT")[0]
    return {"text": text, "length": len(vector), "first_5_numbers": vector[:5]}