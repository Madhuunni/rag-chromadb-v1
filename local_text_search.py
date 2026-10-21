import argparse
import json
import uuid
from pathlib import Path

import chromadb
import joblib
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import re

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())

def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Characters, not tokens.
        chunk_overlap=150,     # Target overlap.
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    return [
        chunk.strip()
        for chunk in splitter.split_text(text)
        if tokenize(chunk)
    ]


def get_client(index_dir):
    return chromadb.PersistentClient(
        path=str(index_dir / "chroma")
    )


def upload_or_update(filename, index_dir):
    index_dir.mkdir(parents=True, exist_ok=True)

    source = Path(filename).resolve()
    text = source.read_text(encoding="utf-8")
    chunks = create_chunks(text)

    if not chunks:
        raise ValueError("The file has no indexable text.")

    # Fit only on this document's chunks.
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,       # tokenize() already lowercases.
        ngram_range=(1, 2),     # Individual words and word pairs.
        max_features=4096,
        norm="l2",
        dtype=np.float32,
    )

    matrix = vectorizer.fit_transform(chunks)

    # Chroma's dense-vector index should not receive zero vectors.
    valid_rows = np.flatnonzero(
        np.asarray(matrix.getnnz(axis=1)).ravel() > 0
    )
    if not len(valid_rows):
        raise ValueError("No non-zero document vectors were produced.")

    client = get_client(index_dir)

    # Build a fresh version before switching the active index.
    version = uuid.uuid4().hex
    collection_name = f"txt_{version}"
    model_filename = f"vectorizer_{version}.joblib"

    collection = client.create_collection(
        name=collection_name,
        embedding_function=None,  # Disable pretrained embeddings.
        configuration={"hnsw": {"space": "cosine"}},
    )

    for start in range(0, len(valid_rows), 100):
        rows = valid_rows[start:start + 100]

        collection.add(
            ids=[f"chunk_{int(i)}" for i in rows],
            documents=[chunks[int(i)] for i in rows],
            embeddings=matrix[rows].toarray().tolist(),
            metadatas=[
                {
                    "source": str(source),
                    "chunk_index": int(i),
                }
                for i in rows
            ],
        )

    # Save the exact vectorizer used for this collection.
    joblib.dump(vectorizer, index_dir / model_filename)

    manifest_path = index_dir / "active.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )

    manifest = {
        "collection": collection_name,
        "vectorizer": model_filename,
        "source": str(source),
    }

    temporary = index_dir / "active.tmp"
    temporary.write_text(json.dumps(manifest), encoding="utf-8")
    temporary.replace(manifest_path)

    # Clean up the prior version after the new one becomes active.
    if previous:
        client.delete_collection(previous["collection"])
        (index_dir / previous["vectorizer"]).unlink(missing_ok=True)

    print(f"Indexed file: {source.name}")
    print(f"Stored chunks: {len(valid_rows)}")
    print(f"Vector dimensions: {matrix.shape[1]}")


def search(question, index_dir, top_k=3):
    manifest_path = index_dir / "active.json"
    if not manifest_path.exists():
        raise ValueError("Upload a file before searching.")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    # Load only the local artifact created by this application.
    vectorizer = joblib.load(
        index_dir / manifest["vectorizer"]
    )

    # Use transform(), never fit_transform(), for a question.
    query_vector = vectorizer.transform([question])

    if query_vector.nnz == 0:
        print("No query words match the indexed vocabulary.")
        return

    collection = get_client(index_dir).get_collection(
        name=manifest["collection"],
        embedding_function=None,
    )

    results = collection.query(
        query_embeddings=query_vector.toarray().tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    found = False

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1.0 - distance

        # Skip results with effectively no lexical overlap.
        if similarity <= 1e-6:
            continue

        found = True
        print(f"\nSimilarity: {similarity:.4f}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(document)

    if not found:
        print("No matching passage found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="./policy_index")

    commands = parser.add_subparsers(dest="command", required=True)

    upload_parser = commands.add_parser("upload")
    upload_parser.add_argument("file")

    search_parser = commands.add_parser("search")
    search_parser.add_argument("question")

    args = parser.parse_args()
    index_dir = Path(args.index_dir).resolve()

    if args.command == "upload":
        upload_or_update(args.file, index_dir)
    else:
        search(args.question, index_dir)
