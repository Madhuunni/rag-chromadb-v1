
import argparse
import hashlib
import json
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from filelock import FileLock
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing import chunks, normalize, tokenize


def client(root):
    return chromadb.PersistentClient(path=str(root / 'chroma'),
                                    settings=Settings(anonymized_telemetry=False))


def upload(filename, root, synonyms=None, size=700, overlap=100):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(filename).resolve()
    if source.suffix.lower() != '.txt':
        raise ValueError('Expected a .txt file')
    if source.stat().st_size > 10 * 1024 * 1024:
        raise ValueError('Demo limit: 10 MiB per text file')
    text = source.read_text(encoding='utf-8-sig')
    records = chunks(text, size, overlap)
    if not records:
        raise ValueError('No indexable paragraphs; use punctuation or # headings')
    synonyms = synonyms or {}
    if not isinstance(synonyms, dict) or any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, str)
        or not v.strip() for k, v in synonyms.items()
    ):
        raise ValueError('Synonyms must map nonempty phrases to nonempty strings')
    documents = [normalize(r['section'] + '\n' + r['text'], synonyms)
                 for r in records]
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
        lowercase=False, ngram_range=(1, 2), sublinear_tf=True,
        norm='l2', dtype=np.float32)
    matrix = vectorizer.fit_transform(documents)
    if matrix.shape[1] > 20000:
        raise ValueError('Vocabulary exceeds dense Chroma demo limit (20000); use sparse retrieval')
    if np.any(np.asarray(matrix.getnnz(axis=1)).ravel() == 0):
        raise ValueError('A chunk has no indexable words; revise the document')
    with FileLock(str(root / 'index.lock'), timeout=120):
        db = client(root)
        version = uuid.uuid4().hex
        name = 'txt_' + version
        artifact = root / f'{version}.joblib'
        manifest_path = root / 'active.json'
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        collection = db.create_collection(name=name, embedding_function=None,
                        configuration={'hnsw': {'space': 'cosine'}})
        active = False
        try:
            for start in range(0, len(records), 64):
                batch = records[start:start+64]
                collection.add(
                    ids=[str(i) for i in range(start, start+len(batch))],
                    documents=[r['text'] for r in batch],
                    embeddings=matrix[start:start+64].toarray().tolist(),
                    metadatas=[{'source': str(source), 'section': r['section'],
                                'chunk_index': r['chunk_index']} for r in batch])
            # Keep sparse matrix for exact scoring: avoid ANN variability.
            joblib.dump({'vectorizer': vectorizer, 'matrix': matrix,
                         'records': records, 'synonyms': synonyms}, artifact)
            manifest = {'collection': name, 'artifact': artifact.name,
                        'source': str(source),
                        'sha256': hashlib.sha256(text.encode()).hexdigest()}
            temp = root / f'active_{version}.tmp'
            temp.write_text(json.dumps(manifest), encoding='utf-8')
            temp.replace(manifest_path)
            active = True
        finally:
            if not active:
                db.delete_collection(name)
                artifact.unlink(missing_ok=True)
        # Retain previous versions: avoids destroying rollback evidence.
        if old:
            (root / 'previous.json').write_text(json.dumps(old), encoding='utf-8')
    return {'chunks': len(records), 'dimensions': matrix.shape[1], 'collection': name}


def search(question, root, top_k=3, min_score=0.15, min_coverage=0.5):
    if not question.strip() or top_k < 1:
        raise ValueError('Provide a question and positive top_k')
    if not 0 <= min_score <= 1 or not 0 <= min_coverage <= 1:
        raise ValueError('Score and coverage thresholds must be between 0 and 1')
    root = Path(root).resolve()
    if not (root / 'active.json').exists():
        raise ValueError('Upload a file first')
    with FileLock(str(root / 'index.lock'), timeout=120):
        manifest = json.loads((root / 'active.json').read_text())
        # Only load trusted application-owned artifacts.
        data = joblib.load(root / manifest['artifact'])
        question = normalize(question, data['synonyms'])
        terms = set(tokenize(question))
        q = data['vectorizer'].transform([question])
        if not terms or q.nnz == 0:
            return {'status': 'no_match', 'reason': 'No indexed query terms', 'matches': []}
        scores = (data['matrix'] @ q.T).toarray().ravel()
        candidates = []
        for i in np.argsort(-scores, kind='stable'):
            rec = data['records'][int(i)]
            present = set(tokenize(normalize(rec['section'] + '\n' + rec['text'], data['synonyms'])))
            shared = terms & present
            coverage = len(shared) / len(terms)  # Includes unknown query terms.
            if scores[i] < min_score or coverage < min_coverage:
                continue
            candidates.append({'id': str(i), 'score': float(scores[i]),
                               'coverage': coverage, 'matched_terms': sorted(shared)})
            if len(candidates) == top_k:
                break
        if not candidates:
            return {'status': 'no_match', 'reason': 'Below lexical relevance thresholds', 'matches': []}
        collection = client(root).get_collection(manifest['collection'], embedding_function=None)
        stored = collection.get(ids=[c['id'] for c in candidates],
                                include=['documents', 'metadatas'])
        lookup = {id_: (doc, meta) for id_, doc, meta in
                  zip(stored['ids'], stored['documents'], stored['metadatas'])}
        for candidate in candidates:
            doc, meta = lookup[candidate.pop('id')]
            candidate.update(text=doc, **meta)
        return {'status': 'matched', 'matches': candidates}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-dir', default='./policy_index_fixed')
    subs = p.add_subparsers(dest='command', required=True)
    u = subs.add_parser('upload')
    u.add_argument('file')
    u.add_argument('--synonyms', help='Optional reviewed JSON phrase aliases')
    u.add_argument('--chunk-size', type=int, default=700)
    u.add_argument('--overlap', type=int, default=100)
    s = subs.add_parser('search')
    s.add_argument('question')
    s.add_argument('--top-k', type=int, default=3)
    s.add_argument('--min-score', type=float, default=0.15)
    s.add_argument('--min-coverage', type=float, default=0.5)
    args = p.parse_args()
    try:
        if args.command == 'upload':
            synonyms = json.loads(Path(args.synonyms).read_text()) if args.synonyms else {}
            result = upload(args.file, args.index_dir, synonyms, args.chunk_size, args.overlap)
        else:
            result = search(args.question, args.index_dir, args.top_k, args.min_score, args.min_coverage)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as exc:
        p.exit(2, f'Error: {exc}\n')

if __name__ == '__main__':
    main()
