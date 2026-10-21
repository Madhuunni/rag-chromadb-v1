
"""Rule-based text processing: no models or language downloads."""
import re
import unicodedata
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Conservative query/function words; negation and conditions are retained.
STOP = set('a an the is are was were be been being do does did how what when where which who why can could would should i me my we our you your it its this that these those of to for in on at by with and or please tell about'.split())
FORMS = {'refunds':'refund', 'payments':'payment', 'receipts':'receipt',
         'employees':'employee', 'requests':'request', 'days':'day',
         'approved':'approve', 'approval':'approve', 'processed':'process',
         'processing':'process', 'cancellations':'cancellation'}
PATTERN = re.compile(r"\d+(?:\.\d+)+|\w+(?:[-']\w+)*", re.UNICODE)


def tokenize(text):
    text = unicodedata.normalize('NFKC', text).lower().replace('’', "'")
    text = re.sub(r"\bcan't\b", 'cannot', text)
    text = re.sub(r"\bwon't\b", 'will not', text)
    text = re.sub(r"n't\b", ' not', text).replace('_', ' ')
    return [FORMS.get(t, t) for t in PATTERN.findall(text) if t not in STOP]


def normalize(text, synonyms):
    """Explicit reviewed phrase aliases, applied equally to queries/documents."""
    text = unicodedata.normalize('NFKC', text).lower()
    for phrase in sorted(synonyms, key=len, reverse=True):
        text = re.sub(r'(?<!\w)' + re.escape(phrase.lower()) + r'(?!\w)',
                      lambda m: synonyms[phrase].lower(), text)
    return text


def chunks(text, size=700, overlap=100):
    """Keep blank-line blocks separate; recognize # headings and short titles.

    Oversized blocks are recursively split; overlap never crosses blocks.
    Plain one-line titles are heuristic. Prefer explicit Markdown headings.
    """
    if size < 100 or not 0 <= overlap < size:
        raise ValueError('Require size >= 100 and 0 <= overlap < size')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=['\n\n', '\n', '. ', '; ', ' ', ''],
        keep_separator='end', length_function=len)
    heading = ''
    output = []
    # Introduce a block boundary before explicit headings.
    text = re.sub(r'(?m)^(#{1,6}\s+)', r'\n\n\1', text)
    for block in re.split(r'\n\s*\n', text):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if re.match(r'^#{1,6}\s+', lines[0]):
            heading = re.sub(r'^#{1,6}\s+', '', lines.pop(0)).strip()
            block = '\n'.join(lines).strip()
        elif len(lines) == 1 and len(block) <= 80 and not re.search(r'[.!?;:]$', block):
            heading = block
            continue
        if not block:
            continue
        for part in splitter.split_text(block):
            if tokenize(part):
                output.append({'text': part, 'section': heading,
                               'chunk_index': len(output)})
    return output
