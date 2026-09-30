"""Conservative command-pattern guard; not a universal prompt-injection detector."""
import re
import unicodedata

# Match behavioral commands, not the topic words 'prompt injection' or 'instructions'.
PATTERNS = (
    ("instruction_override", r"\b(?:ignore|disregard|override|forget)\s+(?:(?:all|any|the|your)\s+)*(?:previous|prior|system|developer|above)\s+(?:instructions?|messages?|prompts?|rules?)\b"),
    ("secret_request", r"\b(?:reveal|print|send|leak|expose)\b[^\n.!?]{0,80}\b(?:system prompt|developer message|api[_ -]?key|environment variables|hidden instructions)\b"),
    ("role_spoofing", r"(?:<\|im_start\|>\s*(?:system|developer)|(?:^|\n)\s*(?:system|developer)\s+(?:override|message)\s*:)"),
)
COMPILED = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in PATTERNS]
REFUSAL = "I can't follow requests to override the assistant's instructions or reveal private configuration. Please ask your corpus question directly."


def normalize(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', value) if unicodedata.category(c) != 'Cf')


def source_risks(value):
    return [name for name, pattern in COMPILED if pattern.search(normalize(value))]


def request_risks(value):
    # Only an explicit imperative at the start is blocked. Discussion/quotation
    # of attacks is allowed; the normal system rules remain the broader defense.
    value = normalize(value).strip()
    value = re.sub(r'^please\s+', '', value, flags=re.IGNORECASE)
    return [name for name, pattern in COMPILED if pattern.match(value)]


def screen_passages(hits):
    allowed, blocked = [], []
    for hit in hits:
        reasons = source_risks(hit.get('text', ''))
        if reasons:
            blocked.append({key: hit.get(key) for key in ('chunk_id', 'source', 'pages')} | {'reasons': reasons})
        else:
            allowed.append(hit)
    return allowed, blocked
