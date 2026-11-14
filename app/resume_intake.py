"""
resume_intake.py - Turn an uploaded file or pasted text into résumé text that is
safe and cheap to send to the LLM.

Order of operations (each step can reject with a user-facing message):
  1. File checks     - size, type, PDF signature, encryption, page count
  2. Extraction      - column-aware PDF text, header/footer removal, clean-up
  3. Verification    - rule-based "is this a résumé?" scoring (no LLM)
  4. Trimming        - first pages / character cap, only after verification

The LLM parse step adds a final `is_resume` check (see ai_engine.parse_resume),
which catches borderline documents the rules can't decide.
"""

import io
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import pdfplumber

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_PASTE_CHARS = 60_000           # pasted text longer than this isn't a résumé
MAX_PDF_PAGES = 10                 # reject outright above this
KEEP_PAGES = 5                     # pages sent to the LLM
MAX_CHARS = 12_000                 # ~3k tokens sent to the LLM
MIN_WORDS = 60
LONG_WORDS = 4_000

ALLOWED_EXTENSIONS = (".pdf", ".txt")


class ResumeRejected(Exception):
    """Raised with a message that can be shown to the user as-is."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class Verdict:
    status: str                      # "pass" | "borderline" | "fail"
    score: int
    signals: List[str] = field(default_factory=list)
    reason: Optional[str] = None     # set when status == "fail"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def prepare_from_upload(content: bytes, filename: str) -> str:
    """Validate, extract, verify and trim an uploaded résumé file."""
    if len(content) > MAX_FILE_BYTES:
        raise ResumeRejected("File is too large. Résumés must be under 5 MB.", 413)
    if not content:
        raise ResumeRejected("The uploaded file is empty.")

    name = (filename or "").lower()
    if not name.endswith(ALLOWED_EXTENSIONS):
        raise ResumeRejected("Unsupported file type. Please upload a PDF or a .txt file.", 415)

    if name.endswith(".pdf"):
        pages = extract_pdf_pages(content)
    else:
        pages = [_decode_text(content)]

    return _verify_and_trim(pages)


def prepare_from_text(text: str) -> str:
    """Verify and trim a pasted résumé."""
    if len(text) > MAX_PASTE_CHARS:
        raise ResumeRejected("That text is far longer than a résumé. Please paste only your résumé.")
    return _verify_and_trim([text])


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_pdf_pages(content: bytes) -> List[str]:
    """Return cleaned text per page. Raises ResumeRejected for unusable PDFs."""
    if not content.lstrip()[:5].startswith(b"%PDF"):
        raise ResumeRejected("This file isn't a valid PDF.", 415)

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                raise ResumeRejected(
                    f"This PDF has {len(pdf.pages)} pages. A résumé should be at most "
                    f"{MAX_PDF_PAGES} pages; please upload just your résumé."
                )
            pages = [_page_text(page) for page in pdf.pages]
    except ResumeRejected:
        raise
    except Exception as e:  # pdfminer raises a variety of types
        kind = f"{type(e).__name__} {e}".lower()
        if "password" in kind or "encrypt" in kind:
            raise ResumeRejected("This PDF is password-protected. Please upload an unlocked copy.")
        log.warning("PDF extraction failed: %s", e)
        raise ResumeRejected("We couldn't read this PDF. Try exporting it again, or paste the text instead.")

    pages = _drop_repeated_lines(pages)
    if not any(p.strip() for p in pages):
        raise ResumeRejected(
            "No text could be found in this PDF. It looks like a scanned image; "
            "please upload a text-based PDF or paste your résumé as text."
        )
    return pages


def _page_text(page) -> str:
    """Extract one page, reading two-column layouts column by column."""
    try:
        words = page.extract_words()
    except Exception:
        words = []
    split = _find_column_split(words, float(page.width)) if words else None
    if split is None:
        return _clean(page.extract_text() or "")

    full, left, right = [], [], []
    for w in words:
        if w["x0"] < split < w["x1"]:
            full.append(w)
        elif w["x1"] <= split:
            left.append(w)
        else:
            right.append(w)
    # Full-width lines (usually the name/contact header) first, then each column
    parts = [_words_to_lines(group) for group in (full, left, right) if group]
    return _clean("\n".join(parts))


def _find_column_split(words: List[dict], width: float) -> Optional[float]:
    """
    Find a vertical gutter in the middle half of the page that almost no words
    cross, with plenty of words on both sides. Returns its x position or None.
    """
    if len(words) < 40:
        return None
    lo, hi = int(width * 0.25), int(width * 0.75)
    coverage = [0] * (hi - lo + 1)
    for w in words:
        start, end = max(int(w["x0"]), lo), min(int(w["x1"]), hi)
        for x in range(start, end + 1):
            coverage[x - lo] += 1

    max_crossing = max(2, len(words) // 50)
    best_start, best_len, run_start = None, 0, None
    for i, count in enumerate(coverage + [max_crossing + 1]):
        if count <= max_crossing:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = None
    if best_start is None or best_len < 6:
        return None

    split = lo + best_start + best_len / 2
    left = sum(1 for w in words if w["x1"] <= split)
    right = sum(1 for w in words if w["x0"] >= split)
    if min(left, right) < len(words) * 0.2:
        return None
    return split


def _words_to_lines(words: List[dict]) -> str:
    """Group words into lines by vertical position, left to right."""
    lines: List[List[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= 3:
            lines[-1].append(w)
        else:
            lines.append([w])
    return "\n".join(" ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"])) for line in lines)


def _drop_repeated_lines(pages: List[str]) -> List[str]:
    """Remove header/footer lines that repeat on every page (and bare page numbers)."""
    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    page_number = re.compile(r"^(page\s*)?#(\s*(of|/)\s*#)?$")
    repeated = set()
    if len(pages) >= 2:
        counts = Counter(k for p in pages for k in {key(l) for l in p.splitlines() if l.strip()})
        repeated = {k for k, n in counts.items() if n == len(pages)}

    cleaned = []
    for i, p in enumerate(pages):
        kept = [
            l for l in p.splitlines()
            # keep repeated lines (often the candidate's name) on the first page only
            if not page_number.match(key(l)) and (i == 0 or key(l) not in repeated)
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def _decode_text(content: bytes) -> str:
    if b"\x00" in content[:4096]:
        raise ResumeRejected("This file isn't plain text. Please upload a PDF or a .txt file.", 415)
    for encoding in ("utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ResumeRejected("Couldn't read this text file. Please save it as UTF-8 and try again.")


def _clean(text: str) -> str:
    text = re.sub(r"\(cid:\d+\)", " ", text)            # pdfminer glyph artefacts
    text = re.sub(r"[^\S\n]+", " ", text)               # collapse spaces/tabs
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(\+?\d[\d\s().-]{7,}\d)(?!\d)")
_PROFILE = re.compile(r"linkedin\.com/|github\.com/|gitlab\.com/", re.I)
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE_RANGE = re.compile(
    rf"((({_MONTH})\s*)?(19|20)\d\d|\d{{1,2}}/(19|20)?\d\d)\s*(-|–|—|to)\s*"
    rf"(present|current|now|today|((({_MONTH})\s*)?(19|20)\d\d)|\d{{1,2}}/(19|20)?\d\d)",
    re.I,
)
_SECTIONS = {
    "experience": r"(professional |work |relevant )?experience|employment( history)?|work history|career history",
    "education": r"education|academic (background|qualifications)|qualifications",
    "skills": r"(technical |core |key )?skills|technologies|tech stack|competencies",
    "projects": r"(personal |academic |key )?projects",
    "certifications": r"certifications?|licenses?( (&|and) certifications)?|courses",
    "summary": r"(professional )?summary|profile|objective|about me",
}
_SECTION_LINE = {
    name: re.compile(rf"^\W*({pattern})\W*$", re.I) for name, pattern in _SECTIONS.items()
}
_INVOICE_TERMS = re.compile(
    r"\b(invoice|bill to|subtotal|amount due|balance due|receipt|unit price|qty|payment terms|tax id|vat)\b",
    re.I,
)
_PAPER_TERMS = re.compile(r"^\W*(abstract|references|bibliography|introduction|conclusions?)\W*$", re.I | re.M)
_CITATION = re.compile(r"\bet al\.|\bdoi\s*:|arxiv|\[\d+\]", re.I)


def verify_resume(text: str) -> Verdict:
    """Score how much `text` looks like a résumé. No LLM involved."""
    words = len(text.split())
    if words < MIN_WORDS:
        return Verdict("fail", 0, [f"{words} words"], "This is too short to be a résumé. Please provide your full résumé.")

    score, signals = 0, []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    if _EMAIL.search(text):
        score += 2; signals.append("email")
    if any(sum(c.isdigit() for c in m) >= 9 for m in _PHONE.findall(text)):
        score += 1; signals.append("phone")
    if _PROFILE.search(text):
        score += 1; signals.append("profile link")

    sections = {
        name for name, rx in _SECTION_LINE.items()
        if any(len(l) <= 40 and rx.match(l) for l in lines)
    }
    if sections:
        score += min(len(sections), 4); signals.append("sections: " + ", ".join(sorted(sections)))

    dates = len(_DATE_RANGE.findall(text))
    if dates:
        score += 2 if dates >= 2 else 1; signals.append(f"{dates} date ranges")

    invoice_hits = len(set(m.lower() for m in _INVOICE_TERMS.findall(text)))
    paper_headings = {m.lower() for m in _PAPER_TERMS.findall(text)}
    non_space = re.sub(r"\s", "", text)
    digit_ratio = sum(c.isdigit() for c in non_space) / max(len(non_space), 1)

    reason = None
    if invoice_hits >= 2:
        score -= 4; signals.append("invoice terms")
        reason = "This looks like an invoice or receipt, not a résumé."
    if "abstract" in paper_headings and len(paper_headings) >= 2 and _CITATION.search(text):
        score -= 4; signals.append("paper structure")
        reason = "This looks like a research paper or report, not a résumé."
    if digit_ratio > 0.35:
        score -= 3; signals.append(f"{digit_ratio:.0%} digits")
        reason = reason or "This looks like a table or data export, not a résumé."
    if words > LONG_WORDS:
        score -= 2; signals.append(f"{words} words")

    if score >= 5:
        return Verdict("pass", score, signals)
    if score <= 1:
        return Verdict("fail", score, signals, reason or (
            "This doesn't look like a résumé: we couldn't find contact details, "
            "section headings (Experience, Education, Skills) or employment dates."
        ))
    return Verdict("borderline", score, signals)


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------

def trim(pages: List[str]) -> str:
    """Keep the first KEEP_PAGES pages and at most MAX_CHARS, cut at a line break."""
    text = "\n\n".join(p for p in pages[:KEEP_PAGES] if p.strip())
    if len(text) <= MAX_CHARS:
        return text
    cut = text.rfind("\n", 0, MAX_CHARS)
    return text[: cut if cut > MAX_CHARS * 0.8 else MAX_CHARS].rstrip()


def _verify_and_trim(pages: List[str]) -> str:
    pages = [_clean(p) for p in pages]
    full_text = "\n\n".join(p for p in pages if p)
    if not full_text:
        raise ResumeRejected("No text could be found. Please upload a text-based PDF or paste your résumé.")

    verdict = verify_resume(full_text)
    log.info("Résumé verification: %s (score %s; %s)", verdict.status, verdict.score, "; ".join(verdict.signals))
    if verdict.status == "fail":
        raise ResumeRejected(verdict.reason or "This doesn't look like a résumé.")

    return trim(pages)
