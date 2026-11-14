import io

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app import resume_intake as ri
from conftest import RESUME_TEXT


def make_pdf(pages):
    """pages: list of lists of (x, y, text)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for page in pages:
        for x, y, text in page:
            c.drawString(x, y, text)
        c.showPage()
    c.save()
    return buf.getvalue()


def text_page(text, x=72, top=740):
    return [(x, top - 14 * i, line) for i, line in enumerate(text.splitlines())]


# --- verification ----------------------------------------------------------

def test_real_resume_passes():
    verdict = ri.verify_resume(RESUME_TEXT)
    assert verdict.status == "pass", verdict


def test_invoice_rejected():
    invoice = ("INVOICE #10442\nBill To: Example Ltd, 1 High Street\nPayment terms: 30 days\n"
               + "Item  Qty  Unit price  Total\n" + "Consulting services for March  1  1200.00  1200.00\n" * 10
               + "Subtotal 12000.00\nTax 2400.00\nAmount due 14400.00\n")
    verdict = ri.verify_resume(invoice)
    assert verdict.status == "fail"
    assert "invoice" in verdict.reason


def test_research_paper_rejected():
    paper = ("Abstract\n" + "We study gradient methods for convex problems and prove convergence. " * 15
             + "\nIntroduction\n" + "Prior work by Smith et al. established bounds [1]. " * 10
             + "\nReferences\n[1] Smith, J. et al. doi:10.1000/xyz\n")
    verdict = ri.verify_resume(paper)
    assert verdict.status == "fail"
    assert "research paper" in verdict.reason


def test_too_short_rejected():
    verdict = ri.verify_resume("Jane Doe\nPython developer")
    assert verdict.status == "fail"
    assert "too short" in verdict.reason


def test_plain_prose_rejected():
    prose = "The quick brown fox jumps over the lazy dog while the sun sets slowly. " * 12
    assert ri.verify_resume(prose).status == "fail"


def test_prepare_from_text_rejects_with_message():
    with pytest.raises(ri.ResumeRejected) as e:
        ri.prepare_from_text("Lorem ipsum dolor sit amet. " * 30)
    assert e.value.status_code == 422


# --- file checks -----------------------------------------------------------

def test_unsupported_extension():
    with pytest.raises(ri.ResumeRejected) as e:
        ri.prepare_from_upload(b"PK\x03\x04...", "resume.docx")
    assert e.value.status_code == 415


def test_oversized_file():
    with pytest.raises(ri.ResumeRejected) as e:
        ri.prepare_from_upload(b"%PDF" + b"0" * ri.MAX_FILE_BYTES, "cv.pdf")
    assert e.value.status_code == 413


def test_fake_pdf_signature():
    with pytest.raises(ri.ResumeRejected) as e:
        ri.prepare_from_upload(b"just text pretending", "cv.pdf")
    assert "valid PDF" in e.value.message


def test_scanned_pdf_has_no_text():
    with pytest.raises(ri.ResumeRejected) as e:
        ri.prepare_from_upload(make_pdf([[]]), "scan.pdf")
    assert "scanned" in e.value.message


def test_too_many_pages():
    with pytest.raises(ri.ResumeRejected) as e:
        ri.prepare_from_upload(make_pdf([text_page(RESUME_TEXT)] * 12), "book.pdf")
    assert "12 pages" in e.value.message


def test_resume_pdf_accepted():
    text = ri.prepare_from_upload(make_pdf([text_page(RESUME_TEXT)]), "cv.pdf")
    assert "jane.doe@example.com" in text
    assert "Senior Software Engineer" in text


def test_two_column_pdf_reads_column_by_column():
    left = [(50, 700 - 14 * i, f"Left column line {i} experience detail") for i in range(30)]
    right = [(330, 700 - 14 * i, f"Right column item {i} skill") for i in range(30)]
    header = [(150, 750, "Jane Doe - Senior Backend Engineer - jane@example.com")]
    page = ri.extract_pdf_pages(make_pdf([header + left + right]))[0]
    assert page.index("Jane Doe") < page.index("Left column line 0")
    assert page.index("Left column line 29") < page.index("Right column item 0")


def test_repeated_header_kept_once_and_page_numbers_dropped():
    lines = RESUME_TEXT.splitlines()
    p1 = text_page("Jane Doe Resume\n" + "\n".join(lines[:15]) + "\nPage 1 of 2")
    p2 = text_page("Jane Doe Resume\n" + "\n".join(lines[15:]) + "\nPage 2 of 2")
    pages = ri.extract_pdf_pages(make_pdf([p1, p2]))
    assert "Jane Doe Resume" in pages[0]
    assert "Jane Doe Resume" not in pages[1]
    assert "Page 1 of 2" not in pages[0] and "Page 2 of 2" not in pages[1]


def test_trim_caps_length_at_line_boundary():
    long_pages = [("A line of résumé text about experience.\n" * 200)] * 3
    trimmed = ri.trim(long_pages)
    assert len(trimmed) <= ri.MAX_CHARS
    assert trimmed.endswith("experience.")


def test_trim_keeps_first_pages_only():
    pages = [f"page {i} content" for i in range(8)]
    trimmed = ri.trim(pages)
    assert "page 4 content" in trimmed and "page 5 content" not in trimmed
