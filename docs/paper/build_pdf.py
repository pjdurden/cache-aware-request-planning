"""Render the paper markdown to a PDF via headless Chrome.

Requires the `markdown` package (pip install markdown) and a Chrome/Chromium
binary on PATH. Produces cache-aware-client-request-planning-paper.pdf next to
this script. The intermediate paper.html is also written (and gitignored).

    /home/pjsump/cache-aware-request-planning/.venv/bin/python docs/paper/build_pdf.py
"""
import shutil
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
MD = HERE / "cache-aware-client-request-planning-paper.md"
HTML = HERE / "paper.html"
PDF = HERE / "cache-aware-client-request-planning-paper.pdf"

CSS = """
@page { size: A4; margin: 22mm 20mm; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; }
h1 { font-size: 19pt; text-align: center; margin: 0 0 4pt 0; line-height: 1.2; }
h2 { font-size: 13pt; margin: 16pt 0 4pt 0; border-bottom: 1px solid #ccc;
     padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 3pt 0; }
p { margin: 0 0 7pt 0; text-align: justify; }
code { font-family: "DejaVu Sans Mono", Consolas, monospace; font-size: 9pt;
       background: #f4f4f4; padding: 1px 3px; border-radius: 3px; }
pre { background: #f4f4f4; padding: 8px 10px; border-radius: 4px;
      font-size: 8.5pt; line-height: 1.35; overflow-x: auto;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; width: 100%;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3px 8px; text-align: left; }
th { background: #eee; }
img { max-width: 78%; display: block; margin: 8pt auto; page-break-inside: avoid; }
blockquote { margin: 8pt 0; padding: 4pt 12pt; border-left: 3px solid #ccc;
             color: #333; font-style: italic; }
h2 { page-break-after: avoid; }
"""


def build_html():
    body = markdown.markdown(
        MD.read_text(),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    HTML.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def find_chrome():
    for name in ("google-chrome", "google-chrome-stable",
                 "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def build_pdf():
    chrome = find_chrome()
    if chrome is None:
        print("No Chrome/Chromium found. HTML written to", HTML)
        return
    subprocess.run(
        [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
         "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True,
    )
    print("PDF written to", PDF)


if __name__ == "__main__":
    build_html()
    build_pdf()
