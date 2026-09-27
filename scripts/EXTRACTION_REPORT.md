# Extraction Report

## Corpus

- **Source:** `OJ_L_202401689_EN_TXT.pdf`, Regulation (EU) 2024/1689 (EU AI Act), Official Journal of the European Union, 12.7.2024
- **Format:** text-based PDF, 144 pages, English
- **Downloaded from:** https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689

## Corpus stats

```
source files: 1
total entries: 144
total characters: 582969
avg characters/entry: 4048
near-empty entries (<50 chars): 0
```

## Extraction method

- PyMuPDF, one entry per page.
- All files read and written in UTF-8 (`ensure_ascii=False`), so special characters (‘ ’, —, €) are preserved.

## Issues encountered and how we handled them

1. **Repeated headers and footers.** Every page repeats "OJ L, 12.7.2024" + "EN" and "ELI: http://data.europa.eu/eli/reg/2024/1689/oj" + the page number (e.g. "3/144"), in a different order on odd and even pages. We removed them with regular expressions, plus a fallback that removes short lines appearing on more than 50% of pages. ELI links that are part of the actual text (legal citations) are kept. Verified: no header or footer remains in `corpus.json`.
2. **Page 1 banner.** The "Official Journal of the European Union / L series / 2024/1689 12.7.2024" banner was mixed into the text of page 1. Removed with specific patterns.
3. **Broken footnote markers.** Superscript footnote numbers were extracted on separate lines, e.g. `(` / `1` / `)`. We rebuilt them as `(1)`.
4. **Line breaks within paragraphs.** The PDF breaks lines mid-sentence. We re-join lines, except when a line starts a new block: a recital `(12)`, a point `(a)`, a numbered paragraph `1.`, or a heading (`Article`, `CHAPTER`, `SECTION`, `ANNEX`). Article headings and titles stay on their own lines.
5. **Footnotes at the bottom of each page.** Footnotes (references to other EU acts) appear at the end of each page's text. We kept them because they contain factual references. As a result, a recital or article that continues on the next page is split across two entries. We will handle this at the chunking step.

## Other checks

- **Near-empty entries (< 50 characters):** 0.
- **Scanned or image-only pages:** none, all pages have selectable text.
- **Tables:** no tables that affect extraction.
- **Encoding problems:** none.