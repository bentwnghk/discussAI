import {
  Document,
  Paragraph,
  TextRun,
  HeadingLevel,
  Packer,
  Table,
  TableRow,
  TableCell,
  WidthType,
  AlignmentType,
  ShadingType,
  LevelFormat,
} from "docx";
import * as cheerio from "cheerio";
import type { Element as DomElement, Text as DomText } from "domhandler";
import type { DialogueItem, LearningNotes, Speaker } from "@/types";

const SPEAKER_COLORS_HEX: Record<Speaker, string> = {
  "Candidate A": "E3F2FD",
  "Candidate B": "FFFDE7",
  "Candidate C": "E8F5E8",
  "Candidate D": "FDECEA",
};

const ORDERED_NUMBERING_REF = "ordered-list";

function stripHtml(html: string): string {
  return html
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#039;/g, "'")
    .trim();
}

function textParagraph(text: string, indent = 0): Paragraph {
  return new Paragraph({
    children: [new TextRun({ text, size: 22 })],
    indent: indent > 0 ? { left: indent * 360 } : undefined,
    spacing: { after: 120 },
  });
}

function buildTranscriptParagraphs(items: DialogueItem[]): Paragraph[] {
  const paragraphs: Paragraph[] = [
    new Paragraph({
      text: "Transcript",
      heading: HeadingLevel.HEADING_1,
    }),
    new Paragraph({ text: "" }),
  ];

  for (const item of items) {
    const color = SPEAKER_COLORS_HEX[item.speaker as Speaker] || "FFFFFF";

    // Speaker name: bold, no background shading (matches reference style)
    paragraphs.push(
      new Paragraph({
        children: [
          new TextRun({ text: `${item.speaker}:`, bold: true, size: 22 }),
        ],
        spacing: { after: 40 },
      })
    );

    // Speech text: indented + speaker color shading (matches reference style)
    paragraphs.push(
      new Paragraph({
        children: [new TextRun({ text: item.text, size: 22 })],
        shading: { type: ShadingType.CLEAR, fill: color },
        indent: { left: 360 },
        spacing: { after: 120 },
      })
    );

    // Empty paragraph separator between exchanges
    paragraphs.push(new Paragraph({ text: "" }));
  }

  return paragraphs;
}

// ─── Inline run types for the line-buffer approach ───────────────────────────

type InlineRun =
  | { kind: "text"; raw: string }
  | { kind: "bold"; text: string }
  | { kind: "italic"; text: string };

/**
 * Converts a buffered "line" (inline runs collected between <br> tags) into a
 * single Paragraph, preserving bullet characters, &nbsp; indentation levels,
 * and mixed bold/italic inline formatting exactly as the UI renders them.
 */
function flushLineBuffer(buffer: InlineRun[]): Paragraph | null {
  if (buffer.length === 0) return null;

  let indentLevel = 0;
  let isBullet = false;

  // Find the index of the first text run — it carries indent/bullet info
  const firstTextIdx = buffer.findIndex((item) => item.kind === "text");

  const runs: TextRun[] = [];

  buffer.forEach((item, idx) => {
    if (idx === firstTextIdx) {
      // ── First text run: extract &nbsp; prefix and bullet char ──
      const raw = (item as { kind: "text"; raw: string }).raw;

      // Count leading non-breaking spaces (\u00a0, i.e. &nbsp;) for indentation
      const nbspMatch = raw.match(/^(\u00a0*)/);
      const nbspCount = nbspMatch ? nbspMatch[1].length : 0;
      indentLevel = Math.floor(nbspCount / 4); // 4 &nbsp; = 1 indent level

      const afterNbsp = raw.slice(nbspCount);
      isBullet =
        afterNbsp.startsWith("•") || afterNbsp.startsWith("- ");

      const afterBullet = isBullet
        ? afterNbsp.replace(/^[•-]\s*/, "")
        : afterNbsp;

      // Normalize remaining &nbsp; → regular space, then trim leading space
      const finalText = afterBullet.replace(/\u00a0/g, " ").trimStart();
      if (finalText) runs.push(new TextRun({ text: finalText, size: 22 }));
    } else if (item.kind === "text") {
      // Subsequent text runs: normalize &nbsp; but keep all spacing
      const text = item.raw.replace(/\u00a0/g, " ");
      if (text) runs.push(new TextRun({ text, size: 22 }));
    } else if (item.kind === "bold") {
      if (item.text) runs.push(new TextRun({ text: item.text, bold: true, size: 22 }));
    } else if (item.kind === "italic") {
      if (item.text)
        runs.push(new TextRun({ text: item.text, italics: true, size: 22 }));
    }
  });

  if (runs.length === 0) return null;

  if (isBullet) {
    return new Paragraph({
      children: runs,
      bullet: { level: Math.min(indentLevel, 8) },
    });
  } else if (indentLevel > 0) {
    return new Paragraph({
      children: runs,
      indent: { left: indentLevel * 720 },
      spacing: { after: 120 },
    });
  } else {
    return new Paragraph({
      children: runs,
      spacing: { after: 120 },
    });
  }
}

/**
 * Parse HTML content into DOCX block elements.
 *
 * Strategy for flat <br>-separated HTML (used in Ideas & Communication
 * Strategies sections):
 *   • Inline nodes (text, <strong>, <em>, <b>, <i>) are buffered until a <br>
 *     is encountered, then flushed as ONE paragraph preserving mixed formatting.
 *   • Two or more consecutive <br> tags emit an empty paragraph (blank line).
 *   • Block elements (table, ul, ol, p, hN) flush the buffer first, then are
 *     emitted as standalone block nodes.
 *
 * Tables MUST be direct section children — never nested inside a Paragraph
 * (invalid OOXML that corrupts the file).
 */
function buildSectionContent(html: string): (Paragraph | Table)[] {
  const result: (Paragraph | Table)[] = [];
  const $ = cheerio.load(html);
  const nodes = (
    $("body").length > 0 ? $("body").contents() : $.root().contents()
  ).toArray();

  let lineBuffer: InlineRun[] = [];
  let consecutiveBrs = 0;

  const flush = () => {
    const para = flushLineBuffer(lineBuffer);
    if (para) result.push(para);
    lineBuffer = [];
  };

  for (const node of nodes) {
    // ── <br> tag ──────────────────────────────────────────────────────────────
    if (
      node.type === "tag" &&
      (node as DomElement).tagName?.toLowerCase() === "br"
    ) {
      if (consecutiveBrs === 0) {
        // First <br> ends the current line
        flush();
      }
      consecutiveBrs++;
      if (consecutiveBrs >= 2) {
        // Two or more consecutive <br> = blank-line separator
        result.push(new Paragraph({ text: "" }));
      }
      continue;
    }

    // Any non-<br> node resets the consecutive-<br> counter
    consecutiveBrs = 0;

    // ── Text node ─────────────────────────────────────────────────────────────
    if (node.type === "text") {
      const raw = (node as DomText).data || "";
      // Skip nodes that are only regular whitespace (no &nbsp;)
      if (!raw.trim() && !raw.includes("\u00a0")) continue;
      lineBuffer.push({ kind: "text", raw });
      continue;
    }

    if (node.type !== "tag") continue;

    const el = $(node as DomElement);
    const tag = (node as DomElement).tagName?.toLowerCase();

    // ── Block elements: flush the inline buffer first ─────────────────────────
    const BLOCK_TAGS = new Set([
      "table", "ul", "ol", "p",
      "h1", "h2", "h3", "h4", "h5", "h6",
      "div", "blockquote",
    ]);

    if (BLOCK_TAGS.has(tag)) {
      flush();
    }

    switch (tag) {
      // ── TABLE ───────────────────────────────────────────────────────────────
      case "table": {
        const rows = el.find("tr");
        if (rows.length === 0) break;

        const docRows: TableRow[] = [];
        rows.each((_, row) => {
          const cells = $(row).find("th, td");
          const isHeader = $(row).find("th").length > 0;
          const docCells: TableCell[] = [];
          cells.each((_, cell) => {
            const cellText = $(cell).text().trim();
            docCells.push(
              new TableCell({
                children: [
                  new Paragraph({
                    children: [
                      new TextRun({ text: cellText, bold: isHeader, size: 20 }),
                    ],
                  }),
                ],
                width: { size: 33, type: WidthType.PERCENTAGE },
              })
            );
          });
          if (docCells.length > 0)
            docRows.push(new TableRow({ children: docCells }));
        });

        if (docRows.length > 0) {
          // Table MUST be a direct block child — never inside a Paragraph
          result.push(
            new Table({
              rows: docRows,
              width: { size: 100, type: WidthType.PERCENTAGE },
            })
          );
          result.push(new Paragraph({ text: "" }));
        }
        break;
      }

      // ── UNORDERED LIST ──────────────────────────────────────────────────────
      case "ul":
        el.find("li").each((_, li) => {
          const text = $(li).text().trim();
          if (text)
            result.push(
              new Paragraph({
                children: [new TextRun({ text, size: 22 })],
                bullet: { level: 0 },
              })
            );
        });
        break;

      // ── ORDERED LIST ────────────────────────────────────────────────────────
      case "ol":
        el.find("li").each((_, li) => {
          const text = $(li).text().trim();
          if (text)
            result.push(
              new Paragraph({
                children: [new TextRun({ text, size: 22 })],
                numbering: { reference: ORDERED_NUMBERING_REF, level: 0 },
              })
            );
        });
        break;

      // ── HEADINGS ────────────────────────────────────────────────────────────
      case "h1":
      case "h2":
      case "h3":
      case "h4":
      case "h5":
      case "h6": {
        const levelMap: Record<string, (typeof HeadingLevel)[keyof typeof HeadingLevel]> = {
          h1: HeadingLevel.HEADING_1,
          h2: HeadingLevel.HEADING_2,
          h3: HeadingLevel.HEADING_3,
          h4: HeadingLevel.HEADING_4,
          h5: HeadingLevel.HEADING_5,
          h6: HeadingLevel.HEADING_6,
        };
        const text = el.text().trim();
        if (text) result.push(new Paragraph({ text, heading: levelMap[tag] }));
        break;
      }

      // ── BLOCK PARAGRAPH ─────────────────────────────────────────────────────
      case "p": {
        const children: TextRun[] = [];
        el.contents().each((_, child) => {
          if (child.type === "text") {
            const text = (child as DomText).data?.trim() || "";
            if (text) children.push(new TextRun({ text, size: 22 }));
          } else if (child.type === "tag") {
            const childTag = (
              child as DomElement
            ).tagName?.toLowerCase();
            const text = $(child).text().trim();
            if (!text) return;
            if (childTag === "strong" || childTag === "b")
              children.push(new TextRun({ text, bold: true, size: 22 }));
            else if (childTag === "em" || childTag === "i")
              children.push(new TextRun({ text, italics: true, size: 22 }));
            else children.push(new TextRun({ text, size: 22 }));
          }
        });
        if (children.length > 0)
          result.push(new Paragraph({ children, spacing: { after: 120 } }));
        else {
          const text = el.text().trim();
          if (text) result.push(textParagraph(text));
        }
        break;
      }

      // ── INLINE ELEMENTS — add to current line buffer ─────────────────────────
      case "strong":
      case "b":
        lineBuffer.push({ kind: "bold", text: el.text() });
        break;

      case "em":
      case "i":
        lineBuffer.push({ kind: "italic", text: el.text() });
        break;

      // ── GENERIC FALLBACK ────────────────────────────────────────────────────
      default: {
        const text = el.text().trim();
        if (text) lineBuffer.push({ kind: "text", raw: text });
        break;
      }
    }
  }

  // Flush any remaining buffered inline content after the last node
  flush();

  // Absolute fallback: if nothing was parsed, render as plain text
  if (result.length === 0) {
    const fallbackText = stripHtml(html);
    if (fallbackText) {
      for (const line of fallbackText.split("\n")) {
        const t = line.trim();
        if (t) result.push(textParagraph(t));
      }
    }
  }

  return result;
}

export async function generateDocx(
  items: DialogueItem[],
  notes: LearningNotes,
  title: string = "Group Discussion Notes"
): Promise<Buffer> {
  const doc = new Document({
    numbering: {
      config: [
        {
          reference: ORDERED_NUMBERING_REF,
          levels: [
            {
              level: 0,
              format: LevelFormat.DECIMAL,
              text: "%1.",
              alignment: AlignmentType.LEFT,
              style: {
                paragraph: {
                  indent: { left: 720, hanging: 360 },
                },
                run: { size: 22 },
              },
            },
          ],
        },
      ],
    },
    sections: [
      {
        children: [
          new Paragraph({
            text: title,
            heading: HeadingLevel.TITLE,
            alignment: AlignmentType.CENTER,
          }),
          new Paragraph({
            children: [
              new TextRun({
                text: "Generated by Mr. DiscussAI",
                size: 20,
                color: "666666",
              }),
            ],
            alignment: AlignmentType.CENTER,
          }),
          new Paragraph({ text: "" }),
          ...buildTranscriptParagraphs(items),
          new Paragraph({ text: "", pageBreakBefore: true }),
          new Paragraph({
            text: "Study Notes",
            heading: HeadingLevel.HEADING_1,
          }),
          new Paragraph({ text: "" }),
          new Paragraph({
            text: "Ideas",
            heading: HeadingLevel.HEADING_2,
          }),
          ...buildSectionContent(notes.ideas),
          new Paragraph({ text: "" }),
          new Paragraph({
            text: "Language",
            heading: HeadingLevel.HEADING_2,
          }),
          ...buildSectionContent(notes.language),
          new Paragraph({ text: "" }),
          new Paragraph({
            text: "Communication Strategies",
            heading: HeadingLevel.HEADING_2,
          }),
          ...buildSectionContent(notes.communication_strategies),
        ],
      },
    ],
  });

  return Packer.toBuffer(doc);
}
