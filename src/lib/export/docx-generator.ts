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

/**
 * Parse HTML content into DOCX block elements.
 * Returns (Paragraph | Table)[] — Tables MUST be top-level block children,
 * never nested inside a Paragraph (invalid OOXML).
 */
function buildSectionContent(html: string): (Paragraph | Table)[] {
  const result: (Paragraph | Table)[] = [];
  const $ = cheerio.load(html);

  const elements = $("body").length > 0 ? $("body").contents() : $.root().contents();

  elements.each((_, node) => {
    if (node.type === "text") {
      const text = (node.data || "").trim();
      if (text) result.push(textParagraph(text));
      return;
    }

    if (node.type !== "tag") return;
    const el = $(node);
    const tag = node.tagName?.toLowerCase();

    // ----------------------------------------------------------------
    // TABLE — must be a direct block child, NOT inside a Paragraph
    // ----------------------------------------------------------------
    if (tag === "table") {
      const rows = el.find("tr");
      if (rows.length === 0) return;

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

        if (docCells.length > 0) {
          docRows.push(new TableRow({ children: docCells }));
        }
      });

      if (docRows.length > 0) {
        // Push Table directly — this is the fix for the corruption bug
        result.push(
          new Table({
            rows: docRows,
            width: { size: 100, type: WidthType.PERCENTAGE },
          })
        );
        // Add spacing paragraph after table
        result.push(new Paragraph({ text: "" }));
      }
      return;
    }

    // ----------------------------------------------------------------
    // UNORDERED LIST
    // ----------------------------------------------------------------
    if (tag === "ul") {
      el.find("li").each((_, li) => {
        const text = $(li).text().trim();
        if (text) {
          result.push(
            new Paragraph({
              children: [new TextRun({ text, size: 22 })],
              bullet: { level: 0 },
            })
          );
        }
      });
      return;
    }

    // ----------------------------------------------------------------
    // ORDERED LIST
    // ----------------------------------------------------------------
    if (tag === "ol") {
      el.find("li").each((_, li) => {
        const text = $(li).text().trim();
        if (text) {
          result.push(
            new Paragraph({
              children: [new TextRun({ text, size: 22 })],
              numbering: { reference: ORDERED_NUMBERING_REF, level: 0 },
            })
          );
        }
      });
      return;
    }

    // ----------------------------------------------------------------
    // HEADINGS
    // ----------------------------------------------------------------
    if (tag === "h1") {
      const text = el.text().trim();
      if (text) result.push(new Paragraph({ text, heading: HeadingLevel.HEADING_1 }));
      return;
    }
    if (tag === "h2") {
      const text = el.text().trim();
      if (text) result.push(new Paragraph({ text, heading: HeadingLevel.HEADING_2 }));
      return;
    }
    if (tag === "h3") {
      const text = el.text().trim();
      if (text) result.push(new Paragraph({ text, heading: HeadingLevel.HEADING_3 }));
      return;
    }
    if (tag === "h4") {
      const text = el.text().trim();
      if (text) result.push(new Paragraph({ text, heading: HeadingLevel.HEADING_4 }));
      return;
    }

    // ----------------------------------------------------------------
    // PARAGRAPH — recurse into inline elements for mixed formatting
    // ----------------------------------------------------------------
    if (tag === "p") {
      const children: TextRun[] = [];
      el.contents().each((_, child) => {
        if (child.type === "text") {
          const text = (child.data || "").trim();
          if (text) children.push(new TextRun({ text, size: 22 }));
        } else if (child.type === "tag") {
          const childEl = $(child);
          const childTag = child.tagName?.toLowerCase();
          const text = childEl.text().trim();
          if (!text) return;
          if (childTag === "strong" || childTag === "b") {
            children.push(new TextRun({ text, bold: true, size: 22 }));
          } else if (childTag === "em" || childTag === "i") {
            children.push(new TextRun({ text, italics: true, size: 22 }));
          } else {
            children.push(new TextRun({ text, size: 22 }));
          }
        }
      });
      if (children.length > 0) {
        result.push(new Paragraph({ children, spacing: { after: 120 } }));
      } else {
        const text = el.text().trim();
        if (text) result.push(textParagraph(text));
      }
      return;
    }

    // ----------------------------------------------------------------
    // INLINE BOLD / ITALIC at block level
    // ----------------------------------------------------------------
    if (tag === "strong" || tag === "b") {
      const text = el.text().trim();
      if (text) {
        result.push(
          new Paragraph({
            children: [new TextRun({ text, bold: true, size: 22 })],
            spacing: { after: 120 },
          })
        );
      }
      return;
    }

    if (tag === "em" || tag === "i") {
      const text = el.text().trim();
      if (text) {
        result.push(
          new Paragraph({
            children: [new TextRun({ text, italics: true, size: 22 })],
            spacing: { after: 120 },
          })
        );
      }
      return;
    }

    // ----------------------------------------------------------------
    // GENERIC FALLBACK — strip tags, handle text bullets
    // ----------------------------------------------------------------
    const text = stripHtml($.html(el) || el.text());
    if (text) {
      const lines = text.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) {
          const isBullet = trimmed.startsWith("•") || trimmed.startsWith("- ");
          const bulletText = isBullet
            ? trimmed.replace(/^[•-]\s*/, "")
            : trimmed;
          result.push(textParagraph(bulletText, isBullet ? 1 : 0));
        }
      }
    }
  });

  // Absolute fallback: if nothing parsed, render plain text
  if (result.length === 0) {
    const fallbackText = stripHtml(html);
    if (fallbackText) {
      const lines = fallbackText.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) result.push(textParagraph(trimmed));
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
