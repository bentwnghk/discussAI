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
} from "docx";
import * as cheerio from "cheerio";
import type { DialogueItem, LearningNotes, Speaker } from "@/types";

const SPEAKER_COLORS_HEX: Record<Speaker, string> = {
  "Candidate A": "E3F2FD",
  "Candidate B": "FFFDE7",
  "Candidate C": "E8F5E8",
  "Candidate D": "FDECEA",
};

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
    paragraphs.push(
      new Paragraph({
        children: [
          new TextRun({ text: `${item.speaker}: `, bold: true, size: 22 }),
        ],
        shading: { type: ShadingType.CLEAR, fill: color },
        spacing: { after: 40 },
      })
    );
    paragraphs.push(
      new Paragraph({
        children: [new TextRun({ text: item.text, size: 22 })],
        indent: { left: 360 },
        spacing: { after: 200 },
      })
    );
  }

  return paragraphs;
}

function buildSectionContent(html: string): Paragraph[] {
  const paragraphs: Paragraph[] = [];
  const $ = cheerio.load(html);

  const elements = $("body").length > 0 ? $("body").contents() : $.root().contents();

  elements.each((_, node) => {
    if (node.type === "text") {
      const text = (node.data || "").trim();
      if (text) paragraphs.push(textParagraph(text));
      return;
    }

    if (node.type !== "tag") return;
    const el = $(node);
    const tag = node.tagName?.toLowerCase();

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
              shading: isHeader
                ? { type: ShadingType.CLEAR, fill: "667eea" }
                : undefined,
            })
          );
        });

        if (docCells.length > 0) {
          docRows.push(new TableRow({ children: docCells }));
        }
      });

      if (docRows.length > 0) {
        paragraphs.push(
          new Paragraph({
            children: [
              new Table({
                rows: docRows,
                width: { size: 100, type: WidthType.PERCENTAGE },
              }),
            ],
          })
        );
      }
      return;
    }

    if (tag === "strong" || tag === "b") {
      const text = stripHtml($.html(el) || el.text());
      if (text) {
        paragraphs.push(
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
        paragraphs.push(
          new Paragraph({
            children: [new TextRun({ text, italics: true, size: 22 })],
            spacing: { after: 120 },
          })
        );
      }
      return;
    }

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
          paragraphs.push(textParagraph(bulletText, isBullet ? 1 : 0));
        }
      }
    }
  });

  if (paragraphs.length === 0) {
    const fallbackText = stripHtml(html);
    if (fallbackText) {
      const lines = fallbackText.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) paragraphs.push(textParagraph(trimmed));
      }
    }
  }

  return paragraphs;
}

export async function generateDocx(
  items: DialogueItem[],
  notes: LearningNotes,
  title: string = "Group Discussion Notes"
): Promise<Buffer> {
  const doc = new Document({
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
