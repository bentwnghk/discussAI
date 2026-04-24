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
  return html.replace(/<[^>]+>/g, "").trim();
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

function buildLearningNotesSections(notes: LearningNotes): Paragraph[] {
  const sections: Paragraph[] = [
    new Paragraph({ text: "", pageBreakBefore: true }),
    new Paragraph({
      text: "Study Notes",
      heading: HeadingLevel.HEADING_1,
    }),
    new Paragraph({ text: "" }),
  ];

  sections.push(
    new Paragraph({
      text: "Ideas",
      heading: HeadingLevel.HEADING_2,
    })
  );
  sections.push(...parseHtmlToParagraphs(notes.ideas));

  sections.push(
    new Paragraph({
      text: "Language",
      heading: HeadingLevel.HEADING_2,
    })
  );
  sections.push(...parseLanguageTable(notes.language));

  sections.push(
    new Paragraph({
      text: "Communication Strategies",
      heading: HeadingLevel.HEADING_2,
    })
  );
  sections.push(...parseHtmlToParagraphs(notes.communication_strategies));

  return sections;
}

function parseHtmlToParagraphs(html: string): Paragraph[] {
  const paragraphs: Paragraph[] = [];
  const $ = cheerio.load(html);
  const body = $("body").length > 0 ? $("body") : $("*").first();
  const children = body.children();

  children.each((_, node) => {
    const el = $(node);
    const text = el.text().trim();
    if (!text) return;

    paragraphs.push(
      new Paragraph({
        children: [new TextRun({ text: stripHtml(el.html() || text), size: 22 })],
        spacing: { after: 120 },
      })
    );
  });

  if (paragraphs.length === 0 && html.trim()) {
    paragraphs.push(
      new Paragraph({
        children: [new TextRun({ text: stripHtml(html), size: 22 })],
      })
    );
  }

  return paragraphs;
}

function parseLanguageTable(html: string): Paragraph[] {
  const paragraphs: Paragraph[] = [];
  const $ = cheerio.load(html);

  const tables = $("table");
  if (tables.length > 0) {
    tables.each((_, table) => {
      const rows = $(table).find("tr");
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
                    new TextRun({
                      text: cellText,
                      bold: isHeader,
                      size: 20,
                    }),
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

        docRows.push(new TableRow({ children: docCells }));
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
    });
  } else {
    paragraphs.push(...parseHtmlToParagraphs(html));
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
                text: `Generated by Mr. DiscussAI`,
                size: 20,
                color: "666666",
              }),
            ],
            alignment: AlignmentType.CENTER,
          }),
          new Paragraph({ text: "" }),
          ...buildTranscriptParagraphs(items),
          ...buildLearningNotesSections(notes),
        ],
      },
    ],
  });

  return Packer.toBuffer(doc);
}
