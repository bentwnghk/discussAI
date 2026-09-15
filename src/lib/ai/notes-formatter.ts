import type { LearningNotes } from "@/types";

const DANGEROUS_BLOCK_RE =
  /<\s*(script|style|iframe|object|embed)\b[^>]*>[\s\S]*?<\s*\/\s*\1\s*>/gi;
const DANGEROUS_TAG_RE =
  /<\s*\/?\s*(script|style|iframe|object|embed|link|meta|base)\b[^>]*>/gi;
const OPEN_TAG_RE = /<([a-z][^>]*)>/gi;
const EVENT_ATTR_RE = /\son\w+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi;
const URL_ATTR_RE = /(href|src|xlink:href)\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi;

const CODE_FENCE_RE = /^\s*```[a-z]*\s*\r?\n?([\s\S]*?)\r?\n?\s*```\s*$/i;

const HTML_BLOCK_RE = /<\s*(br|p|div|li|ul|ol|table|h[1-6])\b/i;
const ANY_HTML_TAG_RE = /<\s*[a-z][^>]*>/i;

const ITEM_START_RE =
  /^(?:(?:<[^>]+>\s*)*(?:\d{1,2}[.)]\s|•\s|·\s|[-*]\s)|<table\b)/i;

// Split points for "wall of text" content: sentence-ending punctuation
// followed by a numbered item like "2. Maintaining discussion ...".
const WALL_SPLIT_RE = /([.!?。！？；;…"“”‘’])\s+(?=\d{1,2}\.\s)/g;
const WALL_DELIM = "\u2028";

const MARKDOWN_BOLD_RE = /\*\*\s*([^*\n]+?)\s*\*\*/g;
const MARKDOWN_EM_RE =
  /(^|[\s(（\[{"“‘'])\*([^*\n]+?)\*(?=$|[\s.,!?;:)}\]）)」、<"。，！？；：”…\s])/g;

const TABLE_LINE_RE = /^\s*\|.*\|\s*$/;
const TABLE_DIVIDER_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

function sanitizeHtml(html: string): string {
  return html
    .replace(DANGEROUS_BLOCK_RE, "")
    .replace(DANGEROUS_TAG_RE, "")
    .replace(OPEN_TAG_RE, (match: string, attrs: string) => {
      const cleaned = attrs
        .replace(EVENT_ATTR_RE, "")
        .replace(URL_ATTR_RE, (attr: string) =>
          /javascript\s*:/i.test(attr) ? "" : attr
        );
      return `<${cleaned}>`;
    });
}

function convertMarkdownInline(text: string): string {
  let out = text.replace(MARKDOWN_BOLD_RE, "<strong>$1</strong>");
  out = out.replace(MARKDOWN_EM_RE, "$1<em>$2</em>");
  return out;
}

function escapeHtmlText(text: string): string {
  return text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function splitTableRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function convertMarkdownTables(text: string): string {
  if (!text.includes("|")) return text;
  const lines = text.split(/\r?\n/);
  const out: string[] = [];
  let i = 0;
  while (i < lines.length) {
    if (
      TABLE_LINE_RE.test(lines[i]) &&
      i + 1 < lines.length &&
      TABLE_DIVIDER_RE.test(lines[i + 1])
    ) {
      const rows: string[][] = [splitTableRow(lines[i])];
      i += 2;
      while (i < lines.length && TABLE_LINE_RE.test(lines[i])) {
        rows.push(splitTableRow(lines[i]));
        i++;
      }
      const [head, ...body] = rows;
      const headCells = head.map((cell) => `<th>${cell}</th>`).join("");
      const bodyRows = body
        .map(
          (row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`
        )
        .join("");
      out.push(`<table><tr>${headCells}</tr>${bodyRows}</table>`);
      continue;
    }
    out.push(lines[i]);
    i++;
  }
  return out.join("\n");
}

function wallSplit(text: string): string[] {
  return text
    .replace(WALL_SPLIT_RE, `$1${WALL_DELIM}`)
    .split(WALL_DELIM)
    .map((part) => part.trim())
    .filter(Boolean);
}

function structureLines(text: string): string {
  if (!/\r?\n/.test(text.trim())) {
    const parts = wallSplit(text);
    return parts.length >= 3 ? parts.join("<br><br>") : text.trim();
  }

  const blocks: string[] = [];
  const paragraphs = text.split(/\r?\n\s*\r?\n/);
  for (const paragraph of paragraphs) {
    const lines = paragraph
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length === 0) continue;

    const hasListLines = lines.some((line) => ITEM_START_RE.test(line));
    if (hasListLines) {
      const items: string[] = [];
      let current: string[] = [];
      for (const raw of lines) {
        const line = raw.replace(/^[-*]\s+/, "• ");
        if (ITEM_START_RE.test(line)) {
          if (current.length > 0) items.push(current.join("<br>"));
          current = [line];
        } else {
          current.push(line);
        }
      }
      if (current.length > 0) items.push(current.join("<br>"));
      blocks.push(items.join("<br><br>"));
    } else {
      const wallParts = lines.flatMap((line) => wallSplit(line));
      if (wallParts.length >= 3 && wallParts.length > lines.length) {
        blocks.push(wallParts.join("<br><br>"));
      } else {
        blocks.push(lines.join("<br>"));
      }
    }
  }
  return blocks.length > 0 ? blocks.join("<br><br>") : text.trim();
}

export function normalizeNotesHtml(raw: string | null | undefined): string {
  if (!raw) return "";
  let content = raw.trim();
  if (!content) return "";

  const fenced = content.match(CODE_FENCE_RE);
  if (fenced?.[1]) content = fenced[1].trim();

  const sanitized = sanitizeHtml(content);
  if (!sanitized) return "";

  if (HTML_BLOCK_RE.test(sanitized)) {
    return convertMarkdownInline(sanitized);
  }
  if (ANY_HTML_TAG_RE.test(sanitized)) {
    return structureLines(convertMarkdownInline(sanitized));
  }

  let text = escapeHtmlText(sanitized);
  text = convertMarkdownInline(text);
  text = convertMarkdownTables(text);
  text = text.replace(/^\s{0,3}#{1,6}\s+(.+)$/gm, "<strong>$1</strong>");
  return structureLines(text);
}

export function normalizeLearningNotes<T extends LearningNotes>(notes: T): T {
  return {
    ...notes,
    ideas: normalizeNotesHtml(notes.ideas),
    language: normalizeNotesHtml(notes.language),
    communication_strategies: normalizeNotesHtml(
      notes.communication_strategies
    ),
  };
}
