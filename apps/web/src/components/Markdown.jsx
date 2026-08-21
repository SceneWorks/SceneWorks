import React from "react";

// Compact, dependency-free Markdown renderer for first-party prompt-guide
// content shipped from /public/prompt-guides. It covers the subset those files
// use — ATX headings, paragraphs, ordered/unordered lists, blockquotes, fenced
// code blocks, GFM pipe tables, and inline bold/italic/code/links. It builds
// React elements directly (never dangerouslySetInnerHTML), so untrusted markup
// cannot inject HTML, and link hrefs are restricted to safe schemes.
//
// Tables joined the subset in sc-17161. Eight shipped guides already used pipe
// tables and every one of them rendered as literal-pipe paragraphs, separator
// row and all — the MiniMax-H3 guide opens with two of them, so its cost table
// (the thing a first-time user most needs to read before starting a two-hour
// render) was the least legible part of the page.

const HEADING = /^(#{1,6})\s+(.*)$/;
const LIST_ITEM = /^\s*([-*+]|\d+\.)\s+/;
const ORDERED_ITEM = /^\s*\d+\.\s+/;
const FENCE = /^```/;
const QUOTE = /^\s*>\s?/;
// A table is a header row followed by a delimiter row — the delimiter is what makes it a table
// rather than a paragraph that happens to contain pipes, which is why both lines are required
// before either is consumed.
const TABLE_ROW = /^\s*\|.*\|\s*$/;
const TABLE_DELIMITER = /^\s*\|(?:\s*:?-{1,}:?\s*\|)+\s*$/;

// Split one pipe row into its cells: drop the leading/trailing pipe, then split on unescaped
// pipes. `\|` is the GFM escape for a literal pipe inside a cell.
function tableCells(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split(/(?<!\\)\|/)
    .map((cell) => cell.trim().replace(/\\\|/g, "|"));
}

// Per-column alignment from the delimiter row (`:--`, `--:`, `:-:`), or undefined for the default.
function tableAlignments(line) {
  return tableCells(line).map((cell) => {
    const left = cell.startsWith(":");
    const right = cell.endsWith(":");
    if (left && right) return "center";
    if (right) return "right";
    if (left) return "left";
    return undefined;
  });
}

function safeHref(url) {
  const trimmed = (url || "").trim();
  if (/^(https?:|mailto:)/i.test(trimmed)) return trimmed;
  if (/^[/#]/.test(trimmed)) return trimmed;
  return undefined;
}

const INLINE_RULES = [
  { re: /`([^`]+)`/, node: (m, key) => <code key={key}>{m[1]}</code> },
  {
    re: /\[([^\]]+)\]\(([^)\s]+)\)/,
    node: (m, key) => {
      const href = safeHref(m[2]);
      if (!href) return <span key={key}>{renderInline(m[1], key)}</span>;
      return (
        <a key={key} href={href} target="_blank" rel="noopener noreferrer">
          {renderInline(m[1], key)}
        </a>
      );
    },
  },
  { re: /\*\*([^*]+)\*\*/, node: (m, key) => <strong key={key}>{renderInline(m[1], key)}</strong> },
  { re: /\*([^*]+)\*/, node: (m, key) => <em key={key}>{renderInline(m[1], key)}</em> },
  { re: /_([^_]+)_/, node: (m, key) => <em key={key}>{renderInline(m[1], key)}</em> },
];

function renderInline(text, keyPrefix) {
  let earliest = null;
  for (const rule of INLINE_RULES) {
    const match = rule.re.exec(text);
    if (match && (!earliest || match.index < earliest.match.index)) {
      earliest = { rule, match };
    }
  }
  if (!earliest) return text;
  const { rule, match } = earliest;
  const before = text.slice(0, match.index);
  const after = text.slice(match.index + match[0].length);
  const nodes = [];
  if (before) nodes.push(before);
  nodes.push(rule.node(match, `${keyPrefix}-${match.index}`));
  const rest = renderInline(after, `${keyPrefix}-r`);
  if (Array.isArray(rest)) nodes.push(...rest);
  else if (rest) nodes.push(rest);
  return nodes;
}

function parseBlocks(content) {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2].trim() });
      i += 1;
      continue;
    }
    if (FENCE.test(line.trim())) {
      i += 1;
      const code = [];
      while (i < lines.length && !FENCE.test(lines[i].trim())) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1; // skip closing fence
      blocks.push({ type: "code", text: code.join("\n") });
      continue;
    }
    if (TABLE_ROW.test(line) && i + 1 < lines.length && TABLE_DELIMITER.test(lines[i + 1])) {
      const header = tableCells(line);
      const align = tableAlignments(lines[i + 1]);
      i += 2;
      const rows = [];
      while (i < lines.length && TABLE_ROW.test(lines[i])) {
        rows.push(tableCells(lines[i]));
        i += 1;
      }
      blocks.push({ type: "table", header, align, rows });
      continue;
    }
    if (LIST_ITEM.test(line)) {
      const ordered = ORDERED_ITEM.test(line);
      const items = [];
      while (i < lines.length && LIST_ITEM.test(lines[i])) {
        items.push(lines[i].replace(LIST_ITEM, ""));
        i += 1;
      }
      blocks.push({ type: "list", ordered, items });
      continue;
    }
    if (QUOTE.test(line)) {
      const quote = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        quote.push(lines[i].replace(QUOTE, ""));
        i += 1;
      }
      blocks.push({ type: "quote", text: quote.join(" ") });
      continue;
    }
    const para = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !HEADING.test(lines[i]) &&
      !LIST_ITEM.test(lines[i]) &&
      !FENCE.test(lines[i].trim()) &&
      !QUOTE.test(lines[i]) &&
      !(TABLE_ROW.test(lines[i]) && i + 1 < lines.length && TABLE_DELIMITER.test(lines[i + 1]))
    ) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push({ type: "paragraph", text: para.join(" ") });
  }
  return blocks;
}

function renderBlock(block, key) {
  switch (block.type) {
    case "heading": {
      const Tag = `h${Math.min(block.level, 6)}`;
      return <Tag key={key}>{renderInline(block.text, key)}</Tag>;
    }
    case "list": {
      const Tag = block.ordered ? "ol" : "ul";
      return (
        <Tag key={key}>
          {block.items.map((item, index) => (
            <li key={`${key}-${index}`}>{renderInline(item, `${key}-${index}`)}</li>
          ))}
        </Tag>
      );
    }
    case "quote":
      return (
        <blockquote key={key}>
          <p>{renderInline(block.text, key)}</p>
        </blockquote>
      );
    case "table":
      return (
        // Wrapped so a wide table scrolls inside its own box rather than widening the modal.
        <div className="markdown-table-wrap" key={key}>
          <table>
            <thead>
              <tr>
                {block.header.map((cell, index) => (
                  <th key={`${key}-h${index}`} style={{ textAlign: block.align[index] }}>
                    {renderInline(cell, `${key}-h${index}`)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, rowIndex) => (
                <tr key={`${key}-r${rowIndex}`}>
                  {block.header.map((_, cellIndex) => (
                    <td key={`${key}-r${rowIndex}c${cellIndex}`} style={{ textAlign: block.align[cellIndex] }}>
                      {renderInline(row[cellIndex] ?? "", `${key}-r${rowIndex}c${cellIndex}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "code":
      return (
        <pre key={key}>
          <code>{block.text}</code>
        </pre>
      );
    case "paragraph":
    default:
      return <p key={key}>{renderInline(block.text, key)}</p>;
  }
}

export function Markdown({ content }) {
  const blocks = parseBlocks(content || "");
  return <div className="markdown-body">{blocks.map((block, index) => renderBlock(block, `b${index}`))}</div>;
}
