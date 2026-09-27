// Minimal, safe Markdown for model answers: everything is escaped first,
// then a small subset is re-introduced. Citations [n] / [Vn] become buttons.
import { esc } from "./ui.js";

function inline(text, citeCount) {
  let s = esc(text);
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/__([^_]+)__/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
  s = s.replace(/\[(V?)(\d+)\]/g, (m, v, n) => {
    const valid = v ? true : citeCount === undefined || (+n >= 1 && +n <= citeCount);
    return `<button class="cite${valid ? "" : " invalid"}" data-cite="${v}${n}" title="${valid ? "Show source" : "Not one of the sources"}">${v}${n}</button>`;
  });
  s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[+i]}</code>`);
  return s;
}

// marks: [{text, title}] — sentences to highlight (e.g. statements the
// claim check found unsupported). They're wrapped in placeholder characters
// before rendering and turned into <mark> afterwards, so escaping and the
// Markdown rules apply to them as usual.
export function renderMarkdown(src, citeCount, marks = []) {
  let text = String(src || "").replace(/\r\n?/g, "\n");
  marks.forEach((m, i) => {
    const at = m.text ? text.indexOf(m.text) : -1;
    if (at >= 0) text = text.slice(0, at) + `\u0001${i}\u0003` + m.text + "\u0002" + text.slice(at + m.text.length);
  });
  return renderBlocks(text, citeCount)
    .replace(/\u0001(\d+)\u0003/g, (_, i) => `<mark class="claim-flag" title="${esc(marks[+i].title || "")}">`)
    .replace(/\u0002/g, "</mark>");
}

function renderBlocks(src, citeCount) {
  const lines = src.split("\n");
  const out = [];
  let i = 0;
  const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*(```|~~~)/);
    if (fence) {
      const body = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence[1])) body.push(lines[i++]);
      i++;
      out.push(`<pre><code>${esc(body.join("\n"))}</code></pre>`);
      continue;
    }
    if (!line.trim()) { i++; continue; }
    const hd = line.match(/^(#{1,4})\s+(.*)$/);
    if (hd) { out.push(`<h${hd[1].length + 1}>${inline(hd[2], citeCount)}</h${hd[1].length + 1}>`); i++; continue; }
    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
      out.push("<table><thead><tr>" + head.map((c) => `<th>${inline(c, citeCount)}</th>`).join("") + "</tr></thead><tbody>"
        + rows.map((r) => "<tr>" + r.map((c) => `<td>${inline(c, citeCount)}</td>`).join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }
    const li = line.match(/^\s*([-*+]|\d+[.)])\s+(.*)$/);
    if (li) {
      const ordered = /\d/.test(li[1]);
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^\s*([-*+]|\d+[.)])\s+(.*)$/);
        if (m && /\d/.test(m[1]) === ordered) { items.push(m[2]); i++; }
        else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) { items[items.length - 1] += " " + lines[i].trim(); i++; }
        else break;
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>` + items.map((t) => `<li>${inline(t, citeCount)}</li>`).join("") + `</${tag}>`);
      continue;
    }
    if (line.startsWith(">")) {
      const q = [];
      while (i < lines.length && lines[i].startsWith(">")) q.push(lines[i++].replace(/^>\s?/, ""));
      out.push(`<blockquote>${inline(q.join(" "), citeCount)}</blockquote>`);
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^\s*(```|~~~|#{1,4}\s|[-*+]\s|\d+[.)]\s|>)/.test(lines[i])
           && !(lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]))) para.push(lines[i++]);
    if (!para.length) { para.push(lines[i++]); }
    out.push(`<p>${para.map((p) => inline(p, citeCount)).join("<br>")}</p>`);
  }
  return out.join("");
}
