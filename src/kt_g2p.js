// JS port of kemetone/g2p.py (Rabe3/kemetone, Apache-2.0). espeak-ng WASM + front end from Oddadmix/nabra-tts (Apache-2.0).
import { espeakPhonemize } from './g2p/espeak.ts';
import { cleanPhonemes } from './g2p/arabic.ts';
import { LEX, VOCAB } from './data.js';

const LEXSET = {};
for (const k of Object.keys(LEX)) LEXSET[k] = new Set(LEX[k]);
const ALEF = { 'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ة': 'ه', 'ى': 'ي' };
const AR_LETTERS = new Set('ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآئؤةى');
const SHADDA = '\u0651';
const FINAL_MARKS = new Set(['\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', '\u0652', '\u0670']);
const ALL_HARAKAT = new Set([...FINAL_MARKS, SHADDA]);
const KEPT_FINAL = new Set([SHADDA, '\u0652']);
const skeleton = (w) => Array.from(w.replace(/\p{Mn}/gu, '')).map((c) => ALEF[c] || c).filter((c) => AR_LETTERS.has(c)).join('');
const stripIrab = (text) => text.split(' ').map((word) => {
  const cs = Array.from(word); let i = cs.length;
  while (i > 0 && ALL_HARAKAT.has(cs[i - 1])) i--;
  return cs.slice(0, i).join('') + cs.slice(i).filter((c) => KEPT_FINAL.has(c)).join('');
}).join(' ');
const SWAP = /([\u064B-\u0652\u0670])(\u0651)/gu, DUP = /([\u064B-\u0652\u0670])\1+/gu;
function normMarks(t) { let p = null; while (p !== t) { p = t; t = t.replace(SWAP, '$2$1'); } return t.replace(DUP, '$1'); }
function normalizeText(t) { t = t.replace(/\[[^\]]*\]/g, ' ').replace(/\u0640/g, ''); t = t.replace(/[A-Za-z]+/g, ' '); return t.replace(/\s+/g, ' ').trim(); }
const PROCL = ['وال', 'فال', 'بال', 'كال', 'لل', 'ال', 'و', 'ف', 'ب', 'ك', 'ل'];
function lexLookup(skel, lex) {
  if (lex.has(skel)) return true;
  const seen = new Set([skel]); let frontier = [skel];
  for (let r = 0; r < 2; r++) {
    const nxt = [];
    for (const w of frontier) for (const p of PROCL) if (w.startsWith(p) && w.length - p.length >= 3) {
      const s = w.slice(p.length); if (lex.has(s)) return true; if (!seen.has(s)) { seen.add(s); nxt.push(s); }
    }
    frontier = nxt;
  }
  return false;
}
const QNEXT = new Set(['سوره', 'سور']);
function rewriteWord(ph, word, prev) {
  const skel = skeleton(word);
  ph = ph.split('ʤ').join('ɡ');
  if (ph.includes('q')) { const keep = lexLookup(skel, LEXSET['ق']) || (prev != null && QNEXT.has(skeleton(prev))); if (!keep) ph = ph.split('q').join('ʔ'); }
  if (ph.includes('θ')) ph = ph.split('θ').join(lexLookup(skel, LEXSET['ث']) ? 's' : 't');
  if (ph.includes('ð')) {
    const hd = skel.includes('ذ'), hz = skel.includes('ظ');
    if (hd && hz) { /* ambiguous: leave */ }
    else if (hz) ph = ph.split('ð').join(lexLookup(skel, LEXSET['ظ']) ? 'dˤ' : 'zˤ');
    else if (hd) ph = ph.split('ð').join(lexLookup(skel, LEXSET['ذ']) ? 'd' : 'z');
  }
  return ph;
}
export async function phonemize(text) {
  text = normalizeText(text); if (!text) return '';
  text = stripIrab(normMarks(text));
  const raw = await espeakPhonemize(text);
  const words = text.split(' '), parts = raw.split(' ');
  let out;
  if (parts.length === words.length) out = parts.map((p, i) => rewriteWord(p, words[i], i ? words[i - 1] : null)).join(' ');
  else { const o = []; for (let i = 0; i < words.length; i++) o.push(rewriteWord(await espeakPhonemize(words[i]), words[i], i ? words[i - 1] : null)); out = o.join(' '); }
  return cleanPhonemes(out);
}
export function tokenize(ipa) {
  const cps = Array.from(ipa).slice(0, 510); const ids = [0];
  for (const ch of cps) { const id = VOCAB[ch]; if (id !== undefined) ids.push(id); }
  ids.push(0); return { ids, styleIndex: Math.min(cps.length - 1, 509) };
}
export function splitText(t) {
  const out = []; for (const line of t.split(/\n+/)) { const s = line.split(/([.!?؟]+)/); let cur = '';
    for (let i = 0; i < s.length; i += 2) { const sent = s[i] + (s[i + 1] || ''); if ((cur + sent).length <= 250) cur += sent; else { if (cur.trim()) out.push(cur.trim()); cur = sent; } }
    if (cur.trim()) out.push(cur.trim()); }
  return out;
}
