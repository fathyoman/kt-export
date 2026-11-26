import { phonemize, tokenize, splitText } from './g2p.js';
const $ = (id) => document.getElementById(id);
const log = (m) => { $('log').textContent += m + '\n'; $('log').scrollTop = 1e9; };
let ort, sess, voice, ac, nextT = 0, stopFlag = false, backend = 'wasm';
const MODEL = 'kt_fp16.onnx', VOICE = 'voice.bin';
window.addEventListener('error', (e) => log('ERR ' + e.message));
window.addEventListener('unhandledrejection', (e) => log('ERR ' + (e.reason && e.reason.message || e.reason)));
$('gk').value = localStorage.getItem('gk') || '';
const SAMPLES = { s1: 'إِزَّيَّك يَا صَاحْبِي؟ عَامِل إِيه النَّهَارْدَه؟', s2: 'الْجَوّ حِلْو أَوِي بَرَّه، تِعَالَى نِتْمَشَّى شُوَيَّة.', s3: 'الِاجْتِمَاع بِتَاع بُكْرَه اِتْأَجِّل لِحَدّ يُوم التَّلَات.' };
for (const k in SAMPLES) $(k).onclick = () => { $('tx').value = SAMPLES[k]; };
async function fetchProg(url, label) {
  const c = await caches.open('kt-v1'); let r = await c.match(url);
  if (r) { log(label + ': from cache'); return new Uint8Array(await r.arrayBuffer()); }
  r = await fetch(url); if (!r.ok) throw new Error(url + ' ' + r.status);
  const total = +r.headers.get('content-length') || 0, rd = r.body.getReader(), parts = []; let got = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); got += value.length; if (total) { $('pb').style.width = (100 * got / total) + '%'; $('st').textContent = label + ' ' + (got / 1e6).toFixed(0) + '/' + (total / 1e6).toFixed(0) + ' MB'; } }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  try { await c.put(url, new Response(buf)); } catch (e) { log('cache put failed: ' + e.message); }
  log(label + ': downloaded ' + (got / 1e6).toFixed(1) + ' MB'); return buf;
}
$('load').onclick = async () => {
  $('load').disabled = true; backend = $('be').value; const t0 = performance.now();
  try {
    ort = await import(backend === 'webgpu' ? './ort/ort.webgpu.min.mjs' : './ort/ort.wasm.min.mjs');
    ort.env.wasm.wasmPaths = new URL('./ort/', location.href).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    log('backend=' + backend + ' threads=' + ort.env.wasm.numThreads + ' coi=' + self.crossOriginIsolated);
    const vb = await fetchProg(VOICE, 'voice'); voice = new Float32Array(vb.buffer, vb.byteOffset, vb.byteLength / 4);
    const mb = await fetchProg(MODEL, 'model');
    $('st').textContent = 'تجهيز النموذج...';
    sess = await ort.InferenceSession.create(mb, { executionProviders: backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'], graphOptimizationLevel: 'all' });
    $('st').textContent = 'جاهز (' + ((performance.now() - t0) / 1000).toFixed(1) + ' ثانية)'; $('say').disabled = false; log('session ready');
  } catch (e) { $('st').textContent = 'فشل التحميل: ' + e.message; log('ERR ' + (e.stack || e)); $('load').disabled = false; }
};
async function synth(text) {
  const ipa = await phonemize(text); if (!ipa) return null;
  const { ids, styleIndex } = tokenize(ipa);
  const feeds = { input_ids: new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]), style: new ort.Tensor('float32', voice.slice(styleIndex * 256, styleIndex * 256 + 256), [1, 256]), speed: new ort.Tensor('float32', new Float32Array([1]), [1]) };
  const t0 = performance.now(); const out = await sess.run(feeds); const w = out.waveform.data; const dt = (performance.now() - t0) / 1000;
  return { w, dt, dur: w.length / 24000, ipa };
}
function play(w) { ac = ac || new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 }); const b = ac.createBuffer(1, w.length, 24000); b.copyToChannel(w, 0); const s = ac.createBufferSource(); s.buffer = b; s.connect(ac.destination); nextT = Math.max(nextT, ac.currentTime + 0.05); s.start(nextT); nextT += b.duration; }
$('say').onclick = async () => {
  stopFlag = false; $('say').disabled = true; if (ac && ac.state === 'suspended') await ac.resume(); nextT = 0;
  try { const parts = splitText($('tx').value); let td = 0, tt = 0;
    for (const p of parts) { if (stopFlag) break; const r = await synth(p); if (!r) continue; play(r.w); td += r.dur; tt += r.dt; log('ipa: ' + r.ipa); $('stats').textContent = 'مدة الصوت ' + td.toFixed(1) + ' ث | زمن التوليد ' + tt.toFixed(1) + ' ث | RTF ' + (tt / td).toFixed(2) + ' (أقل من 1 = أسرع من الكلام)'; }
  } catch (e) { log('ERR ' + (e.stack || e)); } $('say').disabled = false;
};
$('stop').onclick = () => { stopFlag = true; if (ac) { ac.close(); ac = null; } };
$('tash').onclick = async () => {
  const key = $('gk').value.trim(); if (!key) return log('ERR ادخل المفتاح'); localStorage.setItem('gk', key);
  const src = $('tx').value.replace(/[\u064B-\u0652\u0670]/g, '');
  const prompt = 'شكّل النص التالي تشكيلاً كاملاً بالنطق المصري العامي (القاهري) كما يُنطق فعلاً وليس بالفصحى، واستخدم السكون والشدة. لا تغيّر الكلمات ولا تضف شرحاً. أرجع النص المشكّل فقط.\nمثال:\nالنص: انا بحب الاكل المصري اوي، وماما بتعمل محشي وملوخية كل يوم جمعة.\nالمشكّل: أَنَا بَحِبّ الْأَكْل الْمَصْرِي أَوِي، وِمَامَا بِتِعْمِل مَحْشِي وِمُلُوخِيَّة كُلّ يُوم جُمْعَة.\n\nالنص: ' + src + '\nالمشكّل:';
  $('tash').disabled = true;
  for (const m of ['gemini-flash-latest', 'gemini-2.5-flash', 'gemini-2.0-flash']) {
    try { const r = await fetch('https://generativelanguage.googleapis.com/v1beta/models/' + m + ':generateContent?key=' + encodeURIComponent(key), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.2 } }) });
      if (!r.ok) { log('tashkeel ' + m + ' -> ' + r.status); continue; }
      const j = await r.json(); const t = j.candidates && j.candidates[0].content.parts.map((p) => p.text).join('').trim(); if (t) { $('tx').value = t; log('tashkeel ok (' + m + ')'); break; } } catch (e) { log('tashkeel err ' + e.message); }
  } $('tash').disabled = false;
};
