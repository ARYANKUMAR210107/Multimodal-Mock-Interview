"""Reusable live voice recorder component for Streamlit (browser-side MediaRecorder + WebSocket chunk transcription)."""
from __future__ import annotations

import json
import os
import streamlit.components.v1 as components


def render_voice_recorder(api_base_url: str, ws_base_url: str, access_token: str, session_id: str, position: int, height: int = 520) -> None:
    """Embed browser recorder with waveform, timer, live transcript, pause/resume/stop and WS streaming."""
    cfg = {
        "apiBaseUrl": api_base_url.rstrip('/'),
        "wsBaseUrl": ws_base_url.rstrip('/'),
        "accessToken": access_token or "",
        "sessionId": session_id,
        "position": position,
    }
    cfg_json = json.dumps(cfg)

    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root { --bg:#0f172a; --card:#111827; --muted:#94a3b8; --text:#e5e7eb; --cyan:#06b6d4; --blue:#3b82f6; --red:#ef4444; --green:#10b981; --border:#243244; }
  body { margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; background: transparent; color: var(--text); }
  .wrap { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
  .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  button { border:1px solid var(--border); background:#0b1220; color:var(--text); padding:8px 12px; border-radius:8px; cursor:pointer; font-weight:600; }
  button.primary { background: linear-gradient(135deg,var(--cyan),var(--blue)); color:#021018; border:none; }
  button.danger { background: rgba(239,68,68,.15); border-color: rgba(239,68,68,.4); color:#fecaca; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .status { margin-top:10px; font-size:13px; color:var(--muted); }
  .status strong { color:var(--text); }
  .pill { display:inline-block; padding:3px 8px; border-radius:999px; font-size:12px; border:1px solid var(--border); margin-left:6px; }
  .pill.rec { color:#fecaca; border-color: rgba(239,68,68,.5); background: rgba(239,68,68,.12); }
  .pill.ok { color:#bbf7d0; border-color: rgba(16,185,129,.5); background: rgba(16,185,129,.12); }
  canvas { width:100%; height:70px; background:#08101d; border:1px solid var(--border); border-radius:8px; margin-top:10px; }
  .timer { font-variant-numeric: tabular-nums; font-weight:700; color:var(--cyan); }
  .label { font-size:12px; color:var(--muted); margin:10px 0 4px; text-transform:uppercase; letter-spacing:.06em; font-weight:700; }
  .box { width:100%; min-height:110px; max-height:180px; overflow:auto; background:#08101d; border:1px solid var(--border); border-radius:8px; padding:10px; white-space:pre-wrap; line-height:1.45; }
  .final { min-height:130px; }
  .hint { font-size:12px; color:var(--muted); margin-top:8px; }
  .err { color:#fecaca; }
  .okmsg { color:#bbf7d0; }
</style>
</head>
<body>
<div class="wrap">
  <div class="row">
    <button id="btnStart" class="primary">Start Recording</button>
    <button id="btnPause" disabled>Pause</button>
    <button id="btnResume" disabled>Resume</button>
    <button id="btnStop" class="danger" disabled>Stop & Finalize</button>
    <button id="btnClear" disabled>Clear</button>
    <button id="btnUpload" title="Upload a local audio file">Upload Audio</button>
    <input id="fileInput" type="file" accept=".wav,.mp3,.m4a,.webm,audio/*" style="display:none" />
    <span class="timer" id="timer">00:00</span>
    <span class="pill" id="statePill">Idle</span>
  </div>
  <div class="status" id="status">Waiting for microphone permission when you start recording.</div>
  <canvas id="wave" width="900" height="140"></canvas>

  <div class="label">Live Transcript (progressive)</div>
  <div id="liveText" class="box"></div>

  <div class="label">Final Reviewed Transcript</div>
  <textarea id="finalText" class="box final" style="width:100%;resize:vertical;" placeholder="Final transcript will appear here after stop/finalization. You can edit it."></textarea>
  <div class="row" style="margin-top:8px;">
    <button id="btnSave" class="primary" disabled>Save Transcript to Interview</button>
    <button id="btnRetry" disabled>Retry Final Transcription</button>
  </div>
  <div class="hint">Use "Save Transcript to Interview" after reviewing/correcting. Recording stays available until you clear it. If live updates are delayed, short-chunk fallback and final full transcription will still run on stop.</div>
  <audio id="playback" controls style="width:100%; margin-top:10px; display:none;"></audio>
</div>
<script>
const CFG = __CFG_JSON__;
const $ = (id)=>document.getElementById(id);
const statusEl = $("status"), liveEl=$("liveText"), finalEl=$("finalText"), pill=$("statePill"), timerEl=$("timer");
const btnStart=$("btnStart"), btnPause=$("btnPause"), btnResume=$("btnResume"), btnStop=$("btnStop"), btnClear=$("btnClear"), btnSave=$("btnSave"), btnRetry=$("btnRetry"), btnUpload=$("btnUpload"), fileInput=$("fileInput");
const canvas=$("wave"), ctx=canvas.getContext('2d'), playback=$("playback");
let audioCtx, analyser, sourceNode, processor, mediaStream, ws;
let recording=false, paused=false, startTs=0, elapsedMs=0, timerInt=null;
let pcmChunks=[], totalSamples=0, sampleRate=16000, lastSentSamples=0;
let liveTranscript="", finalTranscript="", detectedLang="unknown";
let chunkTimer=null, busy=false, finalized=false;

function setState(text, cls, msg, isErr=false, isOk=false){
  pill.textContent=text; pill.className='pill ' + (cls||'');
  if(msg){ statusEl.innerHTML = (isErr?'<span class="err">':'') + (isOk?'<span class="okmsg">':'') + msg + ((isErr||isOk)?'</span>':''); }
}
function fmtTime(ms){ const s=Math.floor(ms/1000); const m=Math.floor(s/60); const r=s%60; return String(m).padStart(2,'0')+':'+String(r).padStart(2,'0'); }
function drawIdle(){ ctx.fillStyle='#08101d'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.strokeStyle='#243244'; ctx.beginPath(); ctx.moveTo(0,canvas.height/2); ctx.lineTo(canvas.width,canvas.height/2); ctx.stroke(); }
function drawWave(){
  if(!analyser) return drawIdle();
  const buf = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(buf);
  ctx.fillStyle='#08101d'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.lineWidth=2; ctx.strokeStyle= recording && !paused ? '#06b6d4' : '#334155'; ctx.beginPath();
  const slice = canvas.width / buf.length; let x=0;
  for(let i=0;i<buf.length;i++){ const v=buf[i]/128.0; const y=v*canvas.height/2; if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); x+=slice; }
  ctx.stroke();
  if(recording) requestAnimationFrame(drawWave); else drawIdle();
}
function updateTimer(){ if(!recording || paused) return; const now=Date.now(); timerEl.textContent=fmtTime(elapsedMs + (now-startTs)); }
function startTimer(){ startTs=Date.now(); clearInterval(timerInt); timerInt=setInterval(updateTimer, 250); }
function stopTimer(){ if(recording && !paused){ elapsedMs += Date.now()-startTs; } clearInterval(timerInt); timerInt=null; timerEl.textContent=fmtTime(elapsedMs); }
function resetTimer(){ clearInterval(timerInt); timerInt=null; elapsedMs=0; startTs=0; timerEl.textContent='00:00'; }

function wsUrl(){
  const base = (CFG.wsBaseUrl || CFG.apiBaseUrl || '').replace(/^http/,'ws');
  return base + '/api/audio/transcribe/ws';
}
function connectWs(){
  return new Promise((resolve,reject)=>{
    try{
      ws = new WebSocket(wsUrl());
      ws.onopen = ()=>{ ws.send(JSON.stringify({type:'start', session_id: CFG.sessionId, position: CFG.position, sample_rate: sampleRate})); resolve(); };
      ws.onerror = ()=>reject(new Error('WebSocket connection failed'));
      ws.onclose = ()=>{};
      ws.onmessage = (ev)=>{
        try{
          const msg = JSON.parse(ev.data);
          if(msg.type==='partial'){
            if(msg.full_text){ liveTranscript = msg.full_text; liveEl.textContent = liveTranscript; }
            else if(msg.text){ liveTranscript = (liveTranscript + ' ' + msg.text).trim(); liveEl.textContent = liveTranscript; }
            if(msg.detected_language) detectedLang = msg.detected_language;
          } else if(msg.type==='final'){
            finalTranscript = msg.text || liveTranscript || '';
            finalEl.value = finalTranscript;
            detectedLang = msg.detected_language || detectedLang;
            finalized = true;
            setState('Completed','ok','Transcription completed. Review and edit the final transcript, then save it.',false,true);
            btnSave.disabled = !finalTranscript.trim();
            btnRetry.disabled = pcmChunks.length===0;
          } else if(msg.type==='status'){
            const m = msg.message || msg.state || '';
            if(msg.state==='listening') setState('Recording','rec', m || 'Listening and transcribing...');
            else if(msg.state==='processing') setState('Processing','rec', m || 'Transcribing completed chunk...');
            else if(msg.state==='finalizing') setState('Finalizing','rec', m || 'Finalizing transcript from complete recording...');
            else if(msg.state==='ready') setState('Ready','', m || 'Ready.');
            else if(msg.state==='completed') setState('Completed','ok', m || 'Transcription completed.', false, true);
            else if(msg.state==='error') setState('Error','', m || 'Transcription error.', true, false);
          } else if(msg.type==='error'){
            setState('Error','', msg.message || 'Transcription failed.', true, false);
            if(msg.partial_text && !finalEl.value){ finalEl.value = msg.partial_text; btnSave.disabled = !msg.partial_text.trim(); }
            btnRetry.disabled = pcmChunks.length===0;
          }
        }catch(e){}
      };
    }catch(e){ reject(e); }
  });
}
function floatTo16BitPCM(float32){
  const out = new Int16Array(float32.length);
  for(let i=0;i<float32.length;i++){ let s=Math.max(-1,Math.min(1,float32[i])); out[i]= s < 0 ? s*0x8000 : s*0x7FFF; }
  return out;
}
function concatFloat32(chunks){ let len=0; for(const c of chunks) len+=c.length; const out=new Float32Array(len); let off=0; for(const c of chunks){ out.set(c,off); off+=c.length; } return out; }
function encodeWav(samples, sr){
  const buffer = new ArrayBuffer(44 + samples.length*2); const view = new DataView(buffer);
  function ws(s,o){ for(let i=0;i<s.length;i++) view.setUint8(o+i,s.charCodeAt(i)); }
  ws('RIFF',0); view.setUint32(4,36+samples.length*2,true); ws('WAVE',8); ws('fmt ',12); view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
  view.setUint32(24,sr,true); view.setUint32(28,sr*2,true); view.setUint16(32,2,true); view.setUint16(34,16,true); ws('data',36); view.setUint32(40,samples.length*2,true);
  let off=44; for(let i=0;i<samples.length;i++,off+=2){ let s=Math.max(-1,Math.min(1,samples[i])); view.setInt16(off, s<0?s*0x8000:s*0x7FFF, true); }
  return new Blob([view], {type:'audio/wav'});
}
function arrayBufferToBase64(buffer){ let binary=''; const bytes=new Uint8Array(buffer); const chunk=0x8000; for(let i=0;i<bytes.length;i+=chunk){ binary += String.fromCharCode.apply(null, bytes.subarray(i,i+chunk)); } return btoa(binary); }
async function sendChunk(force=false){
  if(!ws || ws.readyState!==1) return;
  if(paused && !force) return;
  const newSamples = totalSamples - lastSentSamples;
  const target = sampleRate * 4; // ~4s chunks
  if(newSamples < target && !force) return;
  // send last window (up to ~5s) for incremental stability; server keeps full audio
  const all = concatFloat32(pcmChunks);
  const window = all.slice(Math.max(0, all.length - Math.floor(sampleRate*5)));
  const pcm16 = floatTo16BitPCM(window);
  const b64 = arrayBufferToBase64(pcm16.buffer);
  ws.send(JSON.stringify({type:'chunk', audio_b64:b64, sample_rate: sampleRate, is_final_chunk: !!force}));
  lastSentSamples = totalSamples;
}
function startChunkLoop(){ clearInterval(chunkTimer); chunkTimer=setInterval(()=>{ sendChunk(false).catch(()=>{}); }, 1200); }
function stopChunkLoop(){ clearInterval(chunkTimer); chunkTimer=null; }

async function startRecording(){
  try{
    setState('Requesting mic','', 'Requesting microphone permission...');
    mediaStream = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true, noiseSuppression:true, autoGainControl:true}});
  }catch(e){
    setState('Mic blocked','', 'Microphone access failed: ' + (e && e.message ? e.message : 'permission denied or unavailable') + '. Please allow microphone access and retry.', true, false);
    return;
  }
  try{
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate:16000});
  }catch(e){ audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
  sampleRate = audioCtx.sampleRate || 16000;
  analyser = audioCtx.createAnalyser(); analyser.fftSize = 2048;
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  pcmChunks=[]; totalSamples=0; lastSentSamples=0; liveTranscript=''; finalTranscript=''; finalized=false;
  liveEl.textContent=''; finalEl.value=''; playback.style.display='none'; playback.src='';
  processor.onaudioprocess = (e)=>{
    if(!recording || paused) return;
    const input = e.inputBuffer.getChannelData(0);
    // copy
    const copy = new Float32Array(input.length); copy.set(input);
    pcmChunks.push(copy); totalSamples += copy.length;
  };
  sourceNode.connect(analyser);
  analyser.connect(processor);
  processor.connect(audioCtx.destination);
  try{ await connectWs(); }catch(e){ setState('Error','', 'Could not connect to transcription service WebSocket. You can still record and use final transcription retry if backend HTTP is available.', true, false); }
  recording=true; paused=false; resetTimer(); startTimer(); drawWave(); startChunkLoop();
  btnStart.disabled=true; btnPause.disabled=false; btnResume.disabled=true; btnStop.disabled=false; btnClear.disabled=true; btnSave.disabled=true; btnRetry.disabled=true; btnUpload.disabled=true;
  setState('Recording','rec','Recording started. Listening and transcribing...');
}
function pauseRecording(){ if(!recording || paused) return; paused=true; stopTimer(); btnPause.disabled=true; btnResume.disabled=false; setState('Paused','', 'Recording paused. Resume to continue speaking.'); }
function resumeRecording(){ if(!recording || !paused) return; paused=false; startTimer(); btnPause.disabled=false; btnResume.disabled=true; setState('Recording','rec','Recording resumed. Listening and transcribing...'); drawWave(); }
async function stopRecording(){
  if(!recording) return;
  paused=false; stopTimer(); stopChunkLoop();
  recording=false; btnPause.disabled=true; btnResume.disabled=true; btnStop.disabled=true; btnStart.disabled=false; btnClear.disabled=false; btnUpload.disabled=false;
  try{ if(ws && ws.readyState===1){ await sendChunk(true); ws.send(JSON.stringify({type:'finalize'})); } }
  catch(e){}
  setState('Finalizing','rec','Finalizing transcript from complete recording...');
  try{
    if(processor){ processor.disconnect(); processor.onaudioprocess=null; }
    if(analyser) analyser.disconnect();
    if(sourceNode) sourceNode.disconnect();
  }catch(e){}
  try{ if(mediaStream){ mediaStream.getTracks().forEach(t=>t.stop()); } }catch(e){}
  try{ if(audioCtx && audioCtx.state!=='closed'){ await audioCtx.close(); } }catch(e){}
  const all = concatFloat32(pcmChunks);
  if(all.length>0){ const wav = encodeWav(all, sampleRate); const url=URL.createObjectURL(wav); playback.src=url; playback.style.display='block'; }
  // If WS final never arrives, enable retry via HTTP using saved blob through backend endpoint by asking user? fallback: enable save if live transcript exists
  setTimeout(()=>{
    if(!finalized){ if(!finalEl.value && liveTranscript){ finalEl.value = liveTranscript; }
      if(finalEl.value.trim()){ setState('Needs review','', 'Final WebSocket response not received. Live/partial transcript is available; you can edit, save, or retry final transcription.', false, false); }
      btnSave.disabled = !finalEl.value.trim(); btnRetry.disabled = pcmChunks.length===0; }
  }, 6000);
}
async function retryFinal(){
  if(pcmChunks.length===0){ setState('Error','', 'No recording available to retry.', true, false); return; }
  setState('Finalizing','rec','Retrying final transcription from complete recording...');
  const all = concatFloat32(pcmChunks); const wav = encodeWav(all, sampleRate);
  try{
    const fd = new FormData(); fd.append('file', wav, 'recording.wav');
    const resp = await fetch((CFG.apiBaseUrl||'') + '/api/audio/transcribe', { method:'POST', headers: CFG.accessToken ? {'Authorization':'Bearer '+CFG.accessToken} : {}, body: fd });
    const data = await resp.json().catch(()=>({detail:'Invalid response'}));
    if(!resp.ok){ throw new Error((data && (data.detail||data.message)) || ('HTTP '+resp.status)); }
    finalTranscript = data.transcript || finalEl.value || liveTranscript || '';
    finalEl.value = finalTranscript; finalized=true; detectedLang = data.detected_language || detectedLang;
    setState('Completed','ok','Transcription completed after retry. Review and edit the final transcript, then save it.', false, true);
    btnSave.disabled = !finalTranscript.trim();
  }catch(e){ setState('Error','', 'Retry failed: ' + (e.message||e), true, false); btnRetry.disabled=false; }
}
async function saveTranscript(){
  const text = (finalEl.value || '').trim();
  if(!text){ setState('Error','', 'Transcript is empty. Nothing to save.', true, false); return; }
  setState('Saving','', 'Saving transcript to the current interview question...');
  const payload = {
    response_mode: 'voice',
    original_transcript: (liveTranscript || text),
    corrected_transcript: text,
    answer_text: text
  };
  try{
    const url = (CFG.apiBaseUrl||'') + '/api/sessions/' + encodeURIComponent(CFG.sessionId) + '/questions/' + encodeURIComponent(CFG.position) + '/response';
    const resp = await fetch(url, { method:'PUT', headers: Object.assign({'Content-Type':'application/json'}, CFG.accessToken?{'Authorization':'Bearer '+CFG.accessToken}:{}), body: JSON.stringify(payload) });
    const data = await resp.json().catch(()=>({}));
    if(!resp.ok){ throw new Error((data && data.detail) || ('HTTP '+resp.status)); }
    setState('Saved','ok','Transcript saved to this interview question. You can still edit and save again, or run voice assessment.', false, true);
  }catch(e){ setState('Error','', 'Could not save transcript: ' + (e.message||e), true, false); }
}
function clearAll(){
  if(recording) return;
  pcmChunks=[]; totalSamples=0; lastSentSamples=0; liveTranscript=''; finalTranscript=''; finalized=false; detectedLang='unknown';
  liveEl.textContent=''; finalEl.value=''; playback.src=''; playback.style.display='none';
  btnSave.disabled=true; btnRetry.disabled=true; btnClear.disabled=true; resetTimer(); drawIdle();
  try{ if(ws && ws.readyState===1){ ws.send(JSON.stringify({type:'reset'})); ws.close(); } }catch(e){}
  ws=null; setState('Ready','', 'Recording cleared. Ready for a new attempt.');
}
async function handleUpload(file){
  if(!file) return;
  clearAll();
  setState('Processing','rec','Audio file selected. Running transcription...');
  try{
    const fd = new FormData(); fd.append('file', file, file.name || 'upload.wav');
    const resp = await fetch((CFG.apiBaseUrl||'') + '/api/audio/transcribe', { method:'POST', headers: CFG.accessToken ? {'Authorization':'Bearer '+CFG.accessToken} : {}, body: fd });
    const data = await resp.json().catch(()=>({detail:'Invalid response'}));
    if(!resp.ok){ throw new Error((data && (data.detail||data.message)) || ('HTTP '+resp.status)); }
    finalTranscript = data.transcript || ''; finalEl.value = finalTranscript; liveEl.textContent = finalTranscript; liveTranscript = finalTranscript;
    detectedLang = data.detected_language || 'unknown'; finalized=true;
    btnSave.disabled = !finalTranscript.trim(); btnClear.disabled=false;
    playback.src = URL.createObjectURL(file); playback.style.display='block';
    setState('Completed','ok','Upload transcribed. Review/edit the transcript and save it to the interview.', false, true);
  }catch(e){ setState('Error','', 'Upload transcription failed: ' + (e.message||e), true, false); btnClear.disabled=false; }
}

btnStart.onclick=startRecording; btnPause.onclick=pauseRecording; btnResume.onclick=resumeRecording; btnStop.onclick=stopRecording; btnClear.onclick=clearAll; btnRetry.onclick=retryFinal; btnSave.onclick=saveTranscript;
btnUpload.onclick=()=>fileInput.click(); fileInput.onchange=(e)=>handleUpload(e.target.files && e.target.files[0]);
finalEl.addEventListener('input', ()=>{ btnSave.disabled = !finalEl.value.trim(); });
window.addEventListener('beforeunload', ()=>{ try{ if(ws) ws.close(); }catch(e){} try{ if(mediaStream) mediaStream.getTracks().forEach(t=>t.stop()); }catch(e){} });
drawIdle();
</script>
</body>
</html>
"""
    html = html.replace("__CFG_JSON__", cfg_json)
    components.html(html, height=height, scrolling=True)
