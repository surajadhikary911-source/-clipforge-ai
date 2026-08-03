import os, re, tempfile, zipfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel
import subprocess

st.set_page_config(page_title="ClipForge AI", page_icon="✂️", layout="centered")

@st.cache_resource
def get_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def ffmpeg(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])
    return p

HOOK = {"how","why","secret","mistake","never","always","truth","actually","crazy","insane",
        "best","worst","important","nobody","everyone","watch","listen","remember","problem",
        "lesson","fact","real"}
EMOTION = {"love","hate","fear","shocked","crazy","amazing","terrible","beautiful","fail",
           "failed","win","won","lose","lost","embarrassing","angry","surprise","surprised",
           "unbelievable","dangerous","wrong"}

def score(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words: return 0
    h = sum(w in HOOK for w in words) / len(words) * 100
    e = sum(w in EMOTION for w in words) / len(words) * 100
    q = 8 if "?" in text else 0
    num = 5 if re.search(r"\b\d+\b", text) else 0
    direct = 3 if any(x in text.lower() for x in ["you ", "your ", "i ", "we "]) else 0
    return min(100, 35 + h*1.7 + e*1.8 + q + num + direct)

def overlap(a,b):
    return max(0, min(a[1],b[1])-max(a[0],b[0]))

def candidates(segs, lo=20, hi=60):
    out=[]
    for i,s in enumerate(segs):
        start=s[0]; text=""
        for j in range(i,len(segs)):
            end=segs[j][1]
            if end-start>hi: break
            text += (" " if text else "") + segs[j][2]
            if end-start>=lo:
                sc=score(text)
                if re.search(r"[.!?]$", text.strip()): sc=min(100,sc+5)
                out.append((start,end,text.strip(),sc))
    out.sort(key=lambda x:x[3],reverse=True)
    chosen=[]
    for c in out:
        if all(overlap((c[0],c[1]),(x[0],x[1])) < 4 for x in chosen):
            chosen.append(c)
        if len(chosen)==10: break
    return sorted(chosen,key=lambda x:x[3],reverse=True)

def render(src, item, n, folder):
    start,end,_,_=item
    out=folder/f"short_{n:02d}.mp4"
    # Simple 9:16 center crop. Production version can add speaker tracking.
    vf="scale=1080:-2,crop=1080:1920"
    ffmpeg(["ffmpeg","-y","-ss",str(max(0,start-.15)),"-i",str(src),
            "-t",str(end-start+.3),"-vf",vf,
            "-c:v","libx264","-preset","veryfast","-crf","23",
            "-c:a","aac","-b:a","128k","-movflags","+faststart",str(out)])
    return out

st.title("✂️ ClipForge AI")
st.write("Turn a long video into 10 ranked Shorts — directly from your iPhone browser.")
st.info("Best for podcasts, interviews, commentary, gaming, and talking-head videos.")

uploaded=st.file_uploader("Upload your long-form video",type=["mp4","mov","m4v","webm","mkv"])

if uploaded:
    if st.button("🚀 Find my 10 best clips",type="primary",use_container_width=True):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); src=td/uploaded.name
            src.write_bytes(uploaded.getbuffer())
            with st.status("AI is analyzing your video...",expanded=True) as status:
                st.write("🎙️ Transcribing...")
                model=get_model()
                segments,_=model.transcribe(str(src),vad_filter=True)
                segs=[(float(s.start),float(s.end),s.text.strip()) for s in segments if s.text.strip()]
                st.write("🧠 Finding hooks and high-potential moments...")
                picks=candidates(segs)
                if not picks:
                    st.error("I couldn't find suitable clips. Try a video with clearer speech.")
                    st.stop()
                st.write(f"✂️ Rendering {len(picks)} Shorts...")
                outputs=[]
                outdir=td/"clips"; outdir.mkdir()
                for n,pick in enumerate(picks,1):
                    outputs.append(render(src,pick,n,outdir))
                status.update(label="Done — your clips are ready!",state="complete")

            st.header("🔥 Your ranked clips")
            zip_path=td/"ClipForge_10_Shorts.zip"
            with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
                for p in outputs: z.write(p,p.name)

            st.download_button("📦 Download all 10 clips",zip_path.read_bytes(),
                               "ClipForge_10_Shorts.zip","application/zip",
                               use_container_width=True)

            for i,(pick,out) in enumerate(zip(picks,outputs),1):
                start,end,text,sc=pick
                with st.expander(f"#{i}  •  Viral potential {sc:.0f}/100  •  {end-start:.0f}s"):
                    st.write(text)
                    st.video(out.read_bytes())
                    st.download_button("Download this Short",out.read_bytes(),out.name,
                                       "video/mp4",key=f"dl{i}")

st.caption("Viral potential is an AI/heuristic ranking, not a guaranteed prediction. Real prediction improves when the model learns from your actual views, retention, shares and comments.")
