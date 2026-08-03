import re, tempfile, zipfile, subprocess
from pathlib import Path
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="ClipForge AI", page_icon="✂️", layout="centered")

@st.cache_resource
def get_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def run(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stderr[-2500:])
    return p

HOOK={"how","why","secret","mistake","never","always","truth","actually","crazy","insane","best","worst","important","nobody","everyone","watch","listen","remember","problem","lesson","fact","real"}
EMOTION={"love","hate","fear","shocked","amazing","terrible","beautiful","fail","failed","win","won","lose","lost","embarrassing","angry","surprise","surprised","unbelievable","dangerous","wrong"}

def score(t):
    w=re.findall(r"[a-zA-Z']+",t.lower())
    if not w:return 0
    return min(100,35+sum(x in HOOK for x in w)/len(w)*170+sum(x in EMOTION for x in w)/len(w)*180+(8 if "?" in t else 0)+(5 if re.search(r"\b\d+\b",t) else 0))

def ov(a,b): return max(0,min(a[1],b[1])-max(a[0],b[0]))

def find_clips(segs):
    cs=[]
    for i,s in enumerate(segs):
        st=s[0]; text=""
        for j in range(i,len(segs)):
            en=segs[j][1]
            if en-st>60:break
            text+=(" " if text else "")+segs[j][2]
            if en-st>=20:
                sc=score(text)+(5 if re.search(r"[.!?]$",text.strip()) else 0)
                cs.append((st,en,text.strip(),min(100,sc)))
    cs.sort(key=lambda x:x[3],reverse=True); chosen=[]
    for c in cs:
        if all(ov((c[0],c[1]),(x[0],x[1]))<4 for x in chosen): chosen.append(c)
        if len(chosen)>=10:break
    return sorted(chosen,key=lambda x:x[3],reverse=True)

def download_url(url,out):
    # Node is installed by packages.txt and is explicitly supplied to yt-dlp.
    cmd=["yt-dlp","--js-runtimes","node:/usr/bin/nodejs","--no-playlist",
         "-f","bv*[height<=720]+ba/b[height<=720]","--merge-output-format","mp4",
         "-o",str(out),url]
    run(cmd)

def render(src,c,n):
    stt,en,_,_=c; out=src.parent/f"short_{n:02d}.mp4"
    run(["ffmpeg","-y","-ss",str(max(0,stt-.15)),"-i",str(src),"-t",str(en-stt+.3),
         "-vf","scale=1080:-2,crop=1080:1920","-c:v","libx264","-preset","veryfast","-crf","23",
         "-c:a","aac","-b:a","128k","-movflags","+faststart",str(out)])
    return out

st.title("✂️ ClipForge AI")
st.caption("Turn long videos into 10 ranked Shorts.")
source=st.radio("Choose your source",["📁 Upload a video","🔗 Paste a video / YouTube URL"],horizontal=True)
uploaded=None; url=None
if source.startswith("📁"):
    uploaded=st.file_uploader("Upload your long-form video",type=["mp4","mov","m4v","webm","mkv"])
else:
    url=st.text_input("Paste a YouTube or supported public video URL",placeholder="https://www.youtube.com/watch?v=...")
    st.caption("Only process videos you own or have permission to use.")

if uploaded or url:
    if st.button("🚀 Generate 10 clips",type="primary",use_container_width=True):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); src=td/"source.mp4"
            try:
                with st.status("ClipForge is working...",expanded=True) as status:
                    if uploaded: src.write_bytes(uploaded.getbuffer())
                    else:
                        st.write("⬇️ Downloading source video...")
                        download_url(url,src)
                    st.write("🎙️ Transcribing...")
                    model=get_model()
                    segments,_=model.transcribe(str(src),vad_filter=True)
                    segs=[(float(s.start),float(s.end),s.text.strip()) for s in segments if s.text.strip()]
                    st.write("🧠 Finding strongest moments...")
                    picks=find_clips(segs)
                    if not picks: raise RuntimeError("No suitable spoken clips were found.")
                    st.write(f"✂️ Rendering {len(picks)} clips...")
                    outputs=[render(src,c,i) for i,c in enumerate(picks,1)]
                    status.update(label="✅ Done!",state="complete")
                st.header("🔥 Ranked clips")
                zip_path=td/"ClipForge_Shorts.zip"
                with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
                    for p in outputs:z.write(p,p.name)
                st.download_button("📦 Download all clips",zip_path.read_bytes(),"ClipForge_Shorts.zip","application/zip",use_container_width=True)
                for i,(c,out) in enumerate(zip(picks,outputs),1):
                    stt,en,text,sc=c
                    with st.expander(f"#{i} • Viral potential {sc:.0f}/100 • {en-stt:.0f}s"):
                        st.write(text); st.video(out.read_bytes())
                        st.download_button("Download this clip",out.read_bytes(),out.name,"video/mp4",key=f"dl{i}")
            except Exception as e:
                st.error("Something went wrong."); st.code(str(e))
st.divider()
st.caption("Viral potential is a heuristic ranking, not a guaranteed prediction.")
