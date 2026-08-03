import re, tempfile, zipfile, subprocess
from pathlib import Path
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="ClipForge AI", page_icon="✂️", layout="centered")

@st.cache_resource
def get_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return p

HOOK={"how","why","secret","mistake","never","always","truth","actually","crazy","insane","best","worst","important","nobody","everyone","watch","listen","remember","problem","lesson","fact","real"}
EMOTION={"love","hate","fear","shocked","amazing","terrible","beautiful","fail","failed","win","won","lose","lost","embarrassing","angry","surprise","surprised","unbelievable","dangerous","wrong"}

def score(text):
    words=re.findall(r"[a-zA-Z']+",text.lower())
    if not words: return 0
    h=sum(w in HOOK for w in words)/len(words)*100
    e=sum(w in EMOTION for w in words)/len(words)*100
    return min(100,35+h*1.7+e*1.8+(8 if "?" in text else 0)+(5 if re.search(r"\b\d+\b",text) else 0))

def overlap(a,b):
    return max(0,min(a[1],b[1])-max(a[0],b[0]))

def find_clips(segs):
    candidates=[]
    for i,s in enumerate(segs):
        start=s[0]; text=""
        for j in range(i,len(segs)):
            end=segs[j][1]
            if end-start>60: break
            text += (" " if text else "")+segs[j][2]
            if end-start>=20:
                sc=score(text)+(5 if re.search(r"[.!?]$",text.strip()) else 0)
                candidates.append((start,end,text.strip(),min(100,sc)))
    candidates.sort(key=lambda x:x[3],reverse=True)
    chosen=[]
    for c in candidates:
        if all(overlap((c[0],c[1]),(x[0],x[1]))<4 for x in chosen):
            chosen.append(c)
        if len(chosen)>=10: break
    return sorted(chosen,key=lambda x:x[3],reverse=True)

def download_url(url,out):
    run(["yt-dlp","--no-playlist","-f","bv*[height<=720]+ba/b[height<=720]","--merge-output-format","mp4","-o",str(out),url])

def render(src,c,n):
    start,end,_,_=c
    out=src.parent/f"short_{n:02d}.mp4"
    run(["ffmpeg","-y","-ss",str(max(0,start-.15)),"-i",str(src),"-t",str(end-start+.3),
         "-vf","scale=1080:-2,crop=1080:1920","-c:v","libx264","-preset","veryfast","-crf","23",
         "-c:a","aac","-b:a","128k","-movflags","+faststart",str(out)])
    return out

st.title("✂️ ClipForge AI")
st.caption("Turn long videos into 10 ranked Shorts.")

source=st.radio("Choose your source",["📁 Upload a video","🔗 Paste a video / YouTube URL"],horizontal=True)

uploaded=None
url=None
if source.startswith("📁"):
    uploaded=st.file_uploader("Upload your long-form video",type=["mp4","mov","m4v","webm","mkv"])
else:
    url=st.text_input("Paste a YouTube or supported public video URL",placeholder="https://www.youtube.com/watch?v=...")
    st.caption("Only download/process videos you own or have permission to use. Some sites may block downloading.")

if uploaded or url:
    if st.button("🚀 Generate 10 clips",type="primary",use_container_width=True):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); src=td/"source.mp4"
            try:
                with st.status("ClipForge is working...",expanded=True) as status:
                    if uploaded:
                        src.write_bytes(uploaded.getbuffer())
                    else:
                        st.write("⬇️ Downloading source video...")
                        download_url(url,src)
                    st.write("🎙️ Transcribing...")
                    model=get_model()
                    segments,_=model.transcribe(str(src),vad_filter=True)
                    segs=[(float(s.start),float(s.end),s.text.strip()) for s in segments if s.text.strip()]
                    st.write("🧠 Finding the strongest moments...")
                    picks=find_clips(segs)
                    if not picks: raise RuntimeError("No suitable spoken clips were found.")
                    st.write(f"✂️ Rendering {len(picks)} clips...")
                    outputs=[render(src,c,i) for i,c in enumerate(picks,1)]
                    status.update(label="✅ Done!",state="complete")

                st.header("🔥 Ranked clips")
                zip_path=td/"ClipForge_Shorts.zip"
                with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
                    for p in outputs: z.write(p,p.name)
                st.download_button("📦 Download all clips",zip_path.read_bytes(),"ClipForge_Shorts.zip","application/zip",use_container_width=True)

                for i,(c,out) in enumerate(zip(picks,outputs),1):
                    start,end,text,sc=c
                    with st.expander(f"#{i} • Viral potential {sc:.0f}/100 • {end-start:.0f}s"):
                        st.write(text)
                        st.video(out.read_bytes())
                        st.download_button("Download this clip",out.read_bytes(),out.name,"video/mp4",key=f"dl{i}")
            except Exception as e:
                st.error("Something went wrong.")
                st.code(str(e))

st.divider()
st.caption("Viral potential is a heuristic ranking, not a guaranteed prediction. Only process videos you own or have permission to use.")
