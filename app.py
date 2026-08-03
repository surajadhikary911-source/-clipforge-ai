import re, tempfile, zipfile, subprocess
from pathlib import Path
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="ClipForge AI", page_icon="✂️", layout="centered")

@st.cache_resource
def get_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

HOOK={"how","why","secret","mistake","never","always","truth","actually","crazy","insane","best","worst","important","nobody","everyone","watch","listen","remember","problem","lesson","fact","real"}
EMOTION={"love","hate","fear","shocked","amazing","terrible","beautiful","fail","failed","win","won","lose","lost","embarrassing","angry","surprise","surprised","unbelievable","dangerous","wrong"}

def score(t):
    words=re.findall(r"[a-zA-Z']+",t.lower())
    if not words:return 0
    h=sum(w in HOOK for w in words)/len(words)*100
    e=sum(w in EMOTION for w in words)/len(words)*100
    direct=3 if any(x in t.lower() for x in ("you ","your ","i ","we ")) else 0
    return min(100,35+h*1.7+e*1.8+(8 if "?" in t else 0)+(5 if re.search(r"\b\d+\b",t) else 0)+direct)

def overlap(a,b):
    return max(0,min(a[1],b[1])-max(a[0],b[0]))

def find_clips(segs,count=10):
    candidates=[]
    for i,s in enumerate(segs):
        start=s[0]; text=""
        for j in range(i,len(segs)):
            end=segs[j][1]
            if end-start>60: break
            text += (" " if text else "")+segs[j][2]
            if end-start>=20:
                bonus=5 if re.search(r"[.!?]$",text.strip()) else 0
                candidates.append((start,end,text.strip(),min(100,score(text)+bonus)))
    candidates.sort(key=lambda x:x[3],reverse=True)
    chosen=[]
    for c in candidates:
        if all(overlap((c[0],c[1]),(x[0],x[1]))<4 for x in chosen):
            chosen.append(c)
        if len(chosen)>=count: break
    return sorted(chosen,key=lambda x:x[3],reverse=True)

def render(src,c,n,outdir):
    start,end,_,_=c
    out=outdir/f"short_{n:02d}.mp4"
    cmd=["ffmpeg","-y","-ss",str(max(0,start-.15)),"-i",str(src),"-t",str(end-start+.3),
         "-vf","scale=1080:-2,crop=1080:1920","-c:v","libx264","-preset","veryfast",
         "-crf","23","-c:a","aac","-b:a","128k","-movflags","+faststart",str(out)]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stderr[-2500:])
    return out

def video_id(url):
    m=re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})",url)
    return m.group(1) if m else None

def fetch_transcript(url):
    # This is deliberately transcript-only: it does NOT download YouTube video data.
    from youtube_transcript_api import YouTubeTranscriptApi
    vid=video_id(url)
    if not vid: raise ValueError("That doesn't look like a valid YouTube URL.")
    api=YouTubeTranscriptApi()
    try:
        rows=api.fetch(vid,languages=["en"])
    except Exception:
        rows=api.fetch(vid)
    return [(float(x.start),float(x.start+x.duration),x.text) for x in rows]

st.title("✂️ ClipForge AI — redesigned")
st.caption("Find the best 10 moments from a YouTube video, then automatically cut your authorized source video.")

st.info("YouTube is used only for transcript analysis. ClipForge does not download the YouTube video. Upload the video file you own/have permission to edit for the final clips.")

url=st.text_input("🔗 Step 1 — Paste YouTube URL (optional)",placeholder="https://www.youtube.com/watch?v=...")
analyzed=None

if url and st.button("🧠 Analyze YouTube transcript",use_container_width=True):
    try:
        with st.spinner("Reading the available transcript…"):
            analyzed=fetch_transcript(url)
        st.session_state["yt_segments"]=analyzed
        st.success(f"Transcript loaded: {len(analyzed)} segments.")
    except Exception as e:
        st.session_state.pop("yt_segments",None)
        st.error("A usable YouTube transcript wasn't available for this video.")
        st.caption("You can still upload the video below; ClipForge will transcribe it locally.")

if "yt_segments" in st.session_state:
    picks=find_clips(st.session_state["yt_segments"])
    st.subheader("🔥 AI-selected moments")
    for i,c in enumerate(picks,1):
        st.write(f"**#{i} — {c[3]:.0f}/100 — {c[0]:.1f}s to {c[1]:.1f}s**")
        st.caption(c[2])

st.divider()
st.subheader("📁 Step 2 — Upload the source video")
uploaded=st.file_uploader("Upload the same video (or an authorized copy)",type=["mp4","mov","m4v","webm","mkv"])

if uploaded and st.button("✂️ Create my 10 Shorts",type="primary",use_container_width=True):
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); src=td/"source.mp4"; src.write_bytes(uploaded.getbuffer()); outdir=td/"clips"; outdir.mkdir()
        with st.status("Creating your Shorts…",expanded=True) as status:
            # If transcript timestamps exist, use them. Otherwise transcribe the uploaded source.
            if "yt_segments" in st.session_state:
                segs=st.session_state["yt_segments"]
                st.write("Using the analyzed YouTube transcript timestamps.")
            else:
                st.write("Transcribing uploaded video locally…")
                segs=[]
                for s in get_model().transcribe(str(src),vad_filter=True)[0]:
                    if s.text.strip(): segs.append((float(s.start),float(s.end),s.text.strip()))
            picks=find_clips(segs)
            if not picks: raise RuntimeError("Couldn't find enough suitable spoken sections.")
            outputs=[render(src,c,i,outdir) for i,c in enumerate(picks,1)]
            status.update(label="✅ 10 Shorts ready",state="complete")

        zpath=td/"ClipForge_Shorts.zip"
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
            for p in outputs:z.write(p,p.name)
        st.download_button("📦 Download all Shorts",zpath.read_bytes(),"ClipForge_Shorts.zip","application/zip",use_container_width=True)
        for i,(c,p) in enumerate(zip(picks,outputs),1):
            st.subheader(f"#{i} — Viral potential {c[3]:.0f}/100")
            st.caption(f"{c[1]-c[0]:.0f}s")
            st.write(c[2]); st.video(p.read_bytes())
            st.download_button("Download this clip",p.read_bytes(),p.name,"video/mp4",key=f"d{i}")

st.caption("Viral potential is a heuristic ranking, not a guarantee. For videos without accessible transcripts, upload the source and ClipForge transcribes it locally.")
