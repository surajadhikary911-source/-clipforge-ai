import re, tempfile, zipfile, subprocess, sys
from pathlib import Path
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="ClipForge AI", page_icon="✂️", layout="centered")

@st.cache_resource
def model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def run(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-3500:])
    return p

def deno_path():
    try:
        import deno
        p=deno.find_deno_bin()
        return str(p)
    except Exception:
        return "deno"

HOOK={"how","why","secret","mistake","never","always","truth","actually","crazy","insane","best","worst","important","nobody","everyone","watch","listen","remember","problem","lesson","fact","real"}
EMOTION={"love","hate","fear","shocked","amazing","terrible","beautiful","fail","failed","win","won","lose","lost","embarrassing","angry","surprise","surprised","unbelievable","dangerous","wrong"}

def score(t):
    w=re.findall(r"[a-zA-Z']+",t.lower())
    if not w:return 0
    h=sum(x in HOOK for x in w)/len(w)*100
    e=sum(x in EMOTION for x in w)/len(w)*100
    return min(100,35+h*1.7+e*1.8+(8 if "?" in t else 0)+(5 if re.search(r"\b\d+\b",t) else 0))

def overlap(a,b): return max(0,min(a[1],b[1])-max(a[0],b[0]))

def find_clips(segs):
    c=[]
    for i,s in enumerate(segs):
        start=s[0]; text=""
        for j in range(i,len(segs)):
            end=segs[j][1]
            if end-start>60: break
            text+=(" " if text else "")+segs[j][2]
            if end-start>=20:
                c.append((start,end,text.strip(),min(100,score(text)+(5 if re.search(r"[.!?]$",text.strip()) else 0))))
    c.sort(key=lambda x:x[3],reverse=True); chosen=[]
    for x in c:
        if all(overlap((x[0],x[1]),(y[0],y[1]))<4 for y in chosen): chosen.append(x)
        if len(chosen)>=10: break
    return sorted(chosen,key=lambda x:x[3],reverse=True)

def download_youtube(url,out):
    d=deno_path()
    base=["yt-dlp","--js-runtimes",f"deno:{d}","--remote-components","ejs:github",
          "--no-playlist","--no-warnings"]
    # Prefer web_safari HLS because current YouTube PO-token enforcement can make
    # ordinary GVS format URLs return 403. Fall back to default extraction.
    attempts=[
        base+["--extractor-args","youtube:player_client=web_safari",
              "-f","best[protocol^=m3u8]/best","--hls-prefer-native",
              "--merge-output-format","mp4","-o",str(out),url],
        base+["-f","bv*[height<=720]+ba/b[height<=720]",
              "--merge-output-format","mp4","-o",str(out),url],
    ]
    errors=[]
    for cmd in attempts:
        try:
            run(cmd)
            if out.exists() and out.stat().st_size>10000: return
        except Exception as e: errors.append(str(e))
    raise RuntimeError(
        "YouTube rejected the server download after two current yt-dlp methods. "
        "This is a YouTube-side access/PO-token restriction, not a ClipForge processing error.\n\n"
        + "\n---\n".join(errors[-2:])
    )

def download_direct(url,out):
    run(["yt-dlp","--no-playlist","-f","bv*[height<=720]+ba/b[height<=720]",
         "--merge-output-format","mp4","-o",str(out),url])

def render(src,c,n):
    stt,en,_,_=c; out=src.parent/f"short_{n:02d}.mp4"
    run(["ffmpeg","-y","-ss",str(max(0,stt-.15)),"-i",str(src),"-t",str(en-stt+.3),
         "-vf","scale=1080:-2,crop=1080:1920","-c:v","libx264","-preset","veryfast",
         "-crf","23","-c:a","aac","-b:a","128k","-movflags","+faststart",str(out)])
    return out

st.title("✂️ ClipForge AI")
st.caption("Long video → 10 ranked Shorts")

source=st.radio("Video source",["📁 Upload video","🔗 YouTube / public URL"],horizontal=True)
uploaded=None; url=None
if source.startswith("📁"):
    uploaded=st.file_uploader("Upload your video",type=["mp4","mov","m4v","webm","mkv"])
else:
    url=st.text_input("Paste YouTube URL",placeholder="https://www.youtube.com/watch?v=...")
    st.caption("Use only videos you own or have permission to process.")

if uploaded or url:
    if st.button("🚀 Generate 10 clips",type="primary",use_container_width=True):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); src=td/"source.mp4"
            try:
                with st.status("ClipForge AI is working...",expanded=True) as status:
                    if uploaded:
                        src.write_bytes(uploaded.getbuffer())
                    elif "youtube.com" in url or "youtu.be" in url:
                        st.write("⬇️ Getting YouTube video...")
                        download_youtube(url,src)
                    else:
                        st.write("⬇️ Getting public video...")
                        download_direct(url,src)
                    st.write("🎙️ Transcribing...")
                    segs=[]
                    for s in model().transcribe(str(src),vad_filter=True)[0]:
                        if s.text.strip(): segs.append((float(s.start),float(s.end),s.text.strip()))
                    st.write("🧠 Finding high-potential moments...")
                    picks=find_clips(segs)
                    if not picks: raise RuntimeError("No suitable spoken sections found.")
                    st.write(f"✂️ Rendering {len(picks)} clips...")
                    outputs=[render(src,c,i) for i,c in enumerate(picks,1)]
                    status.update(label="✅ Finished",state="complete")

                st.header("🔥 Ranked clips")
                zpath=td/"ClipForge_Shorts.zip"
                with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
                    for p in outputs:z.write(p,p.name)
                st.download_button("📦 Download all 10",zpath.read_bytes(),"ClipForge_Shorts.zip","application/zip",use_container_width=True)
                for i,(c,p) in enumerate(zip(picks,outputs),1):
                    st.subheader(f"#{i} — {c[3]:.0f}/100")
                    st.caption(f"{c[1]-c[0]:.0f}s")
                    st.write(c[2]); st.video(p.read_bytes())
                    st.download_button("Download clip",p.read_bytes(),p.name,"video/mp4",key=f"d{i}")
            except Exception as e:
                st.error("Could not process this URL/video.")
                st.code(str(e))

st.divider()
st.caption("Viral score is a ranking heuristic, not a guaranteed prediction.")
