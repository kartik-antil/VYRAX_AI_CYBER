import io
from pathlib import Path

import librosa
from transformers import pipeline

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Vyrax AI Cyber",
    description="Real-Time Speech Emotion Analysis",
    version="0.2.0",
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg"}
MAX_SIZE = 25 * 1024 * 1024

EMOTION_MODEL = "Dpngtm/wav2vec2-emotion-recognition"

emotion_classifier = None


def get_emotion_classifier():
    global emotion_classifier

    if emotion_classifier is None:
        emotion_classifier = pipeline(
            "audio-classification", model=EMOTION_MODEL, device=-1
        )

    return emotion_classifier


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vyrax AI Cyber</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #080b16;
    color: #edf2ff;
}
.layout { display: flex; min-height: 100vh; }
.sidebar {
    width: 235px; padding: 28px 18px;
    background: #0d1222; border-right: 1px solid #242c45;
}
.logo { font-size: 23px; font-weight: bold; color: #72dcff; }
.logo span { color: #a78bfa; }
.tagline { color: #8c98b5; font-size: 12px; margin: 8px 0 35px; }
.nav {
    padding: 13px; margin-bottom: 9px; border-radius: 9px;
    background: #18213a; color: #a9c9ff;
}
.nav.muted { background: transparent; color: #8c98b5; }
.main { flex: 1; padding: 35px; max-width: 1200px; }
.top {
    display: flex; justify-content: space-between;
    align-items: center; gap: 15px; flex-wrap: wrap;
}
h1 { margin: 0; font-size: 29px; }
.subtitle { color: #9aa6c2; margin-top: 9px; line-height: 1.5; }
.status {
    border: 1px solid #245e55; background: #102b2a;
    color: #77efc4; border-radius: 30px; padding: 10px 14px;
    font-size: 12px;
}
.cards {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 16px; margin: 30px 0;
}
.card, .panel {
    background: #101629; border: 1px solid #252e49;
    border-radius: 15px; padding: 22px;
}
.card small { color: #8e9bb8; }
.card h2 { font-size: 20px; margin: 13px 0 5px; }
.card p { color: #8e9bb8; font-size: 12px; margin-bottom: 0; }
.panel { margin-bottom: 20px; }
.panel h3 { margin: 0 0 8px; font-size: 18px; }
.desc { color: #95a1bd; font-size: 13px; line-height: 1.6; }
.drop {
    border: 1px dashed #53658f; border-radius: 13px;
    padding: 30px 15px; margin-top: 20px; text-align: center;
    background: #0c1224;
}
.upload-icon { font-size: 36px; margin-bottom: 12px; }
input[type=file] { display: block; margin: 20px auto; max-width: 100%; }
button {
    border: 0; border-radius: 9px; padding: 13px 20px;
    font-weight: bold; cursor: pointer;
    background: linear-gradient(90deg, #4b8dff, #9874ff);
    color: white;
}
button:disabled { opacity: .55; cursor: wait; }
#message {
    margin-top: 18px; color: #b7c6e8;
    font-size: 14px; line-height: 1.6; overflow-wrap: anywhere;
}
.footer { color: #71809e; font-size: 12px; margin-top: 25px; }
@media(max-width: 760px) {
    .sidebar { width: 100%; border-right: 0; border-bottom: 1px solid #242c45; }
    .layout { display: block; }
    .sidebar { padding: 18px; }
    .tagline { margin-bottom: 16px; }
    .nav { display: inline-block; margin-bottom: 0; }
    .main { padding: 22px 16px; }
    .cards { grid-template-columns: 1fr; }
    h1 { font-size: 24px; }
}
</style>
</head>
<body>
<div class="layout">
    <aside class="sidebar">
        <div class="logo">VYRAX <span>AI</span></div>
        <div class="tagline">CYBER INTELLIGENCE PLATFORM</div>
        <div class="nav">◈ &nbsp; Dashboard</div>
        <div class="nav muted">◉ &nbsp; Speech Analysis</div>
        <div class="nav muted">▤ &nbsp; Reports</div>
        <div class="nav muted">⚙ &nbsp; Settings</div>
    </aside>

    <main class="main">
        <div class="top">
            <div>
                <h1>Speech Analysis Dashboard</h1>
                <div class="subtitle">
                    AI-assisted speech review for cybercrime investigations.
                </div>
            </div>
            <div class="status">● BACKEND ONLINE</div>
        </div>

        <section class="cards">
            <div class="card">
                <small>PLATFORM STATUS</small>
                <h2>Operational</h2>
                <p>Backend health check</p>
            </div>
            <div class="card">
                <small>SUPPORTED FORMATS</small>
                <h2>4 Formats</h2>
                <p>WAV · MP3 · M4A · OGG</p>
            </div>
            <div class="card">
                <small>MAX FILE SIZE</small>
                <h2>25 MB</h2>
                <p>Per audio upload</p>
            </div>
        </section>

        <section class="panel">
            <h3>◉ Analyze Audio</h3>
            <div class="desc">
                Upload an audio recording to begin the analysis workflow.
                Use only recordings you are authorized to process.
            </div>

            
            <form id="uploadForm">
                <div class="drop">
                    <div class="upload-icon">🎙️</div>
                    <strong>Select an audio recording</strong>
                    <div class="desc">WAV, MP3, M4A or OGG · Maximum 25 MB</div>
                    <input
                        type="file" id="audio"
                        accept=".wav,.mp3,.m4a,.ogg,audio/*" required>
                    <button id="analyzeBtn" type="submit">Upload Audio</button>
                </div>
            </form>
            <div id="message" aria-live="polite">
                Waiting for an audio file.
            </div>
        </section>

        
    <section class="panel">
    <h3>AI Analysis Results</h3>
    <div class="desc" id="results" style="white-space: pre-line;">
        No analysis available yet. Upload an audio file to begin.
    </div>
</section>


        <div class="footer">
            Vyrax AI Cyber · Prototype v0.2 · AI signals are not proof of guilt.
        </div>
    </main>
</div>

<script>
const form = document.getElementById("uploadForm");
const audioInput = document.getElementById("audio");
const message = document.getElementById("message");
const results = document.getElementById("results");
const button = document.getElementById("analyzeBtn");

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    

    const file = audioInput.files[0];

    if (!file) {
        message.textContent = "Please select an audio file first.";
        return;
    }

    button.disabled = true;
    message.textContent = "Uploading audio...";
    
    results.textContent = "Waiting for the upload response.";
    
    const data = new FormData();
    data.append("audio", file);

    try {
        const response = await fetch("/analyze-audio", {
            method: "POST",
            body: data
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Upload failed.");
        }

        message.textContent =
            "Upload successful: " + result.filename;

        
if (result.analysis) {
    const predictions = result.analysis.all_predictions;
    const topEmotion = result.analysis.emotion;
    const confidence = result.analysis.confidence;

    results.style.whiteSpace = "normal";

    results.innerHTML = `
        <div style="padding:16px; background:#18213a; border-radius:12px; margin-top:12px;">
            <div style="color:#8c98b5; font-size:12px;">DETECTED EMOTION</div>
            <h2 style="color:#72dcff; margin:8px 0;">${topEmotion.toUpperCase()}</h2>
            <div style="font-size:14px;">Confidence: ${confidence}%</div>
        </div>

        <h4 style="margin:22px 0 14px;">All Emotion Predictions</h4>

        ${predictions.map(item => `
            <div style="margin:14px 0;">
                <div style="display:flex; justify-content:space-between; gap:12px; margin-bottom:6px;">
                    <span>${item.emotion}</span>
                    <strong>${item.confidence}%</strong>
                </div>
                <div style="height:8px; background:#252e49; border-radius:8px; overflow:hidden;">
                    <div style="height:100%; width:${Math.max(0, Math.min(100, Number(item.confidence)))}%; background:linear-gradient(90deg,#4b8dff,#9874ff); border-radius:8px;"></div>
                </div>
            </div>
        `).join("")}
    `;
} else {
    results.textContent =
        result.message || "No analysis result received.";
}


 

    } catch (error) {
        message.textContent = "Error: " + error.message;
    } finally {
        button.disabled = false;
    }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze-audio")
async def analyze_audio(audio: UploadFile = File(...)):
    extension = Path(audio.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail="Upload a WAV, MP3, M4A or OGG audio file."
        )

    content = await audio.read()

    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Maximum file size is 25 MB.")

    try:
        audio_buffer = io.BytesIO(content)

        waveform, sample_rate = librosa.load(audio_buffer, sr=16000, mono=True)

        if len(waveform) == 0:
            raise ValueError("No audio samples found.")

        classifier = get_emotion_classifier()

        predictions = classifier({"raw": waveform, "sampling_rate": sample_rate})

        best_prediction = max(predictions, key=lambda item: item["score"])

        return {
            "filename": Path(audio.filename or "audio").name,
            "status": "analyzed",
            "message": "AI emotion analysis completed.",
            "analysis": {
                "emotion": best_prediction["label"],
                "confidence": round(best_prediction["score"] * 100, 2),
                "all_predictions": [
                    {
                        "emotion": item["label"],
                        "confidence": round(item["score"] * 100, 2),
                    }
                    for item in predictions
                ],
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Audio analysis failed: {str(exc)}"
        )
