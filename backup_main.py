
import os
import tempfile
from pathlib import Path

import torch
import soundfile as sf
import librosa

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from transformers import pipeline

app = FastAPI(
    title="VYRAX AI CYBER",
    version="0.3.0"
)

MAX_FILE_SIZE = 15 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg"}

emotion_model = None
model_error = None


def get_emotion_model():
    global emotion_model, model_error

    if emotion_model is None and model_error is None:
        try:
            emotion_model = pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-er",
                device=-1
            )
        except Exception as exc:
            model_error = str(exc)

    if model_error:
        raise RuntimeError(
            "AI model could not load. Check model dependencies "
            "and internet connection."
        )

    return emotion_model


@app.get("/health")
def health():
    return {
        "status": "online",
        "project": "VYRAX AI CYBER",
        "version": "0.3.0"
    }


@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    filename = file.filename or "audio.wav"
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload WAV, MP3, M4A or OGG audio."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty."
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum audio size is 15 MB."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp:
            temp.write(content)
            temp_path = temp.name

        # Load audio and convert to the model's sample rate.
        audio, sample_rate = sf.read(
            temp_path,
            dtype="float32",
            always_2d=False
        )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if len(audio) == 0:
            raise ValueError("No audio samples found.")

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000
        )

        if len(audio) < 1600:
            raise ValueError(
                "Audio is too short. Record at least 0.1 seconds."
            )

        # Run actual model inference.
        model = get_emotion_model()

        predictions = model(
            {
                "array": audio,
                "sampling_rate": 16000
            }
        )

        if not predictions:
            raise ValueError("The model returned no predictions.")

        results = [
            {
                "emotion": str(item["label"]).lower(),
                "confidence": round(
                    float(item["score"]) * 100, 2
                )
            }
            for item in predictions
        ]

        top = results[0]
        emotion = top["emotion"]

        # These are signal-based flags, not threat detection.
        urgent_signals = {"angry", "fear", "sad"}

        return {
            "success": True,
            "project": "VYRAX AI CYBER",
            "analysis": "completed",
            "predicted_emotion": emotion,
            "confidence": top["confidence"],
            "urgency_indicator": (
                "Review recommended"
                if emotion in urgent_signals
                else "No emotion-based alert"
            ),
            "all_predictions": results,
            "disclaimer": (
                "Emotion estimates are uncertain and do not "
                "prove deception, crime, guilt or danger. "
                "Human review is required."
            )
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis failed: {str(exc)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport"
            content="width=device-width, initial-scale=1">
      <title>VYRAX AI CYBER</title>
      <style>
        * { box-sizing: border-box; }
        body {
          margin: 0; padding: 24px;
          background: #080d18; color: #e7efff;
          font-family: Arial, sans-serif;
        }
        main {
          max-width: 780px; margin: 35px auto;
          padding: 28px; border: 1px solid #263852;
          border-radius: 18px; background: #101a2b;
        }
        h1 { color: #4de2ff; }
        p { line-height: 1.6; color: #b8c8e4; }
        input, button {
          width: 100%; padding: 13px; margin: 12px 0;
          border-radius: 9px; font-size: 15px;
        }
        input { background: #0b1322; color: white; }
        button {
          background: #4de2ff; border: 0;
          font-weight: bold; cursor: pointer;
        }
        pre {
          white-space: pre-wrap; overflow-wrap: anywhere;
          background: #080d18; padding: 18px;
          border-radius: 10px; color: #9fffd0;
        }
        .note { font-size: 13px; }
      </style>
    </head>
    <body>
      <main>
        <h1>VYRAX AI CYBER</h1>
        <p>Speech Emotion Analysis Dashboard</p>
        <p id="status">Backend dashboard ready.</p>

        <form id="form">
          <input id="audio" type="file"
            accept=".wav,.mp3,.m4a,.ogg,audio/*" required>
          <button id="submit" type="submit">
            Analyze Audio
          </button>
        </form>

        <h3>Analysis Results</h3>
        <pre id="result">Upload an audio file to begin.</pre>

        <p class="note">
          Research prototype only. Emotion predictions can be
          wrong and must not be treated as evidence of a crime.
        </p>
      </main>

      <script>
        const form = document.getElementById("form");
        const result = document.getElementById("result");
        const status = document.getElementById("status");
        const button = document.getElementById("submit");

        form.addEventListener("submit", async (event) => {
          event.preventDefault();

          const file = document.getElementById("audio").files[0];
          if (!file) return;

          const data = new FormData();
          data.append("file", file);

          button.disabled = true;
          status.textContent = "Analyzing audio...";
          result.textContent = "Please wait...";

          try {
            const response = await fetch("/analyze-audio", {
              method: "POST",
              body: data
            });

            const output = await response.json();

            if (!response.ok) {
              throw new Error(output.detail || "Analysis failed");
            }

            result.textContent = JSON.stringify(output, null, 2);
            status.textContent = "Analysis completed.";
          } catch (error) {
            status.textContent = "Analysis failed.";
            result.textContent = error.message;
          } finally {
            button.disabled = false;
          }
        });
      </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
