# EdgeIOT Dissertation Defense Presentation Package

**Candidate:** Sadra Farshid (`up202111516@fe.up.pt`)  
**Degree:** Mestrado em Engenharia de Software (MESW)  
**Supervisor:** Prof. João Carlos Pascoal Faria  
**Institution:** Faculdade de Engenharia da Universidade do Porto (FEUP)  
**Date:** October 2026  

---

## 📁 Generated Files in `presentation/output/`

| File | Description |
| :--- | :--- |
| **`EdgeIOT_Dissertation_Defense.pptx`** | **Master Google Slides / PowerPoint deck** (16:9 widescreen, 20 slides, embedded high-res figures, custom card styling, full speaker notes on every slide). Ready for 1-click import into Google Drive. |
| **`EdgeIOT_Dissertation_Defense.pdf`** | Standalone vector PDF export of the complete presentation. |
| **`index.html`** | Interactive web-based presentation deck with keyboard controls, fullscreen mode (`F`), and collapsible presenter script drawer (`S`). |
| **`SPEAKER_NOTES.md`** | Complete verbatim spoken script (10–12 minutes), slide-by-slide timing marks, and Q&A defense strategies. |
| **`generate_deck.py`** | Fully reproducible Python script using `python-pptx` to rebuild the presentation deck. |
| **`previews/`** | High-resolution PNG renders (`slide-01.png` to `slide-20.png`) of each slide for instant inspection. |
| **`assets/`** | Extracted and cropped thesis figures, latency plots, architecture diagrams, and official FEUP logo. |

---

## 🚀 How to Open in Google Slides

1. Open [Google Drive](https://drive.google.com).
2. Click **New** (`+`) → **File upload**.
3. Select `presentation/output/EdgeIOT_Dissertation_Defense.pptx`.
4. Right-click the uploaded `.pptx` file in Google Drive → **Open with** → **Google Slides**.
5. All 20 slides will load with:
   - 100% exact layout fidelity (widescreen 16:9, cards, borders, colors).
   - High-resolution embedded figures and tables.
   - Pre-filled **Speaker notes** in the bottom panel for every single slide.

---

## 🖥️ How to Present Locally in Browser

To launch the web presentation deck:
```bash
python3 -m http.server 8000 --directory /home/jetson/Workspace/Projects/EdgeIOT/presentation/output
```
Then navigate to `http://localhost:8000` in Google Chrome or any browser.
- Press **`→`** or **`Space`** to advance slides.
- Press **`←`** to go back.
- Press **`S`** to toggle the verbatim speaker script.
- Press **`F`** to enter presentation fullscreen.

---

## 📊 Presentation Structure & Time Budget (10–12 min total)

- **Slide 1:** Title & Cover Slide *(0:00 – 0:45)*
- **Slide 2:** Executive Summary: Core Findings at a Glance *(0:45 – 1:45)*
- **Slide 3:** Problem Statement & Research Questions (RQ1–RQ5) *(1:45 – 2:45)*
- **Slide 4:** Methodology: The EdgeIOT Pipeline Architecture *(2:45 – 3:45)*
- **Slide 5:** Experimental Setup: Hardware & Models Evaluated *(3:45 – 4:30)*
- **Slide 6:** Inference Latency: Prefill is Negligible, Decoding Dominates (E1) *(4:30 – 5:30)*
- **Slide 7:** Edge Resource Footprint & Action Syntactic Validity (ASVR) *(5:30 – 6:15)*
- **Slide 8:** Semantic Tokenization: The Attribution Fallacy (E2) *(6:15 – 7:15)*
- **Slide 9:** Response Caching: Decision Reuse via Semantic Tokens (E5) *(7:15 – 8:15)*
- **Slide 10:** Rule Baseline vs. LLMs: Where Does an LLM Add Value? (E3 & RQ5) *(8:15 – 9:15)*
- **Slide 11:** QLoRA Fine-Tuning: Elimination of Tail Latency (E4) *(9:15 – 10:15)*
- **Slide 12:** Multi-Criteria Evaluation: Which Model Performs Best? (RQ4) *(10:15 – 11:00)*
- **Slide 13:** Proposed Solution: The 3-Tier Hybrid Architecture *(11:00 – 11:45)*
- **Slide 14:** Conclusions: Final Synthesis & Answers to Research Questions *(11:45 – 12:30)*
- **Slide 15:** Closure Slide: Summary & Discussion with Miniatures *(12:30 – 13:00)*

### Backup Slides (Reserved for Q&A Committee Defense)
- **Slide 16 (B1):** Threats to Validity & Methodological Limitations (Section 3.12 & 6.4)
- **Slide 17 (B2):** The QLoRA Quantization Chain & Adapter Merge Fragility (Table 5.12)
- **Slide 18 (B3):** Qualitative Probe Case Transcripts & Rationale Auditing (Table 5.17)
- **Slide 19 (B4):** Cross-Study Comparison on Jetson Orin Nano (Table 5.20)
- **Slide 20 (B5):** Canonical Action Taxonomy & JSON Output Schema Specification (Section 4.6)

---
*Created in full compliance with FEUP MESW Dissertation Defense guidelines.*
