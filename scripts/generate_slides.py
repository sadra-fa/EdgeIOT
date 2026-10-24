#!/usr/bin/env python3
"""
Generate streamlined 10-minute presentation slides for Google Slides / PowerPoint (.pptx).
Strictly adheres to Prof. João Pascoal Faria's lecture guidelines:
- Time budget: 10 min presentation + 10 min Q&A (~1 slide/min, 11 content slides + closure + 4 backup)
- Cover slide with date, email, FEUP context
- Executive summary upfront
- KISS: Bold keywords, large font, remove sentence clutter, no reading aloud
- Large figures with clear scales
- Data-driven ("In God we trust, all others bring data")
- Closure slide with miniatures of key results for Q&A discussion
- Comprehensive backup slides for deep technical questions
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# 16:9 Widescreen dimensions
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

# Color Palette: Professional Academic & Modern Tech
COLOR_BG = RGBColor(248, 249, 250)         # Clean soft off-white
COLOR_CARD_BG = RGBColor(255, 255, 255)    # Pure white cards
COLOR_TEXT_MAIN = RGBColor(26, 37, 48)     # Deep navy/charcoal for max legibility
COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # Professional slate gray
COLOR_PRIMARY = RGBColor(14, 116, 144)     # Deep Teal / Cyan-700
COLOR_ACCENT = RGBColor(185, 28, 28)       # U.Porto Crimson red
COLOR_BORDER = RGBColor(226, 232, 240)     # Subtle card border
COLOR_HIGHLIGHT = RGBColor(3, 105, 161)    # Blue highlight

def create_presentation():
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
        bg_shape.fill.solid()
        bg_shape.fill.fore_color.rgb = COLOR_BG
        bg_shape.line.fill.background()
        return bg_shape

    def add_header(slide, title_text, category_text="MESW DISSERTATION PRESENTATION", category=None):
        if category is not None:
            category_text = category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_ACCENT

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(28)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_TEXT_MAIN

        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_BORDER
        line.line.fill.background()

    def add_card(slide, left, top, width, height, title=None, bg_color=COLOR_CARD_BG, border_color=COLOR_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.5)
        else:
            card.line.fill.background()

        if title:
            tbox = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), Inches(0.5))
            tf = tbox.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(18)
            p.font.bold = True
            p.font.color.rgb = COLOR_PRIMARY
        return card

    # =========================================================================
    # SLIDE 1: COVER SLIDE (With Date, Email, Supervisor, FEUP context)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, Inches(0.15))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_ACCENT
    top_bar.line.fill.background()

    logo_path = "dissertation/latex/figures/uporto-feup.png"
    if os.path.exists(logo_path):
        s1.shapes.add_picture(logo_path, Inches(0.9), Inches(0.8), width=Inches(3.2))

    tbox = s1.shapes.add_textbox(Inches(0.9), Inches(2.2), Inches(11.5), Inches(2.5))
    tf = tbox.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "Edge LLM Optimization"
    p0.font.size = Pt(44)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_TEXT_MAIN

    p1 = tf.add_paragraph()
    p1.text = "Deployment, Compression & Evaluation of Small LLMs for Smart-Home Edge Intelligence"
    p1.font.size = Pt(20)
    p1.font.color.rgb = COLOR_PRIMARY
    p1.space_before = Pt(10)

    # Info card with date included
    card_info = add_card(s1, Inches(0.9), Inches(4.7), Inches(11.5), Inches(2.0))
    ibox = s1.shapes.add_textbox(Inches(1.2), Inches(4.85), Inches(10.9), Inches(1.7))
    itf = ibox.text_frame
    itf.word_wrap = True

    p_auth = itf.paragraphs[0]
    p_auth.text = "Author: Sadra Farshid (up202111516@fe.up.pt)"
    p_auth.font.size = Pt(17)
    p_auth.font.bold = True
    p_auth.font.color.rgb = COLOR_TEXT_MAIN

    p_sup = itf.add_paragraph()
    p_sup.text = "Supervisor: Prof. João Carlos Pascoal Faria"
    p_sup.font.size = Pt(16)
    p_sup.font.color.rgb = COLOR_TEXT_MAIN
    p_sup.space_before = Pt(5)

    p_prog = itf.add_paragraph()
    p_prog.text = "Faculdade de Engenharia da Universidade do Porto (FEUP) | MESW — Master in Software Engineering"
    p_prog.font.size = Pt(15)
    p_prog.font.color.rgb = COLOR_TEXT_MUTED
    p_prog.space_before = Pt(5)

    p_date = itf.add_paragraph()
    p_date.text = "Date: October 2026"
    p_date.font.size = Pt(14)
    p_date.font.bold = True
    p_date.font.color.rgb = COLOR_ACCENT
    p_date.space_before = Pt(5)

    # =========================================================================
    # SLIDE 2: EXECUTIVE SUMMARY (Professor's Key Finding Focus)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Executive Summary: Core Findings at a Glance")

    exec_cards = [
        ("The Challenge", [
            ("Cloud IoT Bottlenecks:", "Network latency, privacy leak, outage risk, token fees."),
            ("Edge Hardware Bounds:", "Jetson Orin Nano (8 GB shared DRAM, 102 GB/s bandwidth)."),
            ("Central Question:", "How do 1.5B–3.8B LLMs perform, and when do they beat rules?")
        ]),
        ("The EdgeIOT Approach", [
            ("Integrated Pipeline:", "Preprocessing -> Clustering -> XGBoost -> IsoForest -> LLM."),
            ("4-bit Quantization:", "GGUF serving via Ollama; local QLoRA fine-tuning."),
            ("Semantic Tokens:", "Collapse sensor noise into discrete checkable domain tokens.")
        ]),
        ("Decisive Findings", [
            ("Decoding Dominates:", "Prefill takes ~40 ms; decoding takes 1.3–5.7 s (bandwidth bound)."),
            ("Response Caching:", "Reuses 98.2% of decisions -> drops latency to 56 ms!"),
            ("QLoRA Tuning Impact:", "Boosts strict JSON to 84–95%; eliminates tail latency."),
            ("Rules vs. LLMs:", "Rules win on closed set; LLMs essential for unseen/text (75%).")
        ])
    ]

    for i, (ctitle, cbullets) in enumerate(exec_cards):
        cx = Inches(0.8 + i * 4.0)
        cw = Inches(3.73)
        add_card(s2, cx, Inches(1.8), cw, Inches(5.1), title=ctitle)
        tbox = s2.shapes.add_textbox(cx + Inches(0.2), Inches(2.6), cw - Inches(0.4), Inches(4.1))
        tf = tbox.text_frame
        tf.word_wrap = True
        for j, (bhead, bbody) in enumerate(cbullets):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.text = f"• {bhead} "
            p.font.bold = True
            p.font.size = Pt(14)
            p.font.color.rgb = COLOR_TEXT_MAIN
            if j > 0:
                p.space_before = Pt(10)
            run = p.add_run()
            run.text = bbody
            run.font.bold = False
            run.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 3: PROBLEM STATEMENT & RESEARCH QUESTIONS (CONSOLIDATED)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Problem Statement & Research Questions")

    # Left: The Conflict
    add_card(s3, Inches(0.8), Inches(1.8), Inches(5.4), Inches(5.1), title="The Edge AI Dilemma")
    tbox = s3.shapes.add_textbox(Inches(1.05), Inches(2.6), Inches(4.9), Inches(4.1))
    tf = tbox.text_frame
    tf.word_wrap = True
    dilemma_pts = [
        ("Cloud Dependency:", "High latency, privacy risks, breaks during network outages."),
        ("Edge Constraints:", "8 GB UMA memory shared with OS, memory bandwidth ceiling."),
        ("The Rule Paradox:", "A closed vocabulary can be mapped by 12 rules in 0.5 µs."),
        ("Core Engineering Question:", "Can small LLMs be optimized on-device, and where do they genuinely add value over deterministic rules?")
    ]
    for j, (k, v) in enumerate(dilemma_pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_ACCENT if "Question" in k else COLOR_PRIMARY
        if j > 0:
            p.space_before = Pt(12)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MAIN

    # Right: Targeted RQs (Streamlined)
    add_card(s3, Inches(6.5), Inches(1.8), Inches(6.033), Inches(5.1), title="Research Focus (RQ1 - RQ5)")
    tbox2 = s3.shapes.add_textbox(Inches(6.75), Inches(2.6), Inches(5.5), Inches(4.1))
    tf2 = tbox2.text_frame
    tf2.word_wrap = True
    rq_pts = [
        ("RQ1 (Edge Performance):", "Latency (prefill vs. decode), throughput, memory, and validity of 4-bit small LLMs (1.5B–3.8B)."),
        ("RQ2 (QLoRA Tuning):", "Effect of task-tuning on JSON schema validity, answer length, and adapter merge integrity."),
        ("RQ3 (Semantic Tokens):", "True impact on prompt tokens, prefill time, decode time, and decision reuse."),
        ("RQ4 (Best Trade-off):", "Best model configuration balancing speed, resources, and validity."),
        ("RQ5 (Rules vs. LLM):", "Specific situations where an LLM beats deterministic if-else rules.")
    ]
    for j, (k, v) in enumerate(rq_pts):
        p = tf2.paragraphs[0] if j == 0 else tf2.add_paragraph()
        p.text = f"{k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_PRIMARY
        if j > 0:
            p.space_before = Pt(10)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MAIN

    # =========================================================================
    # SLIDE 4: SYSTEM ARCHITECTURE (FIGURE 4.1)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Methodology: The EdgeIOT Pipeline Architecture")

    diagram_path = "dissertation/slides_assets/pipeline_flow_perfect.png"
    if os.path.exists(diagram_path):
        s4.shapes.add_picture(diagram_path, Inches(0.8), Inches(1.8), width=Inches(6.8))

    add_card(s4, Inches(7.9), Inches(1.8), Inches(4.6), Inches(5.1), title="Pipeline Flow")
    tbox = s4.shapes.add_textbox(Inches(8.15), Inches(2.6), Inches(4.1), Inches(4.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    pipe_stages = [
        ("1. Ingestion:", "5 ambient readings (temp, humidity, noise, light, motion)."),
        ("2. Clustering:", "Ward (k=3) establishes states: sedentary, active, sleeping."),
        ("3. Classifier:", "XGBoost infers behavioral state in microseconds."),
        ("4. Anomaly Filter:", "Isolation Forest (window w=5) flags multivariate outliers (5%)."),
        ("5. Semantic Tokens:", "Maps continuous z-scores to domain symbols (+/-1 sd)."),
        ("6. Decision Step:", "Ollama LLM emits 1 of 7 actions + JSON rationale.")
    ]
    for j, (k, v) in enumerate(pipe_stages):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_PRIMARY
        if j > 0:
            p.space_before = Pt(9)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MAIN

    # =========================================================================
    # SLIDE 5: EXPERIMENTAL SETUP & HARDWARE
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Experimental Setup: Hardware & Models Evaluated")

    add_card(s5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1), title="Jetson Orin Nano Testbed")
    tbox = s5.shapes.add_textbox(Inches(1.1), Inches(2.6), Inches(5.0), Inches(4.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    bench_specs = [
        ("Hardware:", "NVIDIA Jetson Orin Nano Developer Kit (8 GB LPDDR5)."),
        ("Bandwidth & Power:", "102 GB/s in MAXN_SUPER power mode (7W–25W)."),
        ("UMA Shared Memory:", "Single DRAM pool for OS, desktop, CUDA buffers, weights, and KV cache."),
        ("Serving Engine:", "Ollama 0.35.1 (llama.cpp backend)."),
        ("Instrumentation:", "tegrastats logging GPU, RAM, power, and temps @ 100 ms intervals.")
    ]
    for j, (k, v) in enumerate(bench_specs):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_MAIN
        if j > 0:
            p.space_before = Pt(10)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MUTED

    add_card(s5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1), title="Candidate Models (4-bit GGUF)")
    tbox2 = s5.shapes.add_textbox(Inches(7.1), Inches(2.6), Inches(5.1), Inches(4.0))
    tf2 = tbox2.text_frame
    tf2.word_wrap = True
    models_tested = [
        ("LLaMA 3.2 (3.2B) — Meta:", "Reference model. Grouped-query attention, optimized for edge silicon. Format: Q4_K_M (2.0 GB)."),
        ("Phi-3.5-mini (3.8B) — Microsoft:", "High parameter density, 32 layers, synthetic training. Format: Q4_0 (2.2 GB)."),
        ("Qwen 2.5 (1.5B) — Alibaba:", "Ultra-compact instruction model for memory-critical nodes. Format: Q4_K_M (0.99 GB)."),
        ("Protocol Rigor:", "100 test rows (seed 42), isolated runs, warm-up excluded.")
    ]
    for j, (k, v) in enumerate(models_tested):
        p = tf2.paragraphs[0] if j == 0 else tf2.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_PRIMARY
        if j > 0:
            p.space_before = Pt(10)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MAIN

    # =========================================================================
    # SLIDE 6: MULTI-MODEL INFERENCE PERFORMANCE (E1)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Inference Latency: Prefill is Negligible, Decoding Dominates (E1)")

    plot_path = "dissertation/slides_assets/multi_latency-1.png"
    if os.path.exists(plot_path):
        s6.shapes.add_picture(plot_path, Inches(0.8), Inches(1.8), width=Inches(6.8))

    add_card(s6, Inches(7.9), Inches(1.8), Inches(4.6), Inches(5.1), title="Quantitative Breakdown")
    tbox = s6.shapes.add_textbox(Inches(8.15), Inches(2.55), Inches(4.1), Inches(4.1))
    tf = tbox.text_frame
    tf.word_wrap = True
    e1_pts = [
        ("Prefill (~40 ms):", "Processing 134–144 tokens takes 32–49 ms (>2,800 to 4,200 tok/s). Compute-bound."),
        ("Decode (1.3–5.5 s):", "Autoregressive generation reads full weights every token. Memory bandwidth bound."),
        ("Qwen 2.5 (1.5B):", "1,384 ms median | 36.0 tok/s | 53.7 tokens generated."),
        ("LLaMA 3.2 (3B):", "2,080 ms median | 22.3 tok/s | 67.0 tokens generated."),
        ("Phi-3.5 (3.8B):", "5,720 ms median | 23.0 tok/s | 120.3 tokens generated (87% hit 128-token cap)."),
        ("Key Finding:", "Latency is governed by answer length, NOT prompt length!")
    ]
    for j, (k, v) in enumerate(e1_pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_ACCENT if "Key Finding" in k else COLOR_TEXT_MAIN
        if j > 0:
            p.space_before = Pt(6)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 7: RESOURCE FOOTPRINT & VALIDITY (E1)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Edge Resources & Action Syntactic Validity (ASVR)")

    add_card(s7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1), title="Resource Utilization (tegrastats)")
    tbox = s7.shapes.add_textbox(Inches(1.1), Inches(2.6), Inches(5.0), Inches(4.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    res_pts = [
        ("Memory Footprint:", "Weights + KV cache take 1.0 GB (Qwen) to 2.3 GB (Phi). System RAM remains <6.9 GB."),
        ("GPU Saturation:", "GPU active 83%–91% during decode (idle: 3%), fully saturating memory bus."),
        ("Power & Energy:", "Active power 17.9W–20.6W. Energy/decision: Qwen 28 J, LLaMA 65 J, Phi 108 J."),
        ("Thermal Stability:", "Max junction temp <67 °C throughout. Zero thermal throttling observed.")
    ]
    for j, (k, v) in enumerate(res_pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_PRIMARY
        if j > 0:
            p.space_before = Pt(10)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MAIN

    add_card(s7, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1), title="Base Model Validity Challenges")
    tbox2 = s7.shapes.add_textbox(Inches(7.1), Inches(2.6), Inches(5.1), Inches(4.0))
    tf2 = tbox2.text_frame
    tf2.word_wrap = True
    val_pts = [
        ("Strict JSON Rate (ASVR):", "LLaMA achieved 65%. Qwen & Phi had 0% ASVR due to wrapping in markdown code fences (```json)."),
        ("Recoverable Actions:", "97% for Qwen, 91% for Phi, 80% for LLaMA if lenient markdown parsing is applied."),
        ("Active State Hallucination:", "Base LLaMA emitted Python code (SmartHomeController) in 32% of [STATE_ACTIVE] cases."),
        ("Invented Actions:", "Phi invented 'dehumidifier_off' (9%); Qwen invented 'decrease_temperature' (3%).")
    ]
    for j, (k, v) in enumerate(val_pts):
        p = tf2.paragraphs[0] if j == 0 else tf2.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_ACCENT
        if j > 0:
            p.space_before = Pt(10)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MAIN

    # =========================================================================
    # SLIDE 8: SEMANTIC TOKENIZATION & ATTRIBUTION FALLACY (E2)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Semantic Tokenization: The Attribution Fallacy (E2)")

    plot_path = "dissertation/slides_assets/tokenization_latency-1.png"
    if os.path.exists(plot_path):
        s8.shapes.add_picture(plot_path, Inches(0.8), Inches(1.8), width=Inches(6.8))

    add_card(s8, Inches(7.9), Inches(1.8), Inches(4.6), Inches(5.1), title="Attribution Insights")
    tbox = s8.shapes.add_textbox(Inches(8.15), Inches(2.55), Inches(4.1), Inches(4.1))
    tf = tbox.text_frame
    tf.word_wrap = True
    e2_pts = [
        ("The Token Fallacy:", "Characters shrink 79.7%, but full prompt only shrinks 14–19% (instructions/templates unchanged)."),
        ("Negligible Prefill Gain:", "Saves only 15–116 ms of prefill (at most 2% of total request latency)."),
        ("Divergent Decoding:", [
            "Qwen: 28.9% shorter -> latency drops 40.4% (2.43s -> 1.45s).",
            "Phi: Hits max token limit in both -> latency unchanged.",
            "LLaMA: Explains symbols first -> latency increases 7.5%!"
        ]),
        ("Verdict:", "Semantic tokenization is NOT a reliable single-request accelerator. Its true power is caching!")
    ]
    for j, item in enumerate(e2_pts):
        k, v = item
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_ACCENT if "Verdict" in k else COLOR_TEXT_MAIN
        if j > 0:
            p.space_before = Pt(6)
        if isinstance(v, list):
            for sub in v:
                p_sub = tf.add_paragraph()
                p_sub.text = f"   - {sub}"
                p_sub.font.size = Pt(12)
                p_sub.font.color.rgb = COLOR_TEXT_MUTED
                p_sub.space_before = Pt(2)
        else:
            run = p.add_run()
            run.text = v
            run.font.bold = False
            run.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 9: RESPONSE CACHING: DECISION REUSE (E5)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Response Caching: Decision Reuse via Semantic Tokens (E5)")

    metrics = [
        ("98.2%", "Test Cache Hit Rate", "5,893 of 6,000 decisions served instantly without LLM"),
        ("56 ms", "Mean Decision Time", "56x speedup over base LLM inference (~3,000 ms -> 56 ms)"),
        ("0.43 µs", "Hash Lookup Latency", "Microsecond memory lookup vs. seconds of GPU compute")
    ]
    for i, (val, label, desc) in enumerate(metrics):
        cx = Inches(0.8 + i * 4.0)
        cw = Inches(3.73)
        add_card(s9, cx, Inches(1.8), cw, Inches(1.8), border_color=COLOR_PRIMARY)
        tbox = s9.shapes.add_textbox(cx, Inches(1.9), cw, Inches(1.6))
        tf = tbox.text_frame
        tf.word_wrap = True
        
        p_val = tf.paragraphs[0]
        p_val.text = val
        p_val.alignment = PP_ALIGN.CENTER
        p_val.font.size = Pt(36)
        p_val.font.bold = True
        p_val.font.color.rgb = COLOR_PRIMARY

        p_lbl = tf.add_paragraph()
        p_lbl.text = label
        p_lbl.alignment = PP_ALIGN.CENTER
        p_lbl.font.size = Pt(14)
        p_lbl.font.bold = True
        p_lbl.font.color.rgb = COLOR_TEXT_MAIN

        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.alignment = PP_ALIGN.CENTER
        p_desc.font.size = Pt(11)
        p_desc.font.color.rgb = COLOR_TEXT_MUTED

    add_card(s9, Inches(0.8), Inches(3.8), Inches(11.733), Inches(3.1), title="Why Caching Succeeds")
    tbox2 = s9.shapes.add_textbox(Inches(1.1), Inches(4.5), Inches(11.1), Inches(2.2))
    tf2 = tbox2.text_frame
    tf2.word_wrap = True

    cache_pts = [
        ("Finite Grammar Collapse:", "Raw sensor floats never repeat (0.0% hits). Semantic grammar maps stream into only 107 distinct observed states."),
        ("Rapid Cache Saturation:", "67% hit rate after 100 rows; 89% after 500 rows; 98.2% across full 6,000-row test stream."),
        ("One-Time Validation at Storage:", "LLM errors are validated against rules once when cached. Corrected 75 discordant entries, preventing error propagation."),
        ("Execution Profile:", "Pays 1–2s GPU compute latency only on the very first miss; subsequent queries execute at bus memory speed.")
    ]
    for j, (k, v) in enumerate(cache_pts):
        p = tf2.paragraphs[0] if j == 0 else tf2.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_MAIN
        if j > 0:
            p.space_before = Pt(8)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 10: DETERMINISTIC RULES VS. LLMS (E3)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Rule Baseline vs. LLMs: Where Does an LLM Add Value? (E3)")

    add_card(s10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1), title="Closed Vocabulary: Rules Dominate")
    tbox = s10.shapes.add_textbox(Inches(1.1), Inches(2.6), Inches(5.0), Inches(4.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    rule_pts = [
        ("Speed & Overhead:", "12 if-statements run in 0.5 µs in Python (0 MB model overhead)."),
        ("Guaranteed Safety Priority:", "Deterministic ordering: safety > comfort > lights."),
        ("LLM Policy Misalignment:", "Base LLMs agree with stated policy in only 38%–62% of closed cases."),
        ("   - [LOW_LIGHT] Error:", "All 3 LLMs turned lights off in sedentary state (misread as 'dim lights')."),
        ("   - Inappropriate Escalation:", "Models trigger false security alarms on mere humidity spikes.")
    ]
    for j, (k, v) in enumerate(rule_pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_MAIN if j < 3 else COLOR_ACCENT
        if j > 0:
            p.space_before = Pt(8)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MUTED

    add_card(s10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1), title="Open Space: LLM Fallback (Probe Set)")
    tbox2 = s10.shapes.add_textbox(Inches(7.1), Inches(2.6), Inches(5.1), Inches(4.0))
    tf2 = tbox2.text_frame
    tf2.word_wrap = True
    probe_pts = [
        ("Unseen Sensor Types (8 cases):", "LLMs achieve 75% acceptable actions on novel events ([SMOKE_DETECTED], [WATER_LEAK]). Rules score 0%."),
        ("Natural Language Text (6 cases):", "Resident inputs ('I am going to bed now', 'Intruder at door'): LLMs score 50%–83%. Rules score 0%."),
        ("Natural Rationale Generation:", "LLMs provide clear context ('potential security breach...'), enabling human auditing."),
        ("Core Takeaway:", "LLMs should NEVER replace rules on known states; they act as an intelligent fallback for unmodeled exceptions!")
    ]
    for j, (k, v) in enumerate(probe_pts):
        p = tf2.paragraphs[0] if j == 0 else tf2.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_PRIMARY if "Takeaway" not in k else COLOR_ACCENT
        if j > 0:
            p.space_before = Pt(8)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MAIN

    # =========================================================================
    # SLIDE 11: QLORA FINE-TUNING & LATENCY GAINS (MERGED E4)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "QLoRA Fine-Tuning: Elimination of Tail Latency (E4)")

    plot_path = "dissertation/slides_assets/finetune_latency-1.png"
    if os.path.exists(plot_path):
        s11.shapes.add_picture(plot_path, Inches(0.8), Inches(1.8), width=Inches(6.8))

    add_card(s11, Inches(7.9), Inches(1.8), Inches(4.6), Inches(5.1), title="Fine-Tuning Impact")
    tbox = s11.shapes.add_textbox(Inches(8.15), Inches(2.55), Inches(4.1), Inches(4.1))
    tf = tbox.text_frame
    tf.word_wrap = True
    ft_pts = [
        ("Tail Latency Eliminated:", "Base models suffered long verbosity (P95 up to 7.6s). Fine-tuned models emit compact JSON directly."),
        ("Qwen 2.5 FT (1,015 ms):", "Fastest model in entire study (median ~1.0 s, P95 1.76 s)."),
        ("LLaMA 3.2 FT (1,785 ms):", "P95 latency cut from 5.8 s to 2.1 s (p < 0.001)."),
        ("Phi-3.5 FT (2,472 ms):", "Median latency cut by 58% (5.85 s -> 2.47 s)."),
        ("Strict JSON Validity (ASVR):", "LLaMA: 65% -> 95% | Qwen: 0% -> 84% | Phi: 0% -> 94%."),
        ("Quantization Trap Solved:", "Must merge adapter into dequantized NF4 training weights, NOT original FP16 (prevents accuracy drop from 100% to 60%).")
    ]
    for j, (k, v) in enumerate(ft_pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_ACCENT if "Trap" in k else COLOR_TEXT_MAIN
        if j > 0:
            p.space_before = Pt(6)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 12: MULTI-CRITERIA MODEL EVALUATION (TABLE 5.15)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Multi-Criteria Evaluation: Which Model Performs Best? (RQ4)")

    table_shape = s12.shapes.add_table(10, 7, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.3))
    table = table_shape.table
    table.columns[0].width = Inches(3.2)
    for c in range(1, 7):
        table.columns[c].width = Inches(1.42)

    headers = ["Evaluation Metric", "LLaMA Base", "LLaMA FT", "Qwen Base", "Qwen FT", "Phi Base", "Phi FT"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = RGBColor(255, 255, 255)
            p.alignment = PP_ALIGN.CENTER

    data = [
        ("Median Latency (ms)", "2,022", "1,785", "1,389", "1,015 (Best)", "5,846", "2,472"),
        ("95th-Percentile Latency (ms)", "5,796", "2,143", "2,718", "1,761 (Best)", "7,641", "3,111"),
        ("Model Memory (MB)", "2,041", "2,041", "1,011", "1,011 (Best)", "2,274", "2,485"),
        ("Energy / Decision (J)", "62.4", "34.8", "28.4", "21.3 (Best)", "114.9", "53.5"),
        ("Strict JSON (ASVR)", "65%", "95% (Best)", "0%", "84%", "0%", "94%"),
        ("Recoverable Action Rate", "80%", "95%", "97% (Best)", "85%", "91%", "94%"),
        ("Unseen Events Score (Probe)", "75% (Best)", "62%", "75% (Best)", "38%", "75% (Best)", "50%"),
        ("Free-Text Score (Probe)", "83% (Best)", "50%", "50%", "17%", "67%", "33%"),
        ("Best Criteria Count (of 12)", "4", "3", "4", "4", "2", "0")
    ]

    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if r % 2 == 0 else RGBColor(241, 245, 249)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = COLOR_TEXT_MAIN
                if "(Best)" in val:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_ACCENT
                if c > 0:
                    p.alignment = PP_ALIGN.CENTER

    nbox = s12.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(11.733), Inches(0.8))
    np_tf = nbox.text_frame
    np_tf.word_wrap = True
    p_note = np_tf.paragraphs[0]
    p_note.text = "Key Verdict: LLaMA 3.2 FT is the best overall single model (high validity + low latency). Qwen 2.5 FT is the most resource-efficient. Base models remain superior for open-domain free text."
    p_note.font.size = Pt(13)
    p_note.font.bold = True
    p_note.font.color.rgb = COLOR_PRIMARY

    # =========================================================================
    # SLIDE 13: THE RECOMMENDED 3-TIER HYBRID ARCHITECTURE
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13)
    add_header(s13, "Proposed Solution: The 3-Tier Hybrid Architecture")

    tiers = [
        ("Tier 1: Deterministic Rules", "0.5 µs Latency | 0 MB RAM", [
            "Handles closed vocabulary sensor events directly.",
            "Deterministic safety priority: safety > comfort > lights.",
            "Instantaneous lighting control (<1 ms response)."
        ]),
        ("Tier 2: Validated Cache", "0.43 µs Latency | Minimal RAM", [
            "Intercepts repeating symbolic inputs (98.2% hit rate).",
            "Stores validated decisions; one-time check against policy.",
            "Reduces continuous streaming mean latency to ~56 ms."
        ]),
        ("Tier 3: Local LLM Fallback", "1 - 2 s Latency | 1 - 2 GB RAM", [
            "Invoked ONLY on cache miss or open-domain input.",
            "Interprets novel sensor types (smoke, gas leak, air stale).",
            "Parses free-text resident requests with natural rationale."
        ])
    ]

    for i, (ttitle, tsub, tbullets) in enumerate(tiers):
        cx = Inches(0.8 + i * 4.0)
        cw = Inches(3.73)
        add_card(s13, cx, Inches(1.8), cw, Inches(5.1), title=ttitle, border_color=COLOR_PRIMARY)
        
        tbox = s13.shapes.add_textbox(cx + Inches(0.2), Inches(2.55), cw - Inches(0.4), Inches(4.1))
        tf = tbox.text_frame
        tf.word_wrap = True
        
        p_sub = tf.paragraphs[0]
        p_sub.text = tsub
        p_sub.font.size = Pt(13)
        p_sub.font.bold = True
        p_sub.font.color.rgb = COLOR_ACCENT
        p_sub.space_before = Pt(0)

        for j, b in enumerate(tbullets):
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(14)
            p.font.color.rgb = COLOR_TEXT_MAIN
            p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 14: CONCLUSIONS & ANSWERS TO RQS
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14)
    add_header(s14, "Conclusions: Final Synthesis & Takeaways")

    card_concl = add_card(s14, Inches(0.8), Inches(1.8), Inches(11.733), Inches(5.1), title="Answers to Core Research Questions")
    tbox = s14.shapes.add_textbox(Inches(1.1), Inches(2.55), Inches(11.1), Inches(4.2))
    tf = tbox.text_frame
    tf.word_wrap = True

    concl_items = [
        ("1. Edge Viability (RQ1):", "4-bit LLMs (1.5B–3.8B) fit comfortably in 1.0–2.3 GB DRAM, delivering 22–36 tok/s. Latency meets 10s P95 comfort control, but cannot satisfy sub-second lighting needs."),
        ("2. QLoRA Acceleration (RQ2):", "Fine-tuning raises strict JSON from 0–65% to 84–95% and cuts latency by 12–58% by shortening generated output. Adapter must merge into dequantized NF4 weights."),
        ("3. The True Role of Semantic Tokens (RQ3):", "Tokens save only ~40 ms of prefill, but collapse the infinite continuous sensor space to allow a response cache that answers 98.2% of decisions in 0.43 µs (56 ms avg)."),
        ("4. Optimal Configuration (RQ4 & RQ5):", "Rules are 1,000,000x faster for known states. The optimal system is a 3-tier hybrid: Rules -> Validated Cache -> LLM Fallback (LLaMA 3.2 base/FT).")
    ]
    for j, (k, v) in enumerate(concl_items):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"{k} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = COLOR_PRIMARY
        if j > 0:
            p.space_before = Pt(10)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MAIN

    # =========================================================================
    # SLIDE 15: CLOSURE SLIDE (MINIATURES FOR Q&A AS INSTRUCTED)
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15)
    add_header(s15, "Summary & Discussion", category="Q&A SESSION")

    p1 = "dissertation/slides_assets/pipeline_flow_perfect.png"
    p2 = "dissertation/slides_assets/multi_latency-1.png"
    p3 = "dissertation/slides_assets/finetune_latency-1.png"

    if os.path.exists(p1):
        add_card(s15, Inches(0.8), Inches(1.8), Inches(3.7), Inches(2.4))
        s15.shapes.add_picture(p1, Inches(0.9), Inches(1.9), width=Inches(3.5))

    if os.path.exists(p2):
        add_card(s15, Inches(4.8), Inches(1.8), Inches(3.7), Inches(2.4))
        s15.shapes.add_picture(p2, Inches(4.9), Inches(1.9), width=Inches(3.5))

    if os.path.exists(p3):
        add_card(s15, Inches(8.8), Inches(1.8), Inches(3.7), Inches(2.4))
        s15.shapes.add_picture(p3, Inches(8.9), Inches(1.9), width=Inches(3.5))

    add_card(s15, Inches(0.8), Inches(4.5), Inches(11.733), Inches(2.4), title="Thank You! Questions & Discussion Welcome")
    tbox = s15.shapes.add_textbox(Inches(1.1), Inches(5.2), Inches(11.1), Inches(1.5))
    tf = tbox.text_frame
    tf.word_wrap = True

    p_c1 = tf.paragraphs[0]
    p_c1.text = "Candidate: Sadra Farshid | Email: up202111516@fe.up.pt"
    p_c1.font.size = Pt(16)
    p_c1.font.bold = True
    p_c1.font.color.rgb = COLOR_TEXT_MAIN

    p_c2 = tf.add_paragraph()
    p_c2.text = "Supervisor: Prof. João Carlos Pascoal Faria | FEUP — University of Porto"
    p_c2.font.size = Pt(15)
    p_c2.font.color.rgb = COLOR_TEXT_MUTED
    p_c2.space_before = Pt(4)

    p_c3 = tf.add_paragraph()
    p_c3.text = "Repository: https://github.com/sadra-fa/EdgeIOT (All code, datasets, & raw experiment logs published)"
    p_c3.font.size = Pt(14)
    p_c3.font.color.rgb = COLOR_PRIMARY
    p_c3.space_before = Pt(6)

    # =========================================================================
    # BACKUP SLIDES (IN RESERVE FOR TECHNICAL QUESTIONS)
    # =========================================================================

    # BACKUP 1: THREATS TO VALIDITY & LIMITATIONS
    sb1 = prs.slides.add_slide(blank_layout)
    set_slide_background(sb1)
    add_header(sb1, "Backup: Threats to Validity & Methodological Limitations", category="BACKUP SLIDE — FOR Q&A")

    add_card(sb1, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1), title="Methodological Limitations")
    tbox = sb1.shapes.add_textbox(Inches(1.1), Inches(2.6), Inches(5.0), Inches(4.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    threat_pts = [
        ("Synthetic Dataset (RealLife.csv):", "Downloaded from Kaggle without provenance; physically impossible values (negative lux/motion); no temporal autocorrelation."),
        ("No Anomaly Labels:", "Isolation Forest flags exactly 5% by construction (contamination parameter); acts as multivariate outlier filter, not ground truth."),
        ("Circular Upstream Stages:", "Behavior labels included in features, making XGBoost state classification trivial (100% sanity check)."),
        ("Hardware Specificity:", "Evaluated on one Orin Nano board, one JetPack version, MAXN_SUPER mode.")
    ]
    for j, (k, v) in enumerate(threat_pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_ACCENT
        if j > 0:
            p.space_before = Pt(8)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MAIN

    add_card(sb1, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1), title="Future Work")
    tbox2 = sb1.shapes.add_textbox(Inches(7.1), Inches(2.6), Inches(5.1), Inches(4.0))
    tf2 = tbox2.text_frame
    tf2.word_wrap = True
    future_pts = [
        ("Real Sensor Deployment:", "Deploy physical sensors with timestamps in a live home environment to test temporal anomalies."),
        ("Policy-Supervised Fine-Tuning:", "Train LoRA adapters directly on deterministic priority rules rather than self-distilled teacher logs."),
        ("Alternative Serving Engines:", "Evaluate MLC-LLM / TensorRT-LLM (reported ~43 tokens/s vs. Ollama's 22 tokens/s)."),
        ("Grammar-Constrained Decoding:", "Benchmark GBNF grammar constraints directly against QLoRA fine-tuning for schema adherence.")
    ]
    for j, (k, v) in enumerate(future_pts):
        p = tf2.paragraphs[0] if j == 0 else tf2.add_paragraph()
        p.text = f"• {k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_PRIMARY
        if j > 0:
            p.space_before = Pt(8)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MAIN

    # BACKUP 2: QLORA QUANTIZATION CHAIN DETAILS
    sb2 = prs.slides.add_slide(blank_layout)
    set_slide_background(sb2)
    add_header(sb2, "Backup: The QLoRA Quantization Chain (Table 5.12)", category="BACKUP SLIDE — FOR Q&A")

    table_chain = sb2.shapes.add_table(5, 7, Inches(0.8), Inches(1.8), Inches(11.733), Inches(3.2))
    tc = table_chain.table
    tc.columns[0].width = Inches(2.733)
    tc.columns[1].width = Inches(1.5)
    tc.columns[2].width = Inches(1.5)
    tc.columns[3].width = Inches(1.5)
    tc.columns[4].width = Inches(1.5)
    tc.columns[5].width = Inches(1.5)
    tc.columns[6].width = Inches(1.5)

    headers_c = ["Model", "Adapter (NF4)", "FP16 Base", "FP16 Merge", "NF4 Q8_0", "NF4 Q4_K_M", "NF4 Q4_0"]
    for c, h in enumerate(headers_c):
        cell = tc.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = RGBColor(255, 255, 255)
            p.alignment = PP_ALIGN.CENTER

    chain_rows = [
        ("LLaMA 3.2 (3B)", "90%", "95%", "-", "89%", "95%", "-"),
        ("Qwen 2.5 (1.5B)", "100%", "60%", "100%", "100%", "80%", "-"),
        ("Phi-3.5 (3.8B)", "100%", "-", "-", "100%", "95%", "0% (Collapsed)")
    ]
    for r, row in enumerate(chain_rows):
        for c, val in enumerate(row):
            cell = tc.cell(r + 1, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if r % 2 == 0 else RGBColor(241, 245, 249)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = COLOR_TEXT_MAIN
                if "Collapsed" in val:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_ACCENT
                if c > 0:
                    p.alignment = PP_ALIGN.CENTER

    cbox = sb2.shapes.add_textbox(Inches(0.8), Inches(5.3), Inches(11.733), Inches(1.8))
    ctf = cbox.text_frame
    ctf.word_wrap = True
    p_c1 = ctf.paragraphs[0]
    p_c1.text = "• Adapter Merge Integrity: Standard merge into original FP16 weights destroyed Qwen validity (fell from 100% to 60%). Merging into dequantized NF4 training weights preserved 100% validity."
    p_c1.font.size = Pt(14)
    p_c1.font.bold = True
    p_c1.font.color.rgb = COLOR_ACCENT
    p_c2 = ctf.add_paragraph()
    p_c2.text = "• Quantization Fragility: Q4_0 completely collapsed fine-tuned Phi-3.5 (0% validity, emitted end-of-text immediately). Must be served in Q4_K_M (95% validity)."
    p_c2.font.size = Pt(14)
    p_c2.font.color.rgb = COLOR_TEXT_MAIN
    p_c2.space_before = Pt(6)

    # BACKUP 3: QUALITATIVE PROBE TRANSCRIPTS (TABLE 5.17)
    sb3 = prs.slides.add_slide(blank_layout)
    set_slide_background(sb3)
    add_header(sb3, "Backup: Qualitative Probe Case Transcripts (E3 & Table 5.17)", category="BACKUP SLIDE — FOR Q&A")

    add_card(sb3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.1), title="Test Case Row 26,494 (Table 5.17)")
    tbox = sb3.shapes.add_textbox(Inches(1.1), Inches(2.6), Inches(5.0), Inches(4.0))
    tf = tbox.text_frame
    tf.word_wrap = True
    case_pts = [
        ("Input:", "[STATE_SEDENTARY] [TEMP_LOW] [HUMIDITY_LOW] [ANOMALY_ALERT]"),
        ("Rule Action:", "security_alert (anomaly priority)"),
        ("LLaMA 3.2 FT:", "security_alert ✓ (1,658 ms) — Correctly flags anomaly alert in strict JSON."),
        ("Qwen 2.5 FT:", "heat_up ✗ (1,797 ms) — Invents action outside allowed canonical vocabulary."),
        ("Phi-3.5 FT:", "ventilation_on ✓ (2,647 ms) — Inappropriate action for cold/dry environment.")
    ]
    for j, (k, v) in enumerate(case_pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = f"{k} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_PRIMARY if "Action" in k or "Input" in k else COLOR_TEXT_MAIN
        if j > 0:
            p.space_before = Pt(8)
        run = p.add_run()
        run.text = v
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_MUTED

    add_card(sb3, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.1), title="Qualitative Value of Rationale Field")
    tbox2 = sb3.shapes.add_textbox(Inches(7.1), Inches(2.6), Inches(5.1), Inches(4.0))
    tf2 = tbox2.text_frame
    tf2.word_wrap = True
    audit_pts = [
        ("Auditing Hallucinations:", "Rationales expose model reasoning errors before acting:"),
        ("   - Door opened during sleep:", "Phi hallucinated 'inhabitant likely asleep, turn lights off' instead of sounding security alarm."),
        ("   - [LOW_LIGHT] misread:", "LLaMA deduced 'inhabitant likely to sleep given low light', turning lights off instead of on."),
        ("Free Text Handling:", "On 'Someone is trying to open the back door!', LLaMA generated: 'potential security breach, so I will trigger a security alert'."),
        ("Auditability Value:", "Natural language rationale allows homeowners to inspect automated decisions.")
    ]
    for j, item in enumerate(audit_pts):
        if isinstance(item, tuple):
            k, v = item
            p = tf2.paragraphs[0] if j == 0 else tf2.add_paragraph()
            p.text = f"• {k} "
            p.font.bold = True
            p.font.size = Pt(14)
            p.font.color.rgb = COLOR_TEXT_MAIN
            run = p.add_run()
            run.text = v
            run.font.bold = False
            run.font.color.rgb = COLOR_TEXT_MUTED
        else:
            p = tf2.add_paragraph()
            p.text = item
            p.font.size = Pt(13)
            p.font.color.rgb = COLOR_ACCENT
        if j > 0:
            p.space_before = Pt(8)

    # BACKUP 4: CROSS-STUDY BENCHMARK COMPARISON (TABLE 5.20)
    sb4 = prs.slides.add_slide(blank_layout)
    set_slide_background(sb4)
    add_header(sb4, "Backup: Cross-Study Comparison on Jetson Orin Nano (Table 5.20)", category="BACKUP SLIDE — FOR Q&A")

    table_shape2 = sb4.shapes.add_table(5, 4, Inches(0.8), Inches(1.8), Inches(11.733), Inches(3.8))
    table2 = table_shape2.table
    table2.columns[0].width = Inches(2.5)
    table2.columns[1].width = Inches(2.5)
    table2.columns[2].width = Inches(3.3)
    table2.columns[3].width = Inches(3.433)

    headers2 = ["Study / Benchmark", "Setup & Platform", "Published Findings", "Our Findings (EdgeIOT)"]
    for c, h in enumerate(headers2):
        cell = table2.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = RGBColor(255, 255, 255)
            p.alignment = PP_ALIGN.CENTER

    bench_data = [
        ("NVIDIA Jetson AI Lab (2025)", "Jetson Orin Nano, MLC engine, INT4", "LLaMA 3.2: 43.1 tok/s\nPhi-3.5: 38.1 tok/s", "Ollama: 22.3 tok/s (LLaMA), 23.0 tok/s (Phi). Engine is 2x lever!"),
        ("Jang et al. (2025)", "Jetson Orin Nano, Ollama Q4_K_M, MMLU-Pro", "LLaMA: 32.9 s/query\nQwen: 14.0 s/query", "LLaMA: 2.08 s/decision\nQwen: 1.38 s/decision. Output length governs latency!"),
        ("Islam et al. (2025)", "Raspberry Pi & Jetson, llama.cpp", "Phi-3 is most energy-hungry; Orin GPU most efficient", "Agrees: Phi-3.5 consumes 108 J/decision vs. Qwen 28 J."),
        ("Cloud Systems (Sasha / SAGE)", "Cloud GPT-3.5 / GPT-4 via API", "1 - 45 s latency, $0.02 - 0.03 per command", "100% offline, 0 cost, 0 privacy leak, 1-2s latency, 56ms with cache.")
    ]

    for r, row in enumerate(bench_data):
        for c, val in enumerate(row):
            cell = table2.cell(r + 1, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if r % 2 == 0 else RGBColor(241, 245, 249)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = COLOR_TEXT_MAIN
                if c == 3:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_PRIMARY

    bbox = sb4.shapes.add_textbox(Inches(0.8), Inches(5.9), Inches(11.733), Inches(1.1))
    bf = bbox.text_frame
    bf.word_wrap = True
    p_b = bf.paragraphs[0]
    p_b.text = "Key Takeaway: Discrepancies between published benchmarks and domain applications stem directly from output token count and serving engine runtime architectures."
    p_b.font.size = Pt(14)
    p_b.font.bold = True
    p_b.font.color.rgb = COLOR_ACCENT

    # Save presentation
    output_path = "dissertation/EdgeIOT_Dissertation_Presentation.pptx"
    prs.save(output_path)
    print(f"Streamlined 10-minute presentation saved successfully to {output_path}")

if __name__ == "__main__":
    create_presentation()
