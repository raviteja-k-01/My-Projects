# My Projects

Hi, I'm Ravi Teja.

I build Python applications across machine learning, computer vision, and retrieval-augmented generation. This repository groups the source code, implementation notes, and validation evidence for my projects.

---

## Computer Vision

Work involving object detection, tracking, foreground segmentation, and camera reconstruction.

<details open>
<summary><b>Explore projects (3)</b></summary>

<br>

**[Real-Time Warehouse Vision System](Computer-Vision/realtime-warehouse-vision-system)**

Prerecorded-video warehouse pipeline using a fine-tuned six-class YOLOv8 detector and ByteTrack. Includes QR crop decoding, track-level identity caching, box/pallet/zone association, PostgreSQL event persistence, and a Streamlit dashboard. The project does not process a live CCTV feed.

**[AI MatchMove](Computer-Vision/ai-matchmove)**

FastAPI and React application for VFX camera tracking. Supports video and PNG/JPEG/RGB EXR sequences, OpenCV optical-flow foreground analysis, optional reviewed SAM ViT-B masks, SIFT feature matching, PyCOLMAP sparse reconstruction, and Blender export.

**[Segmentation using SAM](Computer-Vision/Segmentation_using_SAM_model)**

Image and video segmentation using the Segment Anything Model.

</details>

---

## Generative AI & RAG

Projects involving document ingestion, retrieval, LLM applications, and AI assistants.

<details>
<summary><b>Explore projects (5)</b></summary>

<br>

1. [Enterprise RAG Assistant](Generative-AI/enterprise-rag-assistant-v1) — workplace-policy RAG assistant over 12 fictional PDFs and 1,985 chunks, with section-aware text/table ingestion, a Python FastAPI/Streamlit + Qdrant/BM25S/RRF pipeline, and a separate Cloudflare Workers/Vectorize/D1 deployment. Evaluation scope and limitations are documented in its README.
2. [Enterprise Knowledge Engine](Generative-AI/Enterprise-Knowledge-Engine)
3. [Enterprise RAG Engine](Generative-AI/enterprise-rag-engine)
4. [RBI Financial RAG Assistant](Generative-AI/rbi-financial-rag-assistant)
5. [Multimodal AI Assistant](Generative-AI/multimodal-ai-assistant)

</details>

---

## Machine Learning

<details>
<summary><b>Explore projects (1)</b></summary>

<br>

**[Customer Churn Prediction](Machine-Learning/Customer-Churn-Prediction)**

XGBoost churn classifier with leakage-safe preprocessing, customer-level train/test separation, a Streamlit single/batch prediction interface, and SHAP explanations. The README reports held-out metrics and dataset limitations.

</details>

---

## Data Analytics

<details>
<summary><b>Explore projects (2)</b></summary>

<br>

1. [Uber Ride Cancellations & Wasted Capacity Analysis](Data-Analytics/Uber-Ride-Cancellations-and-Wasted-Capacity-Analysis)
2. [Retail Sales Insights & Customer Segmentation](Data-Analytics/Retail-Sales-Insights-Customer-Segmentation)

</details>

---

## Python Projects

Some of my earlier Python projects and practice work.

<details>
<summary><b>Explore projects (3)</b></summary>

<br>

1. [ATM System](Python-Projects/ATM)
2. [My First Project](Python-Projects/my-first-project)
3. [Python Codegnan](Python-Projects/Python_Codegnan)

</details>

---

## Portfolio

<details>
<summary><b>View portfolio</b></summary>

<br>

[Personal Portfolio](Portfolio/Portfolio)

</details>

---

### Notes

Some projects require additional setup, datasets, API credentials, or model weights that are not included in this repository.

Earlier projects are retained as part of the development history. Check each project's README for its prerequisites, validation evidence, and known limitations.
