Library	Function
Streamlit	Main Web Framework & Reactive UI.
Pytesseract	Python wrapper for the Tesseract OCR engine.
Tesseract-OCR	The core Open-source OCR engine (System level).
Pillow (PIL)	Advanced image processing and manipulation.
NumPy / OpenCV	(Optional) Used for image pre-processing and noise reduction
Cloud & Deployment Infrastructure
Hosting Platform: Streamlit Cloud (Community Tier).

CI/CD Pipeline: Automated deployment via GitHub Integration.

Operating System (Cloud Environment): Debian-based Linux container.

System Dependencies: Managed via packages.txt (Installs tesseract-ocr, libtesseract-dev, and tesseract-ocr-eng).
Mobile-Ready UI: Responsive layout designed for on-the-field usage via smartphone cameras.

State Management: Streamlit’s session state for handling multiple file uploads without refresh.

Secure API Integration: Environment variables handled via Streamlit Secrets (if any API keys are used).
Live Application: https://aihakathon-jzsntongwqpzxkpckznk9k.streamlit.app/
