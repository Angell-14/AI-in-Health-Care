# Health AI Bot

## Installation

Follow these steps to set up and run the project:

### Frontend Setup
```sh
cd frontend
npm i
npm run dev
```

### Backend Setup
```sh
cd backend
npm i
npm run dev
```

### AI Backend Setup
```sh
cd aibackend
pip install -r requirements.txt
```
> **Note:** Ensure you have the necessary models available locally. These models are not included in the repository due to their size and proprietary nature.

---

## Features

1. **Medical Image Processing**
   - Supports analysis for Edema, Pneumonia, and general chest X-ray evaluations.

2. **Multilingual Text Processing**
   - Handles text in over 10 native Indian languages.

3. **Deployable & Private Chatbot**
   - The chatbot can be deployed and is designed with privacy in mind.
   - The underlying LLM is fine-tuned using a large dataset of relevant medical data.

4. **On-Premise LLM**
   - The LLM runs locally, ensuring no reliance on third-party APIs.
   - This provides lower latency and enhanced privacy.

5. **Personalized Health Plans**
   - Generates customized health, diet, and nutrition plans based on user inputs.

---

## Important Notes
- This chatbot is designed for **health and medicine-related queries**.
- Ensure you have the necessary **models downloaded** before running the AI backend.

