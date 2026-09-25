# Ice Mind — Frontend + Backend (UI you asked for)

Same cream / navy UI as the polished prototype, plus a real FastAPI backend.

## Structure

```
IceMind-App/
├── frontend/     Vite app — the UI you wanted (index.html)
├── backend/      FastAPI — prediction, risk, routing, copilot, WebSocket
└── README.md
```

## Run (VS Code — 2 terminals)

### Terminal 1 — Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Terminal 2 — Frontend
```bash
cd frontend
npm install
npm run dev
```

Open: **http://localhost:5173**

(Frontend proxies `/api` and `/ws` to the backend.)

## Deploy

- **Frontend only (static UI):** deploy `frontend/` (or `npm run build` → `dist/`) to Vercel/Netlify/GitHub Pages  
- **Full stack:** host backend (Railway/Render/Fly) and point frontend API URL to it  

## UI

Cream + navy + ice blue, white text on blue buttons, ocean-style map, layer toggles, live tracking, voice briefing — the design you approved.

**Ice Mind** — Antarctic Digital Twin for Predictive Vessel Navigation
