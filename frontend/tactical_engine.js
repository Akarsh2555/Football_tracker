// Set this to your Render URL after deployment!
// Example: const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? "http://127.0.0.1:8000/api" : "https://synapse-tactical-engine.onrender.com/api";
const API_BASE = "http://127.0.0.1:8000/api";
const PASS_COLORS = ["#ccff00", "#ffffff", "#ff2a4d"]; // Brutalist colors
const TARGET_NAMES = ["ALPHA", "BETA", "GAMMA"];
const RANK_LABELS = ["1ST", "2ND", "3RD"];

document.addEventListener("DOMContentLoaded", () => {
    const slider = document.getElementById("frame-slider");

    // Initial load
    fetchTacticalData(slider.value);

    // Update on slider change
    slider.addEventListener("change", (e) => {
        fetchTacticalData(e.target.value);
    });
});

async function fetchTacticalData(frame) {
    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/tactical-engine/${frame}`);
        if (!response.ok) throw new Error("Backend request failed");

        const data = await response.json();
        updateUI(data);
    } catch (error) {
        console.error("Error fetching tactical data:", error);
        alert("Failed to fetch tactical data. Ensure FastAPI backend is running.");
    } finally {
        showLoading(false);
    }
}

function showLoading(isLoading) {
    const overlay = document.getElementById("loading-overlay");
    if (isLoading) {
        overlay.style.display = "flex";
    } else {
        overlay.style.display = "none";
    }
}

function updateUI(data) {
    const { frame, latency_ms, attackers_count, defenders_count, ranked_passes, pitch_image_b64, intel_data } = data;

    // Update Top Header Nav Ticker Elements
    const tfElem = document.getElementById("ticker-frame");
    if (tfElem) tfElem.innerText = frame;

    const latElem = document.getElementById("ticker-lat");
    if (latElem) latElem.innerText = `${Math.round(latency_ms)}ms`;

    if (intel_data) {
        const xgElem = document.getElementById("ticker-xg");
        const riskElem = document.getElementById("ticker-risk");
        const momElem = document.getElementById("ticker-mom");
        if (xgElem) xgElem.innerText = intel_data.contextual_xg ? intel_data.contextual_xg.toFixed(3) : "0.000";
        if (riskElem) riskElem.innerText = intel_data.risk_index ? intel_data.risk_index.toFixed(2) : "0.00";
        if (momElem) momElem.innerText = intel_data.momentum || "Neutral";
    }

    const evElem = document.getElementById("ticker-ev");
    if (evElem) {
        if (ranked_passes && ranked_passes.length > 0) {
            const bestPass = ranked_passes[0];
            evElem.innerText = `+${bestPass.ev.toFixed(2)}`;
        } else {
            evElem.innerText = "--";
        }
    }

    // 3D rendering removed. 2D overlay and AI chat is now prioritized.

    // Update Bottom Right Telemetry Elements
    const teleAtt = document.getElementById("tele-att");
    const teleDef = document.getElementById("tele-def");
    const teleOpt = document.getElementById("tele-opt");
    if (teleAtt) teleAtt.innerText = attackers_count;
    if (teleDef) teleDef.innerText = defenders_count;
    if (teleOpt) teleOpt.innerText = ranked_passes ? ranked_passes.length : 0;

    // Update Pass Intelligence Cards
    const passCardsContainer = document.getElementById("pass-cards-container");
    passCardsContainer.innerHTML = ""; // Clear existing

    if (!ranked_passes || ranked_passes.length === 0) {
        passCardsContainer.innerHTML = `<div class="text-off-white/40 text-center text-[10px] p-4 font-bold border border-v-gold/10 rounded">No pass options found for this frame.</div>`;
        return;
    }

    const numCards = Math.min(3, ranked_passes.length);
    for (let i = 0; i < numCards; i++) {
        const opt = ranked_passes[i];
        const col = PASS_COLORS[i];
        const name = TARGET_NAMES[i];
        const rankLabel = RANK_LABELS[i];
        const pp = opt.pass_prob * 100;
        const xg = opt.xg;
        const coord = `(${opt.pos[0].toFixed(1)}, ${opt.pos[1].toFixed(1)})`;

        // Circumference for stroke-dasharray is 2 * pi * r. For r=16, it's ~100.5
        const dashArrayFull = 100.5;
        const dashOffset = dashArrayFull * (pp / 100);

        const cardHtml = `
            <div class="bg-black p-4 md:p-6 border-2 flex flex-col gap-4 relative overflow-hidden min-w-[280px]" style="border-color: ${col};">
                <div class="absolute -right-4 -top-6 text-[120px] font-oswald font-black opacity-10 pointer-events-none" style="color: ${col};">${i + 1}</div>
                <div class="flex justify-between items-start z-10 border-b-2 pb-2" style="border-color: ${col}40;">
                    <div>
                        <p class="text-xs uppercase font-oswald font-bold tracking-widest bg-black px-1" style="color: ${col}; border: 1px solid ${col}; display: inline-block;">TGT ${name}</p>
                        <p class="text-lg font-oswald font-bold text-white uppercase mt-2 w-max">EXECUTION HORIZON</p>
                        <p class="text-xs font-mono text-[#8a8a8a] mt-1 tracking-widest border-l-2 pl-2" style="border-color: ${col};">⊕ ${coord}</p>
                    </div>
                </div>
                <div class="flex items-center gap-6 z-10 w-full mt-2">
                    <div class="relative size-16 shrink-0 border-2 bg-[#111] flex items-center justify-center" style="border-color: ${col};">
                        <div class="absolute inset-0 bg-transparent" style="height: ${100 - pp}%;"></div>
                        <div class="absolute bottom-0 left-0 right-0 transition-all opacity-20" style="height: ${pp}%; background-color: ${col};"></div>
                        <div class="text-xl font-oswald font-black z-10" style="color: ${col};">${pp.toFixed(0)}<span class="text-xs">%</span></div>
                    </div>
                    <div class="flex-1 w-full">
                        <div class="flex justify-between text-xs text-[#8a8a8a] mb-1 font-oswald font-bold uppercase tracking-widest">
                            <span>VALUE_EV</span>
                            <span style="color: ${col};">+${opt.ev.toFixed(3)}</span>
                        </div>
                        <div class="h-2 w-full bg-[#111] border border-[#333]">
                            <div class="h-full transition-all" style="width: ${Math.min((opt.ev / 0.5) * 100, 100).toFixed(1)}%; background-color: ${col};"></div>
                        </div>
                        
                        <div class="flex justify-between text-xs text-[#8a8a8a] mb-1 font-oswald font-bold uppercase tracking-widest mt-4">
                            <span>THREAT_xG</span>
                            <span class="text-white">${xg.toFixed(3)}</span>
                        </div>
                        <div class="h-2 w-full bg-[#111] border border-[#333]">
                            <div class="h-full transition-all bg-white" style="width: ${Math.min(xg * 100, 100).toFixed(1)}%;"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        passCardsContainer.innerHTML += cardHtml;
    }
}