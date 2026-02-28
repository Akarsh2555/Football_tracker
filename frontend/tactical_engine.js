// Set this to your Render URL after deployment!
// Example: const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? "http://127.0.0.1:8000/api" : "https://synapse-tactical-engine.onrender.com/api";
const API_BASE = "http://127.0.0.1:8000/api";
const PASS_COLORS = ["#d4af37", "#e6c280", "#c82a2a"]; // Adjusted to vintage theme
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
    const img = document.getElementById("pitch-img");
    if (isLoading) {
        overlay.style.display = "flex";
        img.style.display = "none";
    } else {
        overlay.style.display = "none";
        img.style.display = "block";
    }
}

function updateUI(data) {
    const { frame, latency_ms, attackers_count, defenders_count, ranked_passes, pitch_image_b64, intel_data } = data;

    // Update Top Header Nav Ticker Elements
    document.getElementById("ticker-frame").innerText = frame;
    document.getElementById("ticker-lat").innerText = `${Math.round(latency_ms)}ms`;

    if (intel_data) {
        const xgElem = document.getElementById("ticker-xg");
        const riskElem = document.getElementById("ticker-risk");
        const momElem = document.getElementById("ticker-mom");
        if (xgElem) xgElem.innerText = intel_data.contextual_xg ? intel_data.contextual_xg.toFixed(3) : "0.000";
        if (riskElem) riskElem.innerText = intel_data.risk_index ? intel_data.risk_index.toFixed(2) : "0.00";
        if (momElem) momElem.innerText = intel_data.momentum || "Neutral";
    }

    if (ranked_passes && ranked_passes.length > 0) {
        const bestPass = ranked_passes[0];
        document.getElementById("ticker-ev").innerText = `+${bestPass.ev.toFixed(2)}`;
    } else {
        document.getElementById("ticker-ev").innerText = "--";
    }

    // Update Pitch Image
    if (pitch_image_b64) {
        document.getElementById("pitch-img").src = "data:image/png;base64," + pitch_image_b64;
    }

    // Update Bottom Right Telemetry Elements
    document.getElementById("tele-att").innerText = attackers_count;
    document.getElementById("tele-def").innerText = defenders_count;
    document.getElementById("tele-opt").innerText = ranked_passes ? ranked_passes.length : 0;

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
            <div class="bg-v-dark-red-light p-4 rounded-lg border flex flex-col gap-4 relative overflow-hidden" style="border-color: ${col}40;">
                <div class="absolute -right-2 -top-4 text-[80px] font-display font-bold text-off-white opacity-5 pointer-events-none">${i + 1}</div>
                <div class="flex justify-between items-start z-10">
                    <div>
                        <p class="text-[9px] uppercase font-bold tracking-tighter" style="color: ${col};">Target ${name} - Z${i + 1}</p>
                        <p class="text-sm font-bold text-off-white">Execution Horizon</p>
                        <p class="text-[8px] font-mono text-off-white/40 mt-1">⊕ ${coord}</p>
                    </div>
                    <span class="material-symbols-outlined text-lg" style="color: ${col};">trending_up</span>
                </div>
                <div class="flex items-center gap-4 z-10">
                    <div class="relative size-12 shrink-0">
                        <svg class="size-full -rotate-90" viewBox="0 0 36 36">
                            <circle cx="18" cy="18" fill="none" r="16" stroke-width="3" style="stroke: ${col}20;"></circle>
                            <circle cx="18" cy="18" fill="none" r="16" stroke-dasharray="${dashOffset}, ${dashArrayFull}" stroke-linecap="round" stroke-width="3" style="stroke: ${col}; transition: stroke-dasharray 1s ease;"></circle>
                        </svg>
                        <div class="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-off-white">${pp.toFixed(0)}%</div>
                    </div>
                    <div class="flex-1">
                        <div class="flex justify-between text-[9px] text-off-white/60 mb-1 font-bold">
                            <span>EXPECTED VALUE</span>
                            <span style="color: ${col};">+${opt.ev.toFixed(3)}</span>
                        </div>
                        <div class="h-1 w-full rounded-full overflow-hidden" style="background-color: ${col}20;">
                            <div class="h-full rounded-full" style="width: ${Math.min((opt.ev / 0.5) * 100, 100).toFixed(1)}%; background-color: ${col};"></div>
                        </div>
                        
                        <div class="flex justify-between text-[9px] text-off-white/60 mb-1 font-bold mt-2">
                            <span>GOAL THREAT xG</span>
                            <span>${xg.toFixed(3)}</span>
                        </div>
                        <div class="h-1 w-full rounded-full overflow-hidden bg-off-white/10">
                            <div class="h-full rounded-full" style="width: ${Math.min(xg * 100, 100).toFixed(1)}%; background-color: #f8ecec;"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        passCardsContainer.innerHTML += cardHtml;
    }
}
