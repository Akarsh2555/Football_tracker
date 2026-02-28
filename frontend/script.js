const API_BASE = "http://127.0.0.1:8000/api";

// --- XG AR SLIDERS ---
const xSlider = document.getElementById('xg-x-slider');
const ySlider = document.getElementById('xg-y-slider');
const xVal = document.getElementById('val-x');
const yVal = document.getElementById('val-y');

xSlider.addEventListener('input', (e) => { xVal.innerText = `${e.target.value}m`; });
ySlider.addEventListener('input', (e) => { yVal.innerText = `${e.target.value}m`; });

// --- DOM Elements ---
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('video-upload');
const statusDiv = document.getElementById('upload-status');
const statusText = document.getElementById('status-text');

// --- File Upload Logic ---
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderLeftColor = "var(--accent-orange)";
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderLeftColor = "var(--accent-lime)";
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) handleUpload(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleUpload(e.target.files[0]);
});

async function handleUpload(file) {
    if (!file.type.startsWith('video/')) {
        alert("CRITICAL ERROR: INVALID EXTENSION.");
        return;
    }

    // Update UI
    dropZone.classList.add('hidden');
    statusDiv.classList.remove('hidden');

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_BASE}/upload-video`, { method: 'POST', body: formData });
        if (!response.ok) throw new Error("API REJECTED");
        const data = await response.json();
        statusText.innerText = "DEEP TACTICAL ANALYSIS IN PROGRESS...";
        pollJobStatus(data.job_id);
    } catch (error) {
        statusText.innerText = "SYSTEM FAILURE: " + error.message;
        setTimeout(() => resetApp(), 3000);
    }
}

async function pollJobStatus(jobId) {
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/jobs/${jobId}`);
            if (!response.ok) return;
            const job = await response.json();

            if (job.status === "completed") {
                clearInterval(interval);
                showDashboard(job);
            } else if (job.status === "failed") {
                clearInterval(interval);
                statusText.innerText = "OVERLOAD: " + (job.error || "Unknown Error");
                setTimeout(() => resetApp(), 5000);
            }
        } catch (e) {
            console.error("Polling error", e);
        }
    }, 5000);
}

function showDashboard(job) {
    statusText.innerText = "ANALYSIS COMPLETE. SCROLL DOWN TO OPTICS.";

    const annotatedVideo = document.getElementById('annotated-video');
    const tacticalMap = document.getElementById('tactical-map');

    annotatedVideo.src = `http://127.0.0.1:8000${job.annotated_video_url}`;
    tacticalMap.src = `http://127.0.0.1:8000${job.tactical_map_url}`;

    annotatedVideo.play();
    tacticalMap.play();

    // Smooth scroll down
    document.getElementById('results-dashboard').scrollIntoView({ behavior: 'smooth' });
}

function resetApp() {
    statusDiv.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = "";
}

// --- Live XG Tester API ---
async function queryXgModel() {
    const xInput = xSlider.value;
    const yInput = ySlider.value;
    const resultPercent = document.getElementById('xg-result-percent');
    const resultDec = document.getElementById('xg-result-dec');

    resultPercent.innerText = "CALC...";

    try {
        const response = await fetch(`${API_BASE}/predict-xg`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                player_pos: [parseFloat(xInput), parseFloat(yInput)],
                defenders_pos: []
            })
        });

        if (!response.ok) throw new Error("API Error");

        const data = await response.json();
        const rawScore = data.contextual_xg;

        // Broadcast formatting
        resultPercent.innerText = `${Math.round(rawScore * 100)}%`;
        resultDec.innerText = `LOGREG: ${rawScore.toFixed(3)}`;

        // Glow impact
        resultPercent.style.color = "#fff";
        setTimeout(() => {
            resultPercent.style.color = "transparent";
            resultPercent.style.background = "linear-gradient(to bottom, #fff, var(--text-muted))";
            resultPercent.style.webkitBackgroundClip = "text";
        }, 500);

    } catch (e) {
        resultPercent.innerText = "ERR";
    }
}

// --- RADAR CHARTS ---
document.addEventListener('DOMContentLoaded', () => {
    Chart.defaults.color = '#8b99af';
    Chart.defaults.font.family = 'Oswald';

    const radarData1 = {
        labels: ['PACE', 'SHOOTING', 'PASSING', 'xG/90', 'DEFENDING', 'PHYSICAL'],
        datasets: [{
            label: 'STRIKER',
            data: [88, 92, 75, 89, 40, 82],
            backgroundColor: 'rgba(66, 242, 13, 0.2)', // Lime
            borderColor: '#42f20d',
            pointBackgroundColor: '#42f20d',
            borderWidth: 2
        }]
    };

    const radarData2 = {
        labels: ['PACE', 'SHOOTING', 'PASSING', 'xG/90', 'DEFENDING', 'PHYSICAL'],
        datasets: [{
            label: 'PLAYMAKER',
            data: [78, 80, 94, 65, 70, 75],
            backgroundColor: 'rgba(255, 59, 0, 0.2)', // Orange
            borderColor: '#ff3b00',
            pointBackgroundColor: '#ff3b00',
            borderWidth: 2
        }]
    };

    const radarOptions = {
        scales: {
            r: {
                angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                pointLabels: { color: '#fff', font: { size: 12, weight: 'bold' } },
                ticks: { display: false, max: 100, min: 0 }
            }
        },
        plugins: { legend: { display: false } }
    };

    new Chart(document.getElementById('radarChart1'), { type: 'radar', data: radarData1, options: radarOptions });
    new Chart(document.getElementById('radarChart2'), { type: 'radar', data: radarData2, options: radarOptions });
});

// --- PARALLAX INTERACTION ---
document.addEventListener('mousemove', (e) => {
    const parallaxEl = document.querySelector('.parallax-element');
    if (parallaxEl) {
        // Calculate slight opposing movement to create depth against the 3D background
        const x = (window.innerWidth / 2 - e.pageX) / 40;
        const y = (window.innerHeight / 2 - e.pageY) / 40;
        parallaxEl.style.transform = `translate(${x}px, ${y}px)`;
    }
});
