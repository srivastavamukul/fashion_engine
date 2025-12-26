const API_URL = ""; // Relative path
const API_KEY = "dev-secret-key"; // Hardcoded for demo/MVP (In prod, use env/proxy)

const form = document.getElementById('generator-form');
const generateBtn = document.getElementById('generate-btn');
const spinner = document.querySelector('.spinner');
const btnText = document.querySelector('.btn-text');

const statusPanel = document.getElementById('status-panel');
const statusText = document.getElementById('status-text');
const logStream = document.getElementById('log-stream');
const resultsGrid = document.getElementById('results-grid');

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('image-upload');
const preview = document.getElementById('image-preview');

let uploadedImagePath = null;

// Dropzone Logic
dropzone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.style.backgroundImage = `url(${e.target.result})`;
        preview.classList.remove('hidden');
        document.getElementById('dropzone-text').textContent = "Image Selected";
    };
    reader.readAsDataURL(file);

    // Upload immediately
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData,
            headers: {
                'x-api-key': API_KEY
            }
        });
        const data = await res.json();
        uploadedImagePath = data.path;
        console.log("Uploaded:", uploadedImagePath);
    } catch (err) {
        console.error("Upload failed", err);
        alert("Image upload failed");
    }
});

// Form Submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const product_name = document.getElementById('product_name').value;
    const category = document.getElementById('category').value;
    const features = document.getElementById('features').value.split(',').map(s => s.trim());

    // UI State -> Loading
    generateBtn.disabled = true;
    btnText.textContent = "GENERATING...";
    spinner.classList.remove('hidden');

    statusPanel.classList.remove('hidden');
    const loaderRing = document.querySelector('.loader-ring');
    if (loaderRing) loaderRing.style.display = 'block';
    resultsGrid.innerHTML = ""; // Clear old results
    addLog("🚀 Submitting job...");

    const images = uploadedImagePath ? [uploadedImagePath] : [];

    try {
        // 1. Submit Job
        const res = await fetch(`${API_URL}/jobs`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': API_KEY
            },
            body: JSON.stringify({
                product_name,
                category,
                features,
                images
            })
        });

        const data = await res.json();
        const jobId = data.job_id;
        addLog(`Job ID: ${jobId}`);

        // 2. Poll Status
        pollJob(jobId);

    } catch (err) {
        console.error(err);
        addLog("❌ Error: " + err.message);
        resetUI();
    }
});

async function pollJob(jobId) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_URL}/jobs/${jobId}`);
            const data = await res.json();

            statusText.textContent = `Status: ${data.status.toUpperCase()}`;

            if (data.status === 'processing') {
                if (Math.random() > 0.7) addLog("⚙️ Engine processing...");
            }

            if (data.status === 'completed') {
                clearInterval(interval);
                renderResults(data.result.videos);
                addLog("✅ Job Completed!");
                resetUI();
            }

            if (data.status === 'failed') {
                clearInterval(interval);
                addLog("❌ Job Failed: " + data.error);
                resetUI();
            }

        } catch (err) {
            console.error("Polling error", err);
        }
    }, 2000);
}

function renderResults(videos) {
    resultsGrid.innerHTML = "";
    videos.forEach(vid => {
        const card = document.createElement('div');
        card.className = 'video-card';
        const metricsHtml = vid.metrics ? `
        <div class="metrics-grid">
            <div class="metric"><span class="label">Realism</span><div class="bar-bg"><div class="bar-fill" style="width:${vid.metrics.realism * 10}%"></div></div></div>
            <div class="metric"><span class="label">Brand</span><div class="bar-bg"><div class="bar-fill" style="width:${vid.metrics.brand_alignment * 10}%"></div></div></div>
            <div class="metric"><span class="label">Consistency</span><div class="bar-bg"><div class="bar-fill" style="width:${vid.metrics.visual_consistency * 10}%"></div></div></div>
        </div>
        ` : '';

        card.innerHTML = `
            <video src="${vid.file_path}" controls loop muted autoplay></video>
            <div class="card-info">
                <div class="header-row">
                     <div class="card-title">${vid.video_id}</div>
                     <span class="score-badge">★ ${vid.score.toFixed(1)}</span>
                </div>
                ${metricsHtml}
            </div>
        `;
        resultsGrid.appendChild(card);
    });
}

function addLog(msg) {
    const div = document.createElement('div');
    div.className = 'log-item';
    div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logStream.prepend(div);
}

function resetUI() {
    generateBtn.disabled = false;
    btnText.textContent = "GENERATE CAMPAIGN";
    spinner.classList.add('hidden');
    const loaderRing = document.querySelector('.loader-ring');
    if (loaderRing) loaderRing.style.display = 'none';
}
