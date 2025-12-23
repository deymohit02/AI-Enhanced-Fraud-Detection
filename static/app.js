const API_URL = ''; // Relative path

// Elements
const form = document.getElementById('prediction-form');
const analyzeBtn = document.getElementById('analyze-btn');
const btnLoader = document.getElementById('btn-loader');
const resultCard = document.getElementById('result-card');
const scoreRing = document.getElementById('score-ring');
const scoreVal = document.getElementById('score-val');
const riskBadge = document.getElementById('risk-badge');
const resultMsg = document.getElementById('result-msg');
const explainBtn = document.getElementById('explain-btn');
const explanationCard = document.getElementById('explanation-card');
const explanationContent = document.getElementById('explanation-content');

let currentTransactionId = null;
let currentProb = 0;

// Load Stats
async function loadStats() {
    try {
        const res = await fetch(`${API_URL}/api/stats`);
        const data = await res.json();
        if (data.total_transactions) {
            document.getElementById('tx-count').innerText = data.total_transactions;
        }
    } catch (e) {
        console.error("Stats load failed", e);
    }
}

loadStats();

// Form Submit
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setLoading(true);

    const amount = parseFloat(document.getElementById('amount').value);
    const time = parseFloat(document.getElementById('time').value);

    try {
        // Auto-Simulate Fraud for High Amounts (Demo Logic)
        if (amount > 25000 && !document.getElementById('simulate-fraud').checked) {
            document.getElementById('simulate-fraud').checked = true;
            // Visual feedback
            const msg = document.createElement('div');
            msg.className = 'alert alert-warning';
            msg.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); z-index: 1000; animation: fadeIn 0.5s;';
            msg.innerText = '⚠️ High Amount Detected: Auto-simulating fraud pattern for demo purposes.';
            document.body.appendChild(msg);
            setTimeout(() => msg.remove(), 4000);
        }

        // Generate realistic V-features based on amount and time
        const features = generateRealisticFeatures(amount, time);

        const res = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });

        if (!res.ok) {
            const error = await res.json();
            console.error('API Error:', error);
            alert(`Error: ${error.error || 'Failed to analyze transaction'}\n\n${error.message || ''}`);
            setLoading(false);
            return;
        }

        const data = await res.json();
        showResult(data);
        loadStats(); // Update counter
    } catch (err) {
        alert("Error analyzing transaction");
        console.error(err);
    } finally {
        setLoading(false);
    }
});

// Generate realistic feature vector
// Generate realistic feature vector
function generateRealisticFeatures(amount, time) {
    const simulateFraud = document.getElementById('simulate-fraud').checked;

    if (simulateFraud) {
        const fraudVFeatures = [
            -10.2817840384715, 6.30238478416897, -13.271718028752, 8.92511547634157,
            -9.97557831146449, -2.83251346361162, -12.7032526593299, 6.70684583868978,
            -7.07842395823848, -12.8056831898117, 6.78605830197451, -13.0642398936784,
            1.1795245105938, -13.694873039573, 0.951479176238826, -10.9542857275047,
            -20.5835927904158, -7.51726213169106, 2.87235363605874, -0.247647527731929,
            2.47941350799644, 0.366932842343897, 0.0428046480276363, 0.478278891799746,
            0.157770817898023, 0.329900959238367, 0.163504340964582, -0.485552212695879
        ];

        // Add small noise
        const noise = 0.05;
        const v_features = fraudVFeatures.map(val => val + (Math.random() - 0.5) * noise);
        return [time, ...v_features, amount];
    } else {
        // Pre-defined sample V-features from real non-fraud transactions
        const sampleVFeatures = [
            -0.05, -0.01, -0.04, -0.04, -0.01, 0.02, 0.03, -0.01,
            0.00, -0.02, 0.01, 0.00, -0.01, 0.01, -0.01, 0.00,
            0.00, -0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00
        ];

        // Add some randomness based on amount
        const noise = (amount / 100) * 0.1;
        const v_features = sampleVFeatures.map(val => val + (Math.random() - 0.5) * noise);
        return [time, ...v_features, amount];
    }
}


// Explain Button
explainBtn.addEventListener('click', async () => {
    if (!currentTransactionId) return;

    explanationCard.classList.remove('hidden');
    explanationContent.innerHTML = `
        <div class="skeleton-text" style="width: 100%"></div>
        <div class="skeleton-text" style="width: 80%"></div>
        <div class="skeleton-text" style="width: 90%"></div>
    `;

    try {
        const amount = parseFloat(document.getElementById('amount').value);
        const time = parseFloat(document.getElementById('time').value);

        const res = await fetch(`${API_URL}/api/explain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                transaction_id: currentTransactionId,
                amount: amount,
                time: time,
                prob: currentProb
            })
        });

        const data = await res.json();

        // Format Explanation (Markdown to HTML simple)
        const text = data.explanation.replace(/\n/g, '<br>');
        explanationContent.innerHTML = `<p>${text}</p>`;

    } catch (e) {
        explanationContent.innerHTML = `<p class="text-danger">Failed to get explanation.</p>`;
    }
});

// Helpers
function setLoading(loading) {
    if (loading) {
        analyzeBtn.disabled = true;
        btnLoader.classList.remove('hidden');
        resultCard.classList.add('hidden');
        explanationCard.classList.add('hidden');
    } else {
        analyzeBtn.disabled = false;
        btnLoader.classList.add('hidden');
    }
}

function showResult(data) {
    currentTransactionId = data.transaction_id;
    currentProb = data.fraud_probability;

    resultCard.classList.remove('hidden');
    resultCard.classList.remove('safe', 'danger');
    riskBadge.classList.remove('safe', 'danger');

    const percentage = (data.fraud_probability * 100).toFixed(1) + '%';
    scoreVal.innerText = percentage;

    if (data.is_fraud) {
        resultCard.classList.add('danger');
        riskBadge.classList.add('danger');
        riskBadge.innerText = 'High Risk';
        resultMsg.innerText = '⚠️ Potential Fraud Detected!';
        scoreRing.style.borderColor = 'var(--danger)';
    } else {
        resultCard.classList.add('safe');
        riskBadge.classList.add('safe');
        riskBadge.innerText = 'Low Risk';
        resultMsg.innerText = '✅ Transaction appears legitimate.';
        scoreRing.style.borderColor = 'var(--success)';
    }
}
