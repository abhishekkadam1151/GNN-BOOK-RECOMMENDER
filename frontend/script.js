const API_BASE = "http://127.0.0.1:5000";

async function getRecommendations() {
    const titleInput = document.getElementById("bookInput");
    const resultsDiv = document.getElementById("results");
    const title = titleInput.value.trim();

    if (!title) {
        resultsDiv.innerHTML = '<p class="error">⚠️ Please enter a book name.</p>';
        return;
    }

    // Show loading state
    resultsDiv.innerHTML = '<p class="loading">🔄 Finding recommendations...</p>';

    try {
        const res = await fetch(`${API_BASE}/recommend/${encodeURIComponent(title)}`);

        if (!res.ok) {
            const errorData = await res.json();
            resultsDiv.innerHTML = `<p class="error">❌ ${errorData.error || "Something went wrong."}</p>`;
            return;
        }

        const data = await res.json();

        if (data.length === 0) {
            resultsDiv.innerHTML = '<p class="error">No recommendations found.</p>';
            return;
        }

        let output = "<h2>Recommended Books:</h2>";
        data.forEach(book => {
            output += `<p>📖 ${book}</p>`;
        });

        resultsDiv.innerHTML = output;
    } catch (err) {
        console.error("Fetch error:", err);
        resultsDiv.innerHTML = '<p class="error">❌ Could not connect to the backend. Make sure the server is running on port 5000.</p>';
    }
}

// Allow pressing Enter to search
document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("bookInput");
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            getRecommendations();
        }
    });
});