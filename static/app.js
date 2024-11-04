document.getElementById("nerForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    const text = document.getElementById("textInput").value;
    const response = await fetch("/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ text })
    });
    const result = await response.json();
    
    // Display results in a structured format
    displayResults(result.entities);
});

function displayResults(entities) {
    const resultsDiv = document.getElementById("resultsContent");
    resultsDiv.innerHTML = ""; // Clear previous results

    // Check if any entities are returned
    if (Object.keys(entities).length === 0) {
        resultsDiv.innerHTML = "<p>No high-confidence entities found.</p>";
        return;
    }

    // Display entities grouped by type in the desired format
    Object.entries(entities).forEach(([entityType, words]) => {
        const entityGroup = document.createElement("div");
        entityGroup.classList.add("entity-group");

        const title = document.createElement("h3");
        title.textContent = entityType; // Display entity type (e.g., Date, Government)
        entityGroup.appendChild(title);

        words.forEach(word => {
            const entityEl = document.createElement("p");
            entityEl.textContent = word; // Display the actual entity word
            entityGroup.appendChild(entityEl);
        });

        resultsDiv.appendChild(entityGroup);
    });
}
