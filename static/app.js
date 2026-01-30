async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const status = document.getElementById("status");
  const tableBody = document.querySelector("#resultsTable tbody");

  tableBody.innerHTML = "";

  if (!fileInput.files.length) {
    alert("Please select a file");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  status.textContent = "Processing questions and generating answers...";

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Upload failed");
    }

    const data = await response.json();
    status.textContent = `Processed ${data.processed} questions successfully.`;

    data.results.forEach(row => {
      const tr = document.createElement("tr");

      const scoreClass =
        row.confidence_label === "High" ? "score-high" :
        row.confidence_label === "Medium" ? "score-medium" :
        "score-low";

      tr.innerHTML = `
        <td>${row.question}</td>
        <td>${row.answer}</td>
        <td>
          <span class="score-pill ${scoreClass}">
            ${row.confidence_score} (${row.confidence_label})
          </span>
        </td>
      `;

      tableBody.appendChild(tr);
    });

  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  }
}

function downloadCSV() {
  window.location.href = "/download";
}
