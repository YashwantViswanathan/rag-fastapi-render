async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const status = document.getElementById("status");

  if (!fileInput.files.length) {
    alert("Please select a file");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  status.textContent = "Processing questions and evaluating answer confidence...";

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
    status.textContent =
      `Processed ${data.processed} questions successfully. Confidence scores included in CSV output.`;

  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  }
}

function downloadCSV() {
  window.location.href = "/download";
}
