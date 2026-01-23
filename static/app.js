async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const status = document.getElementById("status");

  if (!fileInput.files.length) {
    alert("Please select a file");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  status.textContent = "Processing questions...";

  const response = await fetch("/upload", {
    method: "POST",
    body: formData
  });

  const data = await response.json();
  status.textContent = `Processed ${data.processed} questions.`;
}

function downloadCSV() {
  window.location.href = "/download";
}
