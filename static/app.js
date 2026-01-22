async function askQuestion() {
  const questionBox = document.getElementById("question");
  const answerBox = document.getElementById("answer");
  const loading = document.getElementById("loading");

  const question = questionBox.value.trim();
  if (!question) {
    alert("Please enter a question");
    return;
  }

  answerBox.textContent = "";
  loading.classList.remove("hidden");

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question })
    });

    const data = await response.json();
    answerBox.textContent = data.answer;
  } catch (err) {
    answerBox.textContent = "Error contacting backend.";
  } finally {
    loading.classList.add("hidden");
  }
}

function downloadCSV() {
  window.location.href = "/download";
}
