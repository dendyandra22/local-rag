const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const inputEl = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const fileInput = document.querySelector("#fileInput");
const fileListEl = document.querySelector("#fileList");
const statusText = document.querySelector("#statusText");
const newChatButton = document.querySelector("#newChat");
const clearFilesButton = document.querySelector("#clearFiles");

let history = [];
let files = [];
let selectedFiles = new Set();

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function clearWelcome() {
  const welcome = messagesEl.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function addMessage(role, content) {
  clearWelcome();
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role === "user" ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrapper.append(roleEl, bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderFiles() {
  fileListEl.innerHTML = "";
  if (!files.length) {
    const empty = document.createElement("p");
    empty.className = "file-meta";
    empty.textContent = "No files uploaded yet.";
    fileListEl.appendChild(empty);
    return;
  }

  for (const file of files) {
    const label = document.createElement("label");
    label.className = "file-item";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selectedFiles.has(file.name);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selectedFiles.add(file.name);
      else selectedFiles.delete(file.name);
    });

    const info = document.createElement("div");
    const name = document.createElement("div");
    name.className = "file-name";
    name.textContent = file.name;
    const meta = document.createElement("div");
    meta.className = "file-meta";
    meta.textContent = `${formatBytes(file.size)}${file.text_preview ? " - text ready" : " - saved"}`;
    info.append(name, meta);

    label.append(checkbox, info);
    fileListEl.appendChild(label);
  }
}

async function loadFiles() {
  const response = await fetch("/api/files");
  const data = await response.json();
  files = data.files || [];
  renderFiles();
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Upload failed");
  selectedFiles.add(data.file.name);
  await loadFiles();
}

function resizeInput() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
}

fileInput.addEventListener("change", async () => {
  const uploadQueue = Array.from(fileInput.files || []);
  if (!uploadQueue.length) return;

  setStatus("Uploading...");
  try {
    for (const file of uploadQueue) {
      await uploadFile(file);
    }
    setStatus("Upload complete");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    fileInput.value = "";
  }
});

inputEl.addEventListener("input", resizeInput);
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;

  addMessage("user", message);
  history.push({ role: "user", content: message });
  inputEl.value = "";
  resizeInput();
  sendButton.disabled = true;
  setStatus("Thinking...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history,
        files: Array.from(selectedFiles),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Chat request failed");
    const answer = data.answer || "";
    addMessage("assistant", answer);
    history.push({ role: "assistant", content: answer });
    setStatus("Ready");
  } catch (error) {
    addMessage("assistant", error.message);
    setStatus(error.message, true);
  } finally {
    sendButton.disabled = false;
    inputEl.focus();
  }
});

newChatButton.addEventListener("click", () => {
  history = [];
  messagesEl.innerHTML = `
    <div class="welcome">
      <h2>Ask your local RAG app</h2>
      <p>Upload files, select them in the sidebar, then send a question. The UI will call your local Python backend.</p>
    </div>
  `;
  setStatus("Ready");
});

clearFilesButton.addEventListener("click", () => {
  selectedFiles.clear();
  renderFiles();
});

loadFiles().catch((error) => setStatus(error.message, true));
