const messagesEl = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const newChatButton = document.querySelector("#newChatButton");
const statusDot = document.querySelector("#statusDot");
const statusText = document.querySelector("#statusText");
const promptChips = document.querySelectorAll(".prompt-chip");

let isSending = false;
let sessionId = localStorage.getItem("ragChatSessionId") || createSessionId();

localStorage.setItem("ragChatSessionId", sessionId);

function createSessionId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID().replaceAll("-", "");
  }

  return `${Date.now()}${Math.random().toString(16).slice(2)}`;
}

function removeEmptyState() {
  const emptyState = messagesEl.querySelector(".empty-state");
  if (emptyState) {
    emptyState.remove();
  }
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function addMessage(role, content = "") {
  removeEmptyState();

  const row = document.createElement("article");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "R";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  row.append(avatar, bubble);
  messagesEl.append(row);
  scrollToBottom();

  return bubble;
}

function setSending(nextValue) {
  isSending = nextValue;
  sendButton.disabled = nextValue;
  input.disabled = nextValue;
}

function renderEmptyState() {
  messagesEl.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">R</div>
      <h2>What manga are we exploring?</h2>
      <p>Ask for recommendations, authors, ratings, genres, or similar titles from your local RAG database.</p>
    </div>
  `;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("API health check failed");
    }
    statusDot.className = "status-dot online";
    statusText.textContent = "API online";
  } catch (error) {
    statusDot.className = "status-dot offline";
    statusText.textContent = "API offline";
  }
}

async function loadHistory() {
  try {
    const response = await fetch(`/history/${sessionId}`);
    if (!response.ok) {
      throw new Error("Could not load chat history");
    }

    const data = await response.json();
    if (!data.messages?.length) {
      renderEmptyState();
      return;
    }

    messagesEl.innerHTML = "";
    data.messages.forEach((message) => {
      addMessage(message.role, message.content);
    });
  } catch (error) {
    renderEmptyState();
  }
}

async function sendMessage(message) {
  if (!message.trim() || isSending) {
    return;
  }

  addMessage("user", message.trim());
  const assistantBubble = addMessage("assistant", "");
  assistantBubble.classList.add("loading");

  setSending(true);

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message.trim(),
        session_id: sessionId,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Chat request failed with status ${response.status}`);
    }

    const responseSessionId = response.headers.get("X-Session-Id");
    if (responseSessionId) {
      sessionId = responseSessionId;
      localStorage.setItem("ragChatSessionId", sessionId);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      assistantBubble.textContent += decoder.decode(value, { stream: true });
      scrollToBottom();
    }

    const tail = decoder.decode();
    if (tail) {
      assistantBubble.textContent += tail;
    }
  } catch (error) {
    assistantBubble.textContent = "Sorry, the local RAG API did not return a response. Make sure FastAPI and Ollama are running, then try again.";
  } finally {
    assistantBubble.classList.remove("loading");
    setSending(false);
    input.focus();
    checkHealth();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value;
  input.value = "";
  resizeInput();
  sendMessage(message);
});

input.addEventListener("input", resizeInput);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

newChatButton?.addEventListener("click", () => {
  sessionId = createSessionId();
  localStorage.setItem("ragChatSessionId", sessionId);
  renderEmptyState();
  input.value = "";
  resizeInput();
  input.focus();
});

promptChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.textContent.trim();
    resizeInput();
    input.focus();
  });
});

resizeInput();
checkHealth();
loadHistory();
