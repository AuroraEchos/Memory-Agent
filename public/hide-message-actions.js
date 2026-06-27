const HIDDEN_BUTTON_LABELS = new Set([
  "Copy to clipboard",
  "Helpful",
  "Not helpful",
  "Edit feedback",
  "复制到剪贴板",
  "有帮助",
  "没有帮助",
  "编辑反馈",
  "查看记忆",
  "搜索记忆",
  "当前状态",
  "删除",
  "确认删除",
  "取消",
]);

function normalizeText(value) {
  return (value || "").replace(/\s+/g, " ").trim();
}

function shouldHideButton(element) {
  const candidates = [
    element.getAttribute("aria-label"),
    element.getAttribute("title"),
    element.textContent,
  ]
    .map(normalizeText)
    .filter(Boolean);

  return candidates.some((candidate) => HIDDEN_BUTTON_LABELS.has(candidate));
}

function hideMessageButtons(root = document) {
  root.querySelectorAll('button, [role="button"]').forEach((element) => {
    if (!shouldHideButton(element)) {
      return;
    }

    element.style.display = "none";

    const parent = element.parentElement;
    if (parent && parent.childElementCount === 1) {
      parent.style.display = "none";
    }
  });
}

function markMemoryAgentUi() {
  document.documentElement.dataset.memoryAgentUi = "true";
  document.body?.classList.add("memory-agent-ui");
}

function createLoginStage() {
  const stage = document.createElement("div");
  stage.className = "memory-agent-login-stage";
  stage.setAttribute("aria-hidden", "true");
  stage.innerHTML = `
    <div class="memory-agent-login-stage__halo"></div>
    <div class="memory-agent-login-stage__ring memory-agent-login-stage__ring--outer"></div>
    <div class="memory-agent-login-stage__ring memory-agent-login-stage__ring--middle"></div>
    <div class="memory-agent-login-stage__ring memory-agent-login-stage__ring--inner"></div>
    <div class="memory-agent-login-stage__core">
      <span class="memory-agent-login-stage__core-dot"></span>
    </div>
    <div class="memory-agent-login-stage__node memory-agent-login-stage__node--a"></div>
    <div class="memory-agent-login-stage__node memory-agent-login-stage__node--b"></div>
    <div class="memory-agent-login-stage__node memory-agent-login-stage__node--c"></div>
    <div class="memory-agent-login-stage__node memory-agent-login-stage__node--d"></div>
  `;
  return stage;
}

function enhanceLoginPage() {
  const layout = document.querySelector('div[class*="min-h-svh"][class*="lg:grid-cols-2"]');
  if (!(layout instanceof HTMLElement)) {
    return;
  }

  const panels = Array.from(layout.children).filter(
    (element) => element instanceof HTMLElement,
  );

  if (panels.length < 2) {
    return;
  }

  const [formPanel, visualPanel] = panels;
  if (!(formPanel instanceof HTMLElement) || !(visualPanel instanceof HTMLElement)) {
    return;
  }

  layout.classList.add("memory-agent-login-layout");
  formPanel.classList.add("memory-agent-login-panel");
  visualPanel.classList.add("memory-agent-login-visual");

  const image = visualPanel.querySelector('img[alt="Image"]');
  if (image instanceof HTMLElement) {
    image.style.display = "none";
    image.setAttribute("aria-hidden", "true");
  }

  if (!visualPanel.querySelector(".memory-agent-login-stage")) {
    visualPanel.appendChild(createLoginStage());
  }
}

const observer = new MutationObserver(() => {
  hideMessageButtons();
  enhanceLoginPage();
});

window.addEventListener("load", () => {
  markMemoryAgentUi();
  hideMessageButtons();
  enhanceLoginPage();
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["aria-label", "title"],
  });
});
