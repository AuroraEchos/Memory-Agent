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

const observer = new MutationObserver(() => hideMessageButtons());

window.addEventListener("load", () => {
  hideMessageButtons();
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["aria-label", "title"],
  });
});
