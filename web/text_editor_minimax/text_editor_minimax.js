import { app } from "/scripts/app.js";

// Matches node_id in TextEditorMinimax.define_schema() (text_editor_minimax.py)
const NODE_NAME = "TextEditorMinimax";
const WIDGET_NAME = "text"; // must match the INPUT_TYPES key

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function highlightLine(line) {
  let escaped = escapeHtml(line);
  escaped = escaped.replace(/\[([^\]]+)\]/g, '<span class="tem-keyword">[$1]</span>');
  escaped = escaped.replace(/&lt;d&gt;(.*?)&lt;\/d&gt;/g, '<span class="tem-tag">&lt;d&gt;$1&lt;/d&gt;</span>');
  escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<span class="tem-bold">**$1**</span>');
  escaped = escaped.replace(/\b(\d+:\d+\.\d+|\d+:\d+|\d+\.\d+)\b/g, '<span class="tem-time">$1</span>');
  escaped = escaped.replace(/\b(subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music|diegetic_music)\b/gi, '<span class="tem-field">$1</span>');
  escaped = escaped.replace(/&lt;((?:subject|picture|video|audio)\s*\d+)&gt;/gi, '<span class="tem-ref">&lt;$1&gt;</span>');
  escaped = escaped.replace(/\((s\d+)\)/gi, '<span class="tem-subj">($1)</span>');
  return escaped;
}

// Injected once; scoped with tem- prefixed classes so it won't collide with
// ComfyUI's own styles.
function injectStyles() {
  if (document.getElementById("text-editor-minimax-style")) return;
  const style = document.createElement("style");
  style.id = "text-editor-minimax-style";
  style.textContent = `
    .tem-wrap {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 100%;
      background: #1e1e1e;
      border: 1px solid #333;
      border-radius: 6px;
      overflow: hidden;
      font-family: "SF Mono", "Fira Code", Menlo, Consolas, monospace;
    }
    .tem-toolbar {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 6px;
      padding: 4px 6px;
      background: #2d2d2d;
      border-bottom: 1px solid #333;
      flex: 0 0 auto;
    }
    .tem-toolbar button {
      background: #3a3a3a;
      color: #d4d4d4;
      border: 1px solid #333;
      border-radius: 4px;
      padding: 2px 8px;
      font-size: 11px;
      font-family: inherit;
      cursor: pointer;
    }
    .tem-toolbar button.active {
      background: #0e639c;
      border-color: #0e639c;
    }
    .tem-font-group {
      display: flex;
      align-items: center;
      gap: 2px;
    }
    .tem-font-group button {
      padding: 2px 7px;
      line-height: 1;
    }
    .tem-font-size-label {
      color: #9d9d9d;
      font-size: 11px;
      min-width: 26px;
      text-align: center;
      user-select: none;
    }
    .tem-editor {
      position: relative;
      display: flex;
      flex: 1 1 auto;
      min-height: 0;
    }
    .tem-wrap {
      --tem-font-size: 12px;
    }
    .tem-body {
      position: relative;
      flex: 1 1 auto;
      display: flex;
      align-items: stretch;
      overflow: auto;
    }
    .tem-gutter {
      flex: 0 0 auto;
      position: sticky;
      left: 0;
      z-index: 1;
      padding: 6px 8px 6px 0;
      text-align: right;
      color: #6e7681;
      background: #252526;
      border-right: 1px solid #333;
      user-select: none;
      font-size: var(--tem-font-size);
      line-height: 1.5;
    }
    .tem-gutter div { white-space: pre; }
    .tem-text {
      position: relative;
      flex: 1 1 auto;
      min-width: 0;
    }
    .tem-highlight, .tem-input {
      margin: 0;
      padding: 6px 8px;
      font-size: var(--tem-font-size);
      line-height: 1.5;
      font-family: inherit;
      min-height: 100%;
      width: max-content;
      min-width: 100%;
      box-sizing: border-box;
    }
    .tem-highlight { color: #d4d4d4; pointer-events: none; }
    .tem-highlight > div { white-space: inherit; }
    .tem-input {
      position: absolute;
      top: 0; left: 0;
      min-height: 100%;
      background: transparent;
      color: transparent;
      caret-color: #fff;
      border: none;
      outline: none;
      resize: none;
      overflow: hidden;
    }
    .tem-body.wrap .tem-highlight, .tem-body.wrap .tem-input {
      white-space: pre-wrap;
      word-break: break-word;
      width: 100%;
    }
    .tem-body.nowrap .tem-highlight, .tem-body.nowrap .tem-input {
      white-space: pre;
    }
    .tem-keyword { color: #ec6091; font-weight: 600; }
    .tem-tag     { color: #4fc1ff; }
    .tem-bold    { color: #6bcf6b; font-weight: 600; }
    .tem-time    { color: #d19a66; }
    .tem-field   { color: #c586c0; font-weight: 600; }
    .tem-ref     { color: #4ec9b0; }
    .tem-subj    { color: #e5c07b; }
  `;
  document.head.appendChild(style);
}

function buildEditorElement(initialValue, onChange, initialFontSize, onFontSizeChange) {
  injectStyles();

  const root = document.createElement("div");
  root.className = "tem-wrap";
  root.innerHTML = `
    <div class="tem-toolbar">
      <div class="tem-font-group">
        <button class="tem-font-minus" type="button" title="Decrease font size">A-</button>
        <span class="tem-font-size-label">12px</span>
        <button class="tem-font-plus" type="button" title="Increase font size">A+</button>
      </div>
      <button class="tem-wrap-btn active" type="button">Wrap: On</button>
    </div>
    <div class="tem-editor">
      <div class="tem-body wrap">
        <div class="tem-gutter"></div>
        <div class="tem-text">
          <div class="tem-highlight"></div>
          <textarea class="tem-input" spellcheck="false"></textarea>
        </div>
      </div>
    </div>
  `;

  const gutter = root.querySelector(".tem-gutter");
  const highlight = root.querySelector(".tem-highlight");
  const input = root.querySelector(".tem-input");
  const body = root.querySelector(".tem-body");
  const wrapBtn = root.querySelector(".tem-wrap-btn");
  const fontMinusBtn = root.querySelector(".tem-font-minus");
  const fontPlusBtn = root.querySelector(".tem-font-plus");
  const fontSizeLabel = root.querySelector(".tem-font-size-label");

  let wrapEnabled = true;
  const FONT_MIN = 4;
  const FONT_MAX = 500;
  let fontSize = Math.min(FONT_MAX, Math.max(FONT_MIN, initialFontSize || 12));
  input.value = initialValue || "";

  // Renders one wrapper <div> per source line in the highlight layer so each
  // line's actual rendered height (after real word-wrapping) can be measured
  // from the DOM, instead of estimating wrapped-row counts from char widths.
  function renderHighlight() {
    const lines = input.value.split("\n");
    highlight.innerHTML = lines
      .map((line) => `<div>${highlightLine(line) || "&nbsp;"}</div>`)
      .join("");
  }

  function updateGutter() {
    const lineEls = highlight.children;
    const singleLineHeight = lineEls[0]?.getBoundingClientRect().height || 0;
    const rows = [];
    for (let i = 0; i < lineEls.length; i++) {
      rows.push(String(i + 1));
      if (wrapEnabled && singleLineHeight > 0) {
        const lineHeight = lineEls[i].getBoundingClientRect().height;
        const visualRows = Math.max(1, Math.round(lineHeight / singleLineHeight));
        for (let r = 1; r < visualRows; r++) rows.push("");
      }
    }
    gutter.innerHTML = rows.map((r) => `<div>${r || "&nbsp;"}</div>`).join("");
  }

  // Grows the textarea to its full content height so `.tem-body` is the only
  // scroll container; gutter and text then scroll together natively instead
  // of relying on mirrored scrollTop, which drifted once scrollbars appeared.
  function syncInputHeight() {
    input.style.height = "0px";
    input.style.height = `${Math.max(input.scrollHeight, highlight.offsetHeight)}px`;
  }

  function update() {
    renderHighlight();
    syncInputHeight();
    updateGutter();
    onChange?.(input.value);
  }

  function applyFontSize() {
    root.style.setProperty("--tem-font-size", `${fontSize}px`);
    fontSizeLabel.textContent = `${fontSize}px`;
    fontMinusBtn.disabled = fontSize <= FONT_MIN;
    fontPlusBtn.disabled = fontSize >= FONT_MAX;
    onFontSizeChange?.(fontSize);
    update(); // line heights depend on font size
  }

  input.addEventListener("input", update);
  // Stop ComfyUI's canvas from also handling drag/zoom while typing/scrolling.
  // Middle-click (button 1) is left alone so canvas panning still works over the textarea.
  // MIDDLE: 
  // input.addEventListener("pointerdown", (e) => {
  //   if (e.button !== 1) e.stopPropagation();
  // });
  // Scroll by a fixed number of lines per wheel click instead of the
  // browser's default (which varies by OS/pixel-vs-line delta mode).
  const SCROLL_LINES_PER_WHEEL = 3;
  input.addEventListener("wheel", (e) => {
    e.stopPropagation();
    e.preventDefault();
    const lineHeight = fontSize * 1.5;
    body.scrollTop += Math.sign(e.deltaY) * SCROLL_LINES_PER_WHEEL * lineHeight;
  });

  fontMinusBtn.addEventListener("click", () => {
    fontSize = Math.max(FONT_MIN, fontSize - 1);
    applyFontSize();
  });

  fontPlusBtn.addEventListener("click", () => {
    fontSize = Math.min(FONT_MAX, fontSize + 1);
    applyFontSize();
  });

  wrapBtn.addEventListener("click", () => {
    wrapEnabled = !wrapEnabled;
    body.classList.toggle("wrap", wrapEnabled);
    body.classList.toggle("nowrap", !wrapEnabled);
    wrapBtn.textContent = `Wrap: ${wrapEnabled ? "On" : "Off"}`;
    wrapBtn.classList.toggle("active", wrapEnabled);
    update();
  });

  const resizeObserver = new ResizeObserver(() => {
    if (wrapEnabled) {
      // Container width changed, so wrapped line heights may have too.
      syncInputHeight();
      updateGutter();
    }
  });
  resizeObserver.observe(body);

  applyFontSize();
  update();

  return {
    root,
    getValue: () => input.value,
    setValue: (v) => {
      input.value = v ?? "";
      update();
    },
  };
}

console.log("[text_editor_minimax] extension script loaded"); // confirms the JS file itself was fetched and executed

app.registerExtension({
  name: "minimax.TextEditorMinimax",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    console.log("[text_editor_minimax] starting........ 0 ");

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const node = this;

      // Some ComfyUI frontend versions finish attaching the default
      // "text" STRING widget AFTER onNodeCreated returns (e.g. on the
      // next frame), so removing it synchronously here can be a no-op
      // and leave both the default textarea and our editor on the node.
      // Deferring one frame guarantees the default widget already exists.
      requestAnimationFrame(() => {
        // LiteGraph swallows exceptions thrown inside these callbacks
        // without printing anything to console, so we catch and log
        // ourselves — otherwise a failure here is invisible.
        try {
          if (node.widgets?.some((w) => w.name === WIDGET_NAME && w.type === "TEM_EDITOR")) {
            return; // already attached (e.g. re-entrant call)
          }

          // Remove any default widget(s) ComfyUI auto-created for the
          // "text" STRING input, so we can replace them with our editor.
          let initialValue = "";
          if (node.widgets) {
            for (let i = node.widgets.length - 1; i >= 0; i--) {
              const w = node.widgets[i];
              if (w.name === WIDGET_NAME) {
                initialValue = w.value ?? initialValue;
                w.onRemove?.();
                node.widgets.splice(i, 1);
              }
            }
          }

          // Stored on node.properties so it round-trips through the saved
          // workflow JSON (like size/position), surviving a page refresh.
          node.properties = node.properties || {};
          const editor = buildEditorElement(
            initialValue,
            () => {
              node.setDirtyCanvas(true, false);
            },
            node.properties.fontSize,
            (fontSize) => {
              node.properties.fontSize = fontSize;
            }
          );
          editor.root.style.minHeight = "220px";

          if (typeof node.addDOMWidget !== "function") {
            console.error(
              "[text_editor_minimax] node.addDOMWidget is not available on this ComfyUI frontend version."
            );
            return;
          }

          const widget = node.addDOMWidget(WIDGET_NAME, "TEM_EDITOR", editor.root, {
            getValue: editor.getValue,
            setValue: editor.setValue,
            getMinHeight: () => 220,
          });
          widget.serializeValue = () => editor.getValue();

          // Only auto-size brand-new nodes. A node restored from a saved
          // workflow already has its user-chosen size applied by
          // configure() before this runs, so leave it alone — otherwise
          // every page refresh would stomp the saved size.
          if (!node.size || (node.size[0] === 0 && node.size[1] === 0)) {
            node.setSize?.(node.computeSize());
          }
        } catch (err) {
          console.error("[text_editor_minimax] failed to build custom widget:", err);
        }
      });

      return result;
    };
  },
});
