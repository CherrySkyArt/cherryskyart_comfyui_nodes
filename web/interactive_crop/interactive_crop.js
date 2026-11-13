import { app } from "/scripts/app.js"
import { api } from "/scripts/api.js"

// V3 node_id from interactive_crop.py
const NODE_TYPE = "CherryInteractiveCrop"
// ui key the Python node sends the uncropped preview under
const UI_KEY = "cherry_crop"
// Saved in node.properties so a reloaded workflow can show the last preview again
// (only works while the temp file still exists, i.e. until ComfyUI restarts)
const PREVIEW_PROP = "cherry_crop_preview"
const EDGES = ["crop_left", "crop_top", "crop_right", "crop_bottom"]

const MIN_EDITOR_HEIGHT = 220
const PAD = 4
// Cap on the canvas backing-store size so zooming way in can't allocate huge bitmaps
const MAX_BACKING_PX = 4096
// Grab distance for edges/corners, in screen pixels (independent of canvas zoom)
const HIT_TOLERANCE = 9
const HANDLE_SIZE = 7

/** Same rules as clamp_crop() in interactive_crop.py: always keep at least 1x1. */
function clampCrop(W, H, c) {
  const left = Math.max(0, Math.min(c.left, W - 1))
  const right = Math.max(0, Math.min(c.right, W - 1 - left))
  const top = Math.max(0, Math.min(c.top, H - 1))
  const bottom = Math.max(0, Math.min(c.bottom, H - 1 - top))
  return { left, top, right, bottom }
}

function getWidget(node, name) {
  return node.widgets?.find((w) => w.name === name)
}

function readCrop(node) {
  const v = (name) => Math.max(0, Math.round(Number(getWidget(node, name)?.value) || 0))
  return { left: v("crop_left"), top: v("crop_top"), right: v("crop_right"), bottom: v("crop_bottom") }
}

function writeCrop(node, c) {
  const values = { crop_left: c.left, crop_top: c.top, crop_right: c.right, crop_bottom: c.bottom }
  for (const [name, value] of Object.entries(values)) {
    const w = getWidget(node, name)
    if (w && w.value !== value) w.value = value
  }
  node.setDirtyCanvas?.(true, true)
}

function previewUrl(info) {
  const params = new URLSearchParams({
    filename: info.filename,
    subfolder: info.subfolder ?? "",
    type: info.type ?? "temp",
    rand: String(Math.random()),
  })
  return api.apiURL(`/view?${params}`)
}

/** Where the image sits inside the editor (CSS px): fit, centered horizontally, pinned to the top. */
function computeLayout(state, cw, ch) {
  const W = state.imgW
  const H = state.imgH
  const availW = Math.max(1, cw - PAD * 2)
  const availH = Math.max(1, ch - PAD * 2)
  const scale = Math.min(availW / W, availH / H)
  const dw = W * scale
  const dh = H * scale
  return { scale, ox: (cw - dw) / 2, oy: PAD, dw, dh }
}

function cropRectOnScreen(state, lay, c) {
  return {
    x0: lay.ox + c.left * lay.scale,
    y0: lay.oy + c.top * lay.scale,
    x1: lay.ox + (state.imgW - c.right) * lay.scale,
    y1: lay.oy + (state.imgH - c.bottom) * lay.scale,
  }
}

/** Which crop edges a point grabs: subset of {l,t,r,b}, "move" if inside, or null. */
function hitTest(rect, x, y, tol) {
  const inX = x >= rect.x0 - tol && x <= rect.x1 + tol
  const inY = y >= rect.y0 - tol && y <= rect.y1 + tol
  if (!inX || !inY) return null

  // When the box is tiny, prefer the edge that's actually closest
  const hit = { l: false, t: false, r: false, b: false }
  const dl = Math.abs(x - rect.x0)
  const dr = Math.abs(x - rect.x1)
  const dt = Math.abs(y - rect.y0)
  const db = Math.abs(y - rect.y1)
  if (dl <= tol || dr <= tol) dl <= dr ? (hit.l = true) : (hit.r = true)
  if (dt <= tol || db <= tol) dt <= db ? (hit.t = true) : (hit.b = true)

  if (hit.l || hit.t || hit.r || hit.b) return hit
  if (x > rect.x0 && x < rect.x1 && y > rect.y0 && y < rect.y1) return "move"
  return null
}

function cursorFor(hit) {
  if (!hit) return "default"
  if (hit === "move") return "move"
  if ((hit.l && hit.t) || (hit.r && hit.b)) return "nwse-resize"
  if ((hit.r && hit.t) || (hit.l && hit.b)) return "nesw-resize"
  if (hit.l || hit.r) return "ew-resize"
  return "ns-resize"
}

function draw(node) {
  const state = node.__cherryCrop
  if (!state) return
  const { canvas } = state
  state.lastDrawKey = cropKey(node)
  const cw = canvas.clientWidth
  const ch = canvas.clientHeight
  if (cw <= 0 || ch <= 0) return

  // The DOM widget is CSS-scaled with the graph zoom, so render at the on-screen
  // size (not the layout size) or everything gets stretched and blurry.
  const zoom = canvas.getBoundingClientRect().width / cw || 1
  state.lastZoom = zoom
  const dpr = window.devicePixelRatio || 1
  const k = Math.min(dpr * zoom, MAX_BACKING_PX / Math.max(cw, ch))
  const bw0 = Math.round(cw * k)
  const bh0 = Math.round(ch * k)
  if (canvas.width !== bw0 || canvas.height !== bh0) {
    canvas.width = bw0
    canvas.height = bh0
  }
  const ctx = canvas.getContext("2d")
  ctx.setTransform(k, 0, 0, k, 0, 0)
  ctx.clearRect(0, 0, cw, ch)
  ctx.fillStyle = "#1b1b1b"
  ctx.fillRect(0, 0, cw, ch)

  if (!state.img) {
    state.info.textContent = state.message ?? "Queue the workflow to load the image"
    return
  }

  const lay = computeLayout(state, cw, ch)
  const c = clampCrop(state.imgW, state.imgH, readCrop(node))
  const r = cropRectOnScreen(state, lay, c)

  ctx.imageSmoothingEnabled = true
  ctx.drawImage(state.img, lay.ox, lay.oy, lay.dw, lay.dh)

  // Dim everything outside the crop
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)"
  ctx.fillRect(lay.ox, lay.oy, lay.dw, r.y0 - lay.oy)
  ctx.fillRect(lay.ox, r.y1, lay.dw, lay.oy + lay.dh - r.y1)
  ctx.fillRect(lay.ox, r.y0, r.x0 - lay.ox, r.y1 - r.y0)
  ctx.fillRect(r.x1, r.y0, lay.ox + lay.dw - r.x1, r.y1 - r.y0)

  const bw = r.x1 - r.x0
  const bh = r.y1 - r.y0

  // Rule-of-thirds guides
  ctx.strokeStyle = "rgba(255, 255, 255, 0.25)"
  ctx.lineWidth = 1
  ctx.beginPath()
  for (const f of [1 / 3, 2 / 3]) {
    ctx.moveTo(r.x0 + bw * f, r.y0)
    ctx.lineTo(r.x0 + bw * f, r.y1)
    ctx.moveTo(r.x0, r.y0 + bh * f)
    ctx.lineTo(r.x1, r.y0 + bh * f)
  }
  ctx.stroke()

  // Box
  ctx.strokeStyle = "#ffffff"
  ctx.lineWidth = 1.5
  ctx.strokeRect(r.x0, r.y0, bw, bh)

  // Handles on corners and edge midpoints
  const mx = (r.x0 + r.x1) / 2
  const my = (r.y0 + r.y1) / 2
  const hs = HANDLE_SIZE
  ctx.fillStyle = "#ffffff"
  ctx.strokeStyle = "#000000"
  ctx.lineWidth = 1
  for (const [hx, hy] of [
    [r.x0, r.y0], [mx, r.y0], [r.x1, r.y0],
    [r.x0, my], [r.x1, my],
    [r.x0, r.y1], [mx, r.y1], [r.x1, r.y1],
  ]) {
    ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs)
    ctx.strokeRect(hx - hs / 2, hy - hs / 2, hs, hs)
  }

  // Size readout
  const outW = state.imgW - c.left - c.right
  const outH = state.imgH - c.top - c.bottom
  state.info.textContent = `${state.imgW} × ${state.imgH}  →  ${outW} × ${outH}   ·   x ${c.left}, y ${c.top}`
}

function cropKey(node) {
  const c = readCrop(node)
  return `${c.left},${c.top},${c.right},${c.bottom}`
}

function loadPreview(node, info) {
  const state = node.__cherryCrop
  if (!state || !info?.filename) return
  const img = new Image()
  img.onload = () => {
    state.img = img
    // Prefer the size reported by Python; fall back to the decoded image
    state.imgW = info.width || img.naturalWidth
    state.imgH = info.height || img.naturalHeight
    state.message = null
    draw(node)
  }
  img.onerror = () => {
    state.img = null
    state.message = "Preview expired - queue the workflow again"
    draw(node)
  }
  img.src = previewUrl(info)
}

/** Local (unzoomed CSS px) coordinates of a pointer event inside the editor canvas. */
function localPoint(canvas, e) {
  const rect = canvas.getBoundingClientRect()
  const sx = rect.width > 0 ? canvas.clientWidth / rect.width : 1
  const sy = rect.height > 0 ? canvas.clientHeight / rect.height : 1
  return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy, zoom: sx }
}

function setupEditor(node) {
  const container = document.createElement("div")
  container.style.cssText =
    "display:flex;flex-direction:column;width:100%;height:100%;min-height:0;overflow:hidden;background:#1b1b1b;border-radius:4px;"
  const canvasWrap = document.createElement("div")
  canvasWrap.style.cssText = "position:relative;flex:1 1 auto;min-height:0;"
  const canvas = document.createElement("canvas")
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;"
  canvasWrap.appendChild(canvas)
  // Readout is real HTML text (not canvas text) so it stays crisp at any zoom
  const info = document.createElement("div")
  info.style.cssText =
    "flex:0 0 auto;padding:2px 6px 4px;text-align:center;font:12px/16px var(--font-family, sans-serif);" +
    "font-variant-numeric:tabular-nums;color:#c8c8c8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;user-select:none;"
  container.append(canvasWrap, info)

  const state = { canvas, info, img: null, imgW: 0, imgH: 0, drag: null, message: null, lastDrawKey: "", lastZoom: 1 }
  node.__cherryCrop = state

  const widget = node.addDOMWidget("crop_editor", "cherry_crop_editor", container, {
    serialize: false,
    hideOnZoom: false,
    getValue: () => undefined,
    setValue: () => {},
    getMinHeight: () => MIN_EDITOR_HEIGHT,
  })
  widget.serialize = false

  const geometry = () => {
    const lay = computeLayout(state, canvas.clientWidth, canvas.clientHeight)
    const c = clampCrop(state.imgW, state.imgH, readCrop(node))
    return { lay, c, rect: cropRectOnScreen(state, lay, c) }
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (!state.img || e.button !== 0) return
    const p = localPoint(canvas, e)
    const { lay, c, rect } = geometry()
    const hit = hitTest(rect, p.x, p.y, HIT_TOLERANCE * p.zoom)
    if (!hit) return

    e.preventDefault()
    e.stopPropagation()
    canvas.setPointerCapture(e.pointerId)
    state.drag = { hit, startX: p.x, startY: p.y, start: c, scale: lay.scale }
  })

  canvas.addEventListener("pointermove", (e) => {
    if (!state.img) return
    const p = localPoint(canvas, e)

    if (!state.drag) {
      const { rect } = geometry()
      canvas.style.cursor = cursorFor(hitTest(rect, p.x, p.y, HIT_TOLERANCE * p.zoom))
      return
    }

    e.preventDefault()
    e.stopPropagation()
    const { hit, start, scale } = state.drag
    const W = state.imgW
    const H = state.imgH
    const dx = Math.round((p.x - state.drag.startX) / scale)
    const dy = Math.round((p.y - state.drag.startY) / scale)
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
    const next = { ...start }

    if (hit === "move") {
      const mx = clamp(dx, -start.left, start.right)
      const my = clamp(dy, -start.top, start.bottom)
      next.left = start.left + mx
      next.right = start.right - mx
      next.top = start.top + my
      next.bottom = start.bottom - my
    } else {
      if (hit.l) next.left = clamp(start.left + dx, 0, W - start.right - 1)
      if (hit.r) next.right = clamp(start.right - dx, 0, W - start.left - 1)
      if (hit.t) next.top = clamp(start.top + dy, 0, H - start.bottom - 1)
      if (hit.b) next.bottom = clamp(start.bottom - dy, 0, H - start.top - 1)
    }

    writeCrop(node, next)
    draw(node)
  })

  const endDrag = (e) => {
    if (!state.drag) return
    state.drag = null
    if (canvas.hasPointerCapture?.(e.pointerId)) canvas.releasePointerCapture(e.pointerId)
    e.stopPropagation()
  }
  canvas.addEventListener("pointerup", endDrag)
  canvas.addEventListener("pointercancel", endDrag)

  // Double-click resets to the full image
  canvas.addEventListener("dblclick", (e) => {
    if (!state.img) return
    e.preventDefault()
    e.stopPropagation()
    writeCrop(node, { left: 0, top: 0, right: 0, bottom: 0 })
    draw(node)
  })

  new ResizeObserver(() => draw(node)).observe(canvas)

  // Redraw when the numbers are typed/dragged in the normal widgets
  for (const name of EDGES) {
    const w = getWidget(node, name)
    if (!w) continue
    const cb = w.callback
    w.callback = function (...args) {
      const r = cb?.apply(this, args)
      draw(node)
      return r
    }
  }

  draw(node)
}

app.registerExtension({
  name: "Cherry.InteractiveCrop",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return

    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments)
      setupEditor(this)
      const size = this.computeSize()
      this.setSize([Math.max(this.size[0], 340), Math.max(this.size[1], size[1] + 120)])
      return result
    }

    const onExecuted = nodeType.prototype.onExecuted
    nodeType.prototype.onExecuted = function (message) {
      const result = onExecuted?.apply(this, arguments)
      const info = message?.[UI_KEY]?.[0]
      if (info) {
        this.properties ??= {}
        this.properties[PREVIEW_PROP] = info
        loadPreview(this, info)
      }
      return result
    }

    const onConfigure = nodeType.prototype.onConfigure
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments)
      const info = this.properties?.[PREVIEW_PROP]
      if (info) loadPreview(this, info)
      else draw(this)
      return result
    }

    // Catch value changes that bypass widget callbacks (undo, links, etc.)
    const onDrawForeground = nodeType.prototype.onDrawForeground
    nodeType.prototype.onDrawForeground = function () {
      const result = onDrawForeground?.apply(this, arguments)
      const state = this.__cherryCrop
      if (!state || state.drag) return result
      const cw = state.canvas.clientWidth
      const zoom = cw ? state.canvas.getBoundingClientRect().width / cw : state.lastZoom
      // Redraw on value changes, and on graph zoom so the canvas re-renders sharp
      if (state.lastDrawKey !== cropKey(this) || Math.abs(zoom - state.lastZoom) > 0.01) draw(this)
      return result
    }
  },
})
