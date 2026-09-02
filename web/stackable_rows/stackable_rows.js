import { app } from "/scripts/app.js"

// V3 node_id, becomes the litegraph node type name.
const NODE_TYPE = "CherryStackableRows"
// Autogrow input id ("rows") + TemplatePrefix prefix ("row_") from stackable_rows.py.
const GROUP_NAME = "rows"
const ROW_PREFIX = "row_"
const MAX_ROWS = 100
const ROW_RE = new RegExp(`^${GROUP_NAME}\\.${ROW_PREFIX}(\\d+)$`)
// Reserved strip on the right of each row where the drag handle icon lives.
const HANDLE_WIDTH = 20

function rowOrdinal(name) {
  const m = ROW_RE.exec(name)
  return m ? parseInt(m[1], 10) : -1
}

function rowInputs(node) {
  return (node.inputs ?? []).filter((i) => ROW_RE.test(i.name)).sort((a, b) => rowOrdinal(a.name) - rowOrdinal(b.name))
}

function rowWidgets(node) {
  return (node.widgets ?? []).filter((w) => ROW_RE.test(w.name))
}

/**
 * StackableRows' row_N Autogrow inputs are created socket-only in this
 * ComfyUI build — no widget gets attached (verified live: node.inputs has
 * rows.row_N with widget:undefined, and node.widgets has no matching
 * entry). So we own widget creation ourselves: for every row input lacking
 * a widget, create a plain string widget and bind it to the socket the
 * same way addInputWidget does (input.widget = {name}), which is what
 * LGraphNode.getSlotFromWidget/isWidgetInputSlot use to align the socket
 * to its widget's row and skip it from normal socket layout.
 */
function ensureRowWidget(node, input) {
  const existing = node.widgets?.find((w) => w.name === input.name)
  if (existing) return existing

  const widget = node.addWidget("string", input.name, "", () => {}, { multiline: false })
  widget.label = `row ${rowOrdinal(input.name)}`
  input.widget = { name: input.name }
  attachDragReorder(node, widget)
  return widget
}

/** Ensures every existing row_N socket has a bound widget, in row-number order. */
function ensureAllRowWidgets(node) {
  for (const input of rowInputs(node)) {
    ensureRowWidget(node, input)
  }
  reorderRowWidgets(node, rowWidgets(node).sort((a, b) => rowOrdinal(a.name) - rowOrdinal(b.name)))
}

/**
 * Adds one more row_N input+widget, the same way a real connection would
 * grow Autogrow — used by our own "+ Add row" button so rows can be added
 * by typing alone, without wiring a node into the last socket first.
 */
function addRow(node) {
  const existing = rowInputs(node)
  const nextOrdinal = existing.length ? rowOrdinal(existing[existing.length - 1].name) + 1 : 0
  if (nextOrdinal >= MAX_ROWS) return null

  const name = `${GROUP_NAME}.${ROW_PREFIX}${nextOrdinal}`
  const input = node.addInput(name, "STRING", {})
  const widget = ensureRowWidget(node, input)

  syncRowOrder(node)
  node.size[1] = node.computeSize([...node.size])[1]
  node.setDirtyCanvas(true, true)
  return widget
}

/** True if a node-relative x is within the drag-handle strip reserved on the right of a row. */
function inHandleZone(x, nodeWidth) {
  return x >= nodeWidth - 15 - HANDLE_WIDTH && x <= nodeWidth - 15
}

/**
 * Adds a drag handle to a StackableRows row widget: a small grip icon in a
 * reserved strip on the right, draggable to reorder. Everywhere else on the
 * row keeps its normal behavior (click opens the text-edit prompt).
 *
 * litegraph hit-tests a widget by full row width (no native sub-region
 * support — see LGraphNode.getWidgetOnPos), so the split has to happen
 * ourselves: onPointerDown is checked before litegraph wires up its own
 * click/drag handlers (LGraphCanvas.processWidgetClick), so returning true
 * there when the press starts in the handle zone lets us take over the
 * drag directly via the pointer, and returning false/undefined elsewhere
 * leaves the normal click-to-edit path untouched.
 */
function attachDragReorder(node, widget) {
  widget.onPointerDown = (pointer, ownerNode, canvas) => {
    const nodeWidth = ownerNode.size[0]
    const x = pointer.eDown.canvasX - ownerNode.pos[0]
    if (!inHandleZone(x, nodeWidth)) {
      // getWidgetOnPos is overridden below to include disabled (connected)
      // rows so the handle stays draggable even when a wire is plugged in.
      // A connected row's value comes from the wire, so a click outside the
      // handle shouldn't open the text-edit prompt — swallow it here.
      return !!widget.computedDisabled
    }

    canvas.canvas.style.cursor = "grabbing"
    pointer.onDragEnd = () => {
      canvas.canvas.style.cursor = ""
    }

    pointer.onDrag = (eMove) => {
      const rows = rowWidgets(ownerNode)
      const pointerY = eMove.canvasY - ownerNode.pos[1]

      let targetIndex = rows.length - 1
      for (let i = 0; i < rows.length; i++) {
        const w = rows[i]
        const top = w.last_y ?? w.y ?? 0
        const height = w.computedHeight ?? window.LiteGraph.NODE_WIDGET_HEIGHT
        if (pointerY < top + height / 2) {
          targetIndex = i
          break
        }
      }

      const srcIndex = rows.indexOf(widget)
      if (srcIndex === -1 || srcIndex === targetIndex) return

      rows.splice(srcIndex, 1)
      rows.splice(targetIndex, 0, widget)
      reorderRowWidgets(ownerNode, rows)
      syncRowOrder(ownerNode)
      canvas.setDirty(true, true)
    }
    return true
  }

  // TextWidget.drawWidget hardcodes rightPadding to 0 internally (no way to
  // pass it through options), so long values can render under the handle
  // zone. Painting a background patch behind the icon keeps it legible
  // without needing to reimplement TextWidget's text-truncation logic.
  const prevDrawWidget = widget.drawWidget?.bind(widget)
  widget.drawWidget = function (ctx, options) {
    prevDrawWidget(ctx, options)

    const { width } = options
    const { y, height } = this
    const zoneLeft = width - 15 - HANDLE_WIDTH
    const zoneRight = width - 15
    const cx = (zoneLeft + zoneRight) / 2
    const cy = y + height / 2

    ctx.save()
    ctx.fillStyle = this.background_color
    ctx.fillRect(zoneLeft, y + 1, HANDLE_WIDTH, height - 2)

    ctx.fillStyle = this.secondary_text_color
    for (const dy of [-4, 0, 4]) {
      ctx.beginPath()
      ctx.arc(cx - 3, cy + dy, 1.3, 0, Math.PI * 2)
      ctx.arc(cx + 3, cy + dy, 1.3, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.restore()
  }
}

/** Splices `node.widgets` so row widgets appear in `orderedRows` order; other widgets keep their relative order. */
function reorderRowWidgets(node, orderedRows) {
  const firstRowIdx = node.widgets.findIndex((w) => ROW_RE.test(w.name))
  const before = node.widgets.slice(0, firstRowIdx === -1 ? node.widgets.length : firstRowIdx).filter((w) => !ROW_RE.test(w.name))
  const rest = node.widgets.filter((w) => !ROW_RE.test(w.name))
  const after = rest.slice(before.length)
  node.widgets = [...before, ...orderedRows, ...after]
}

/** Writes the current top-to-bottom row name order into the hidden row_order widget. */
function syncRowOrder(node) {
  const orderWidget = node.widgets?.find((w) => w.name === "row_order")
  if (!orderWidget) return
  const names = rowWidgets(node).map((w) => `${ROW_PREFIX}${rowOrdinal(w.name)}`)
  orderWidget.value = JSON.stringify(names)
}

/** Restores widget order from a saved row_order (e.g. after loading a workflow). */
function applySavedOrder(node) {
  const orderWidget = node.widgets?.find((w) => w.name === "row_order")
  if (!orderWidget?.value) return
  let savedOrder
  try {
    savedOrder = JSON.parse(orderWidget.value)
  } catch {
    return
  }
  if (!Array.isArray(savedOrder) || savedOrder.length < 2) return

  const byShortName = new Map(rowWidgets(node).map((w) => [`${ROW_PREFIX}${rowOrdinal(w.name)}`, w]))
  const ordered = savedOrder.map((short) => byShortName.get(short)).filter(Boolean)
  for (const w of byShortName.values()) {
    if (!ordered.includes(w)) ordered.push(w)
  }
  if (ordered.length) reorderRowWidgets(node, ordered)
}

app.registerExtension({
  name: "cherryFrontend.StackableRows",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return

    // getWidgetOnPos skips disabled widgets by default (a row becomes
    // "disabled" once its socket has a link — see
    // LGraphNode.updateComputedDisabled), which made connected rows
    // unhittable entirely, including their drag handle. Row widgets should
    // stay reorderable regardless of connection state, so always include
    // them; onPointerDown above still blocks clicks outside the handle on
    // a connected row so its (wire-driven) value can't be edited.
    const getWidgetOnPos = nodeType.prototype.getWidgetOnPos
    nodeType.prototype.getWidgetOnPos = function (canvasX, canvasY, includeDisabled) {
      const rowsInclusive = getWidgetOnPos.call(this, canvasX, canvasY, true)
      if (rowsInclusive && ROW_RE.test(rowsInclusive.name)) return rowsInclusive
      return getWidgetOnPos.call(this, canvasX, canvasY, includeDisabled)
    }

    // Show a grab cursor while hovering the handle zone. LGraphCanvas only
    // recomputes canvas.style.cursor from its own hoveringOver/dragging_canvas/
    // read_only state (see _updateCursorStyle), so setting it here directly
    // during a plain hover move doesn't get immediately overwritten.
    const onMouseMove = nodeType.prototype.onMouseMove
    nodeType.prototype.onMouseMove = function (e, pos, canvas) {
      onMouseMove?.apply(this, arguments)
      const widget = this.getWidgetOnPos(e.canvasX, e.canvasY, true)
      const overHandle = widget && ROW_RE.test(widget.name) && inHandleZone(pos[0], this.size[0])
      if (canvas?.canvas) canvas.canvas.style.cursor = overHandle ? "grab" : ""
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments)
      const node = this

      const orderWidget = node.widgets?.find((w) => w.name === "row_order")
      if (orderWidget) {
        orderWidget.computeSize = () => [0, -4]
        orderWidget.hidden = true
      }

      ensureAllRowWidgets(node)
      applySavedOrder(node)

      node.addWidget("button", "+ Add row", null, () => addRow(node))

      node.size[1] = node.computeSize([...node.size])[1]
      node.setDirtyCanvas(true, true)
    }

    // litegraph's own configure() restores widget VALUES positionally against
    // this.widgets, before onConfigure ever runs (see LGraphNode.configure:
    // widgets_values are assigned by walking this.widgets in array order).
    // Our row widgets don't exist yet at that point unless we create them
    // first — otherwise values land on the wrong widget, or get dropped for
    // any row created too late to receive its slot. So we pre-create widgets
    // for every rows.row_N socket in the incoming save data *before* calling
    // the original configure(), using the same order the sockets are saved
    // in, matching how they'll be counted for the positional value restore.
    const configure = nodeType.prototype.configure
    nodeType.prototype.configure = function (info) {
      const node = this
      const savedRowInputs = (info?.inputs ?? [])
        .filter((i) => ROW_RE.test(i.name))
        .sort((a, b) => rowOrdinal(a.name) - rowOrdinal(b.name))

      node.widgets ??= []
      for (const savedInput of savedRowInputs) {
        if (node.widgets.some((w) => w.name === savedInput.name)) continue
        const widget = node.addWidget("string", savedInput.name, "", () => {}, { multiline: false })
        widget.label = `row ${rowOrdinal(savedInput.name)}`
      }
      if (savedRowInputs.length) {
        reorderRowWidgets(node, rowWidgets(node).sort((a, b) => rowOrdinal(a.name) - rowOrdinal(b.name)))
      }

      configure?.apply(node, arguments)

      // Bind sockets to their widgets and wire up drag now that both exist.
      for (const input of rowInputs(node)) {
        input.widget = { name: input.name }
        const widget = node.widgets.find((w) => w.name === input.name)
        if (widget && !widget.onPointerDown) attachDragReorder(node, widget)
      }
      applySavedOrder(node)
      node.setDirtyCanvas(true, true)
    }
  },
})
