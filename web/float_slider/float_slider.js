import { app } from "/scripts/app.js";

// Value always keeps at least this many decimals, even when stepping by 1
const MIN_PRECISION = 2

// Push the chosen step into the number widget's options.
// The frontend reads step2 as the real step (step is the legacy 10x value).
// Step only controls how far arrows/drag move the value; rounding is decoupled
// so a step of 1 turns 0.53 into 1.53 instead of snapping to whole numbers.
function applyStep(node, snapValue) {
  const valueWidget = node.widgets?.find((w) => w.name === "value")
  const stepWidget = node.widgets?.find((w) => w.name === "step")
  if (!valueWidget || !stepWidget) return

  const step = Number(stepWidget.value)
  if (!(step > 0)) return
  const precision = Math.max(MIN_PRECISION, -Math.floor(Math.log10(step)))
  const round = 10 ** -precision

  valueWidget.options.step2 = step
  valueWidget.options.step = step * 10
  valueWidget.options.round = round
  valueWidget.options.precision = precision

  if (snapValue && typeof valueWidget.value === "number") {
    const snapped = Math.round(valueWidget.value / round) * round
    valueWidget.value = Number(snapped.toFixed(precision))
  }

  node.setDirtyCanvas?.(true, true)
}

app.registerExtension({
  name: "Cherry.FloatSlider",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CherryFloatSlider") return

    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments)

      const stepWidget = this.widgets?.find((w) => w.name === "step")
      if (stepWidget) {
        const cb = stepWidget.callback
        stepWidget.callback = (...args) => {
          const r = cb?.apply(stepWidget, args)
          applyStep(this, true)
          return r
        }
      }
      applyStep(this, false)

      return result
    }

    // Re-apply after a saved workflow restores the dropdown value
    const onConfigure = nodeType.prototype.onConfigure
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments)
      applyStep(this, false)
      return result
    }
  },
})
