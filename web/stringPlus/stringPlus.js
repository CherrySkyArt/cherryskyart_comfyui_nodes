import { app } from "/scripts/app.js";

// Inlined from ComfyUI_frontend src/utils/formatUtil.ts
const dateParts = { d: (d) => d.getDate(), M: (d) => d.getMonth() + 1, h: (d) => d.getHours(), m: (d) => d.getMinutes(), s: (d) => d.getSeconds() }
const dateFormat = new RegExp(Object.keys(dateParts).map((k) => k + k + "?").join("|") + "|yyy?y?", "g")

function formatDate(text, date) {
  return text.replace(dateFormat, (token) => {
    if (token === "yy") return (date.getFullYear() + "").substring(2)
    if (token === "yyyy") return date.getFullYear().toString()
    if (token[0] in dateParts) return (dateParts[token[0]](date) + "").padStart(token.length, "0")
    return token
  })
}

// Inlined from ComfyUI_frontend src/utils/searchAndReplace.ts
function applyTextReplacements(graph, value) {
  // collect nodes from graph, handling subgraphs
  const allNodes = []
  const stack = [graph]
  while (stack.length) {
    const g = stack.pop()
    for (const node of (g._nodes ?? g.nodes ?? [])) {
      allNodes.push(node)
      if (node.subgraph) stack.push(node.subgraph)
    }
  }

  return value.replace(/%([^%]+)%/g, (match, text) => {
    const split = text.split(".")
    if (split.length !== 2) {
      if (split[0].startsWith("date:")) return formatDate(split[0].substring(5), new Date())
      return match
    }

    let nodes = allNodes.filter((n) => n.properties?.["Node name for S&R"] === split[0])
    if (!nodes.length) nodes = allNodes.filter((n) => n.title === split[0])
    if (!nodes.length) return match

    // mode 4 = bypassed in LiteGraph; resolve bypassed nodes to empty string
    if (nodes[0].mode === 4) return ""

    const widget = nodes[0].widgets?.find((w) => w.name === split[1])
    if (!widget) return match

    return ((widget.value ?? "") + "").replaceAll(/[/?<>\\:*|"\x00-\x1F\x7F]/g, "_")
  })
}

app.registerExtension({
  name: "Cherry.StringTemplate",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CherryStringTemplate") return

    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments)

      const widget = this.widgets?.find((w) => w.name === "template")
      if (widget) {
        widget.serializeValue = () => applyTextReplacements(app.graph, widget.value)
      }

      return result
    }
  },
})
