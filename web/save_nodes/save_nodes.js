import { app } from "/scripts/app.js";

// ComfyUI only expands %date:...% / %Node.widget% in filename_prefix for a hardcoded
// list of core save nodes (Comfy.SaveImageExtraOutput). Do the same for ours.
const SAVE_NODES = new Set(["SaveImage_Cherry", "SaveVideo_Cherry"]);

app.registerExtension({
  name: "Cherry.SaveNodesFilenamePrefix",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!SAVE_NODES.has(nodeData.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const widget = this.widgets?.find((w) => w.name === "filename_prefix");
      const applyTextReplacements = window.comfyAPI?.utils?.applyTextReplacements;
      if (widget && applyTextReplacements) {
        widget.serializeValue = () => applyTextReplacements(app, widget.value);
      }
      return r;
    };
  },
});
