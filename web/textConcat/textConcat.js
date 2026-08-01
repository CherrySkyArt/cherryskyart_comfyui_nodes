import { app } from "/scripts/app.js";

const NODE_ID = "TextConcat_Cherry";
const INPUT_PREFIX = "text_";

app.registerExtension({
  name: "Cherry.TextConcat",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_ID) return;

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, _index, connected, _link_info) {
      const result = onConnectionsChange?.apply(this, arguments);

      if (type !== 1 /* LiteGraph.INPUT */) return result;

      // Renumber all text_N inputs sequentially (they all live before separator)
      let slot_i = 1;
      for (let i = 0; i < this.inputs.length; i++) {
        if (this.inputs[i].name.startsWith(INPUT_PREFIX)) {
          this.inputs[i].name = `${INPUT_PREFIX}${slot_i}`;
          slot_i++;
        }
      }

      if (connected) {
        // If the last text slot (just before separator) is now connected, insert a new empty one before separator
        const insertBefore = this.inputs.findIndex(i => i.name === "separator");
        const lastTextIndex = insertBefore - 1;
        if (lastTextIndex >= 0 && this.inputs[lastTextIndex].name.startsWith(INPUT_PREFIX)) {
          const lastText = this.inputs[lastTextIndex];
          if (lastText.link !== null && lastText.link !== undefined) {
            // addInput appends at end; we'll add then move it before separator
            this.addInput(`${INPUT_PREFIX}${slot_i}`, "STRING");
            // Move the newly added slot (currently at end) to just before separator
            const newIndex = this.inputs.length - 1;
            const targetIndex = this.inputs.findIndex(i => i.name === "separator");
            if (newIndex !== targetIndex) {
              const [moved] = this.inputs.splice(newIndex, 1);
              this.inputs.splice(targetIndex, 0, moved);
            }
          }
        }
      } else {
        // Remove trailing disconnected text slots (keep at least text_1)
        // "trailing" means the slot just before separator
        let sepIdx = this.inputs.findIndex(i => i.name === "separator");
        while (sepIdx > 1) {
          const last = this.inputs[sepIdx - 1];
          const secondLast = this.inputs[sepIdx - 2];
          if (
            last.name.startsWith(INPUT_PREFIX) &&
            (last.link === null || last.link === undefined) &&
            secondLast.name.startsWith(INPUT_PREFIX) &&
            (secondLast.link === null || secondLast.link === undefined)
          ) {
            this.removeInput(sepIdx - 1);
            sepIdx--;
          } else {
            break;
          }
        }
      }

      this.setDirtyCanvas(true, true);
      return result;
    };
  },
});
