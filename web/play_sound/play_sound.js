import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Simple WebAudio beep, no external asset needed.
function playBeep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sine";
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.35);

    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.4);
    osc.onended = () => ctx.close();
  } catch (err) {
    console.warn("[Cherry] Play Sound On Finish: failed to play beep", err);
  }
}

app.registerExtension({
  name: "Cherry.PlaySoundOnFinish",
  async setup() {
    api.addEventListener("executed", ({ detail }) => {
      const node = app.graph.getNodeById(detail?.node);
      if (node?.type === "CherryPlaySoundOnFinish") {
        playBeep();
      }
    });
  },
});
