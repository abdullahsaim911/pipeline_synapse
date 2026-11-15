// Run with the API speech test server on 8767 and Vite on 5178. Uses an
// existing Edge installation; no browser binaries or profiles outside F:.
const path = require('node:path');
const { chromium } = require(path.resolve(__dirname, '../.runtime/tts-browser/node_modules/playwright'));

(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    headless: true,
    args: ['--autoplay-policy=no-user-gesture-required'],
  });
  try {
    const page = await browser.newPage();
    await page.goto('http://127.0.0.1:5178');
    const result = await page.evaluate(async () => {
      const { createStreamAudio, streamSpeech } = await import('/src/lib/ttsStream.js');
      const player = await createStreamAudio();
      await player.play();
      const started = performance.now();
      let firstAudio = null;
      let chunks = 0;
      let complete = false;
      let resolveEnded;
      const ended = new Promise((resolve) => { resolveEnded = resolve; });
      player.addEventListener('ended', () => resolveEnded());
      try {
        await streamSpeech('/voice/speak/stream', {
          text: `The graph shows velocity increasing over time. Sample ${Date.now() % 1000}.`,
        }, { onEvent: async (event) => {
          if (event.type === 'audio') {
            firstAudio ??= performance.now() - started;
            chunks++;
            await player.push(event.samples);
            if (chunks === 1) {
              player.pause();
              setTimeout(() => player.play(), 200);
            }
          }
          if (event.type === 'complete') { complete = true; player.finish(); }
        } });
        await Promise.race([ended, new Promise((_, reject) => setTimeout(() => reject(new Error('Playback did not drain')), 20000))]);
        if (!complete || !chunks || player.currentTime <= 0) throw new Error('No streamed playback');
        return { firstAudioMs: Math.round(firstAudio), chunks, durationSeconds: player.duration,
          playedSeconds: player.currentTime, complete };
      } finally { player.dispose(); }
    });
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
