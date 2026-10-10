const form = document.querySelector('#pipeline-form');
const videoInput = document.querySelector('#video');
const detectionsInput = document.querySelector('#detections');
const truthInput = document.querySelector('#truth');
const visualiseInput = document.querySelector('#visualise-detections');
const metricsInput = document.querySelector('#run-metrics');
const confField = document.querySelector('#conf-field');
const confInput = document.querySelector('#conf');
const visualiseField = document.querySelector('#visualise-field');
const metricsField = document.querySelector('#metrics-field');
const visualiseTracksInput = document.querySelector('#visualise-tracks');
const validation = document.querySelector('#source-validation');
const outputPanel = document.querySelector('#output-panel');
const log = document.querySelector('#log');
const files = document.querySelector('#files');
const status = document.querySelector('#status');
const statusDot = document.querySelector('#status-dot');
const runButton = document.querySelector('#run-button');
const resetButton = document.querySelector('#reset-button');
const cancelButton = document.querySelector('#cancel-button');
const shutdownButton = document.querySelector('#shutdown-button');
let activeJobId = null;

function setName(input, target) {
  input.addEventListener('change', () => {
    document.querySelector(target).textContent = input.files[0]?.name || (target === '#truth-name' ? 'Optional' : 'Drop a file here or browse');
    updateAvailability();
  });
}
setName(videoInput, '#video-name');
setName(detectionsInput, '#detections-name');
setName(truthInput, '#truth-name');

function updateAvailability() {
  const hasVideo = videoInput.files.length > 0;
  const hasDetections = detectionsInput.files.length > 0;
  const hasTruth = truthInput.files.length > 0;
  const detectionsMode = hasDetections;
  confInput.disabled = detectionsMode;
  confField.style.opacity = detectionsMode ? '.42' : '1';
  visualiseInput.disabled = !hasVideo;
  visualiseField.style.opacity = hasVideo ? '1' : '.42';
  visualiseTracksInput.disabled = !hasVideo;
  visualiseTracksInput.closest('.switch-card').style.opacity = hasVideo ? '1' : '.42';
  metricsInput.disabled = !hasTruth;
  metricsField.style.opacity = hasTruth ? '1' : '.42';
  if (!hasTruth) metricsInput.checked = false;
  if (!hasVideo) visualiseInput.checked = false;
}

for (const zone of document.querySelectorAll('.dropzone')) {
  zone.addEventListener('dragover', (event) => { event.preventDefault(); zone.classList.add('dragging'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', (event) => {
    event.preventDefault(); zone.classList.remove('dragging');
    const input = zone.querySelector('input');
    if (event.dataTransfer.files.length) {
      const transfer = new DataTransfer();
      transfer.items.add(event.dataTransfer.files[0]);
      input.files = transfer.files;
      input.dispatchEvent(new Event('change'));
    }
  });
}
updateAvailability();

function validFloat(input, label) {
  const value = Number(input.value);
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error(`${label} must be between 0.00 and 1.00.`);
  return value.toFixed(2);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  validation.textContent = '';
  const hasVideo = videoInput.files.length > 0;
  const hasDetections = detectionsInput.files.length > 0;
  const hasTruth = truthInput.files.length > 0;
  if (!hasVideo && !hasDetections) { validation.textContent = 'Add a video or a detection CSV to begin.'; return; }
  if (metricsInput.checked && !hasTruth) { validation.textContent = 'Ground truth XML is required when metrics are enabled.'; return; }
  try {
    const conf = validFloat(confInput, 'Confidence');
    const iou = validFloat(document.querySelector('#iou'), 'IoU');
    const data = new FormData(form);
    data.set('conf', conf); data.set('iou', iou);
    outputPanel.hidden = false; log.textContent = ''; files.innerHTML = '';
    runButton.disabled = true; cancelButton.disabled = false; status.textContent = 'Pipeline running'; statusDot.className = 'status-dot busy';
    resetButton.disabled = true;
    const response = await fetch('/run', { method: 'POST', body: data });
    if (!response.ok) {
      const details = await response.text();
      throw new Error(`Request failed (${response.status}): ${details}`);
    }
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
    let receivedFinishedEvent = false;
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const lines = buffer.split('\n'); buffer = lines.pop();
      for (const line of lines) {
        if (!line) continue;
        const message = JSON.parse(line);
        if (message.event === 'started') { activeJobId = message.job_id; }
        if (message.event === 'output') { log.textContent += message.text; log.scrollTop = log.scrollHeight; }
        if (message.event === 'finished') {
          receivedFinishedEvent = true;
          const ok = message.status === 'complete';
          activeJobId = null; cancelButton.disabled = true;
          status.textContent = ok ? 'Pipeline complete' : (message.status === 'cancelled' ? 'Run cancelled' : 'Pipeline failed');
          statusDot.className = `status-dot ${ok ? 'done' : (message.status === 'failed' ? 'error' : 'busy')}`;
          if (typeof message.output === 'string') log.textContent = message.output;
          if (!ok && message.status === 'failed' && !message.output) {
            log.textContent += `Pipeline failed${message.return_code == null ? '.' : ` (exit code ${message.return_code}).`}\n`;
          }
          log.scrollTop = log.scrollHeight;
          for (const file of message.files || []) {
            const link = document.createElement('a'); link.href = `/download/${message.job_id}/${encodeURIComponent(file)}`; link.textContent = `Download ${file}`; link.download = ''; files.append(link);
          }
        }
      }
    }
    if (!receivedFinishedEvent) {
      throw new Error('The server closed the run stream before sending a completion status.');
    }
  } catch (error) {
    log.textContent += `\nERROR: ${error.message || 'The pipeline could not be started.'}\n`;
    log.scrollTop = log.scrollHeight;
    activeJobId = null;
    status.textContent = 'Run failed'; statusDot.className = 'status-dot error';
  } finally { runButton.disabled = false; resetButton.disabled = false; cancelButton.disabled = true; }
});

resetButton.addEventListener('click', () => {
  if (activeJobId) return;
  form.reset();
  for (const [input, target, empty] of [
    [videoInput, '#video-name', 'Drop a video here or browse'],
    [detectionsInput, '#detections-name', 'Drop detections here or browse'],
    [truthInput, '#truth-name', 'Optional'],
  ]) {
    input.value = '';
    document.querySelector(target).textContent = empty;
  }
  activeJobId = null;
  validation.textContent = '';
  outputPanel.hidden = true;
  log.textContent = '';
  files.innerHTML = '';
  status.textContent = 'Ready to run';
  statusDot.className = 'status-dot';
  runButton.disabled = false;
  cancelButton.disabled = true;
  resetButton.disabled = false;
  updateAvailability();
});

cancelButton.addEventListener('click', async () => {
  if (!activeJobId) return;
  cancelButton.disabled = true;
  status.textContent = 'Cancelling run';
  try {
    const response = await fetch('/cancel', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({job_id: activeJobId}),
    });
    if (!response.ok) throw new Error(await response.text());
  } catch (error) {
    validation.textContent = error.message || 'The run could not be cancelled.';
    cancelButton.disabled = false;
  }
});

shutdownButton.addEventListener('click', async () => {
  if (!window.confirm('Stop the web server and any active pipeline run?')) return;
  shutdownButton.disabled = true;
  cancelButton.disabled = true;
  status.textContent = 'Shutting down server';
  try {
    const response = await fetch('/shutdown', {method: 'POST'});
    if (!response.ok) throw new Error(await response.text());
    statusDot.className = 'status-dot done';
    validation.textContent = 'Server stopped. Restart the Python wrapper to use the page again.';
  } catch (error) {
    validation.textContent = error.message || 'The server could not be stopped.';
    shutdownButton.disabled = false;
  }
});
