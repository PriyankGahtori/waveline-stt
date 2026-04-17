(async () => {
  const statusEl = document.getElementById('status');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    stream.getTracks().forEach(t => t.stop());
    statusEl.textContent = '✓ Permission granted — you can close this tab.';
    statusEl.style.color = '#059669';
    chrome.runtime.sendMessage({ type: 'MIC_PERMISSION_RESULT', granted: true });
  } catch (e) {
    statusEl.textContent = 'Permission denied. Enable microphone in Chrome settings and try again.';
    statusEl.style.color = '#dc2626';
    chrome.runtime.sendMessage({ type: 'MIC_PERMISSION_RESULT', granted: false, reason: e.name });
  }
})();
