(() => {
  const button = document.getElementById('copy-bibtex');
  const citation = document.getElementById('bibtex-citation');
  const status = document.getElementById('bibtex-status');
  if (!button || !citation || !status) return;

  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(citation.textContent.trim());
      status.textContent = 'BibTeX copied to clipboard.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(citation);
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Citation selected. Copy it using your browser or keyboard.';
    }
  });
})();
