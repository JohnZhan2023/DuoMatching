(() => {
  const stage = document.getElementById('method-stage');
  const preview = document.getElementById('method-preview');
  const player = document.getElementById('method-player');
  const video = document.getElementById('method-animation');
  const toolbar = document.getElementById('method-animation-toolbar');
  const back = document.getElementById('method-animation-back');
  const status = document.getElementById('method-animation-status');
  let expanded = false;
  let version = 0;

  function close() {
    expanded = false;
    version++;
    video.pause();
    if (video.readyState > 0) video.currentTime = 0;
    stage.classList.remove('is-expanded');
    player.inert = true;
    player.setAttribute('aria-hidden', 'true');
    preview.inert = false;
    preview.setAttribute('aria-expanded', 'false');
    toolbar.hidden = true;
    status.textContent = '';
    preview.focus({ preventScroll: true });
  }

  preview.addEventListener('click', async () => {
    if (expanded) return;
    expanded = true;
    const token = ++version;
    player.inert = false;
    player.removeAttribute('aria-hidden');
    preview.setAttribute('aria-expanded', 'true');
    stage.classList.add('is-expanded');
    toolbar.hidden = false;
    back.focus({ preventScroll: true });
    preview.inert = true;
    status.textContent = '';
    if (video.readyState > 0) video.currentTime = 0;
    try {
      // Start inside the user gesture, without waiting for the expansion.
      await video.play();
      if (token !== version && !expanded) video.pause();
    } catch (error) {
      if (token === version && expanded) {
        status.textContent = 'Use the video controls to start the animation, or return to the method figure.';
      }
    }
  });

  back.addEventListener('click', close);
  stage.parentElement.addEventListener('keydown', event => {
    if (event.key === 'Escape' && expanded && !document.fullscreenElement) {
      event.preventDefault();
      close();
    }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) video.pause();
  });
})();
