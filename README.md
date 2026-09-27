# DuoMatching demo

Static supplementary video comparisons for DuoMatching, based on the September 26, 2026 submission.

The comparison gallery shows two groups per row on screens wider than 1000 px, with one baseline video and one DuoMatching video in each group. Narrower screens show one group per row. Click either video to play or pause its pair.

The page presents the teaser, idea animation, abstract, method overview, and video comparisons in that order. Teaser and method figures have responsive WebP previews at 1400 and 2800 px wide; click either figure to open its original PDF in `assets/figures/`. The 1080p idea animation in `assets/videos/` uses native playback controls and appears immediately before the abstract.

## Local preview

```sh
python3 -m http.server 8000
```

Open http://localhost:8000/ in a browser.

## GitHub Pages

Publish the `demo` branch from `/ (root)` using GitHub Pages (Settings → Pages → Deploy from a branch). The `.nojekyll` file serves these static files directly.

Website: https://johnzhan2023.github.io/DuoMatching/
