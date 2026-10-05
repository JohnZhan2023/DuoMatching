# DuoMatching demo

[Paper (arXiv)](https://arxiv.org/abs/2610.03543) · [Project page](https://johnzhan2023.github.io/DuoMatching/) · [Models](https://huggingface.co/JohnZhan/DuoMatching)

Static supplementary video comparisons for DuoMatching, based on the September 26, 2026 submission.

The comparison gallery shows two groups per row on screens wider than 1000 px, with one baseline video and one DuoMatching video in each group. Narrower screens show one group per row. Click either video to play or pause its pair.

On narrow screens and touch devices, prompts show three lines with a button to expand the full text in the page. Prompts do not have their own scroll area on these devices, so swiping over them scrolls the page. The Causal Forcing++ examples feature the DuoMatching bamboo cat first and the wool-felt corgi third, with the lion dance moved to the end.

The Reward Forcing tab uses the ten cases from `29344_DuoMatching_Joint_Margin_Supplementary Material (5).zip`, with Ice dragon as its second example in place of Magical transformation. Its videos, prompts, and posters match the supplied archive.

The page presents the teaser, abstract, interactive method overview, and video comparisons in that order. Teaser and method figures have responsive WebP previews at 1400 and 2800 px wide, with original PDFs in `assets/figures/`. Clicking the Method figure expands the left panel to fill the figure and starts the 1080p idea animation in `assets/videos/`. Use the back button or Escape to return to the method overview; the PDF link remains available. Expansion respects reduced-motion preferences.

## Local preview

```sh
python3 -m http.server 8000
```

Open http://localhost:8000/ in a browser.

## GitHub Pages

Publish the `demo` branch from `/ (root)` using GitHub Pages (Settings → Pages → Deploy from a branch). The `.nojekyll` file serves these static files directly.

Website: https://johnzhan2023.github.io/DuoMatching/
