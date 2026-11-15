# Publishing checklist

```bash
# 1. New repo: platform
cd rec-eval-platform
git init && git add . && git commit -m "Add recommendation evaluation platform"
git branch -M main
git remote add origin https://github.com/<you>/rec-eval-platform.git
git push -u origin main
```

CineXplain: open your repo, copy `docs/CineXplain_README.md` over `README.md`, fill every `<!-- -->` section with your real details, set repo visibility to **Public** (Settings → Danger Zone → Change visibility).

ai-agent: rename the repo, copy `docs/ai-agent_README.md` over `README.md`, fill in, add screenshots.

Also: pin the three repos on your GitHub profile, and add a one-line description and topics (`recommender-system`, `machine-learning`, `fastapi`) to each.
