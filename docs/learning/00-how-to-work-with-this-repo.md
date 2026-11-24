# Working with this repo

## The pieces
- **Terminal**: the app where you type commands. On a Mac: Cmd+Space, type "Terminal".
- **git**: tracks every change to the code. A *commit* is a saved snapshot with a message.
- **GitHub**: hosts the repo online. *Push* sends your commits there.
- **Virtual environment (`.venv`)**: a private copy of Python and packages just for this project, so
  nothing clashes with other projects on your laptop.
- **GitHub Actions**: GitHub's servers run our scripts for free. `ci.yml` runs the tests on every push.
  `refresh.yml` downloads fresh data every weekday, rebuilds everything and publishes the dashboard.
- **GitHub Pages**: free hosting for the dashboard at `mustafa-waqar98.github.io/policy-shocks-term-premia`.

## Everyday commands (run inside the project folder)

```bash
cd ~/Desktop/policy-shocks-term-premia   # go to the project
make all            # download data, estimate, build the site
make serve          # look at the dashboard locally, then open http://localhost:8000
make test           # run the checks

git status                       # what changed
git add -A                       # stage all changes
git commit -m "what I changed"   # save a snapshot
git push                         # send it to GitHub
```

## Reading the code in order
1. `src/tpshock/config.py`: every setting in one place
2. `src/tpshock/data/gsw.py`: the yield curve
3. `src/tpshock/models/acm.py`: the term premium model (read the docstring at the top first)
4. `src/tpshock/shocks/identify.py`: the FOMC shocks
5. `src/tpshock/transmission/lp.py`: event regressions and local projections
6. `src/tpshock/pipeline.py`: how the steps chain together
