# Deployment Guide

How to take this project from your laptop to a public website. The main route is **GitHub + Render**. PythonAnywhere is covered as an alternative.

## 1. How the app works (read this first)

```
Browser  ──►  gunicorn  ──►  Flask (app.py)  ──►  model/heart_model.joblib
 (form)        (server)      validates input        (scikit-learn Pipeline)
```

- There is no database. The model is a single file loaded when the server starts.
- The page sends the 13 values to `/api/predict` and shows the probability that comes back.
- Hosting means: a machine that installs `requirements.txt`, runs `gunicorn app:app`, and has the `model/` folder next to the code.

**The model file must be in your repository.** The hosting platform only has what you push to GitHub, so `model/heart_model.joblib` and `model/model_metadata.json` must be committed. (`.gitignore` already allows them.)

## 2. Prerequisites

| Need | Notes |
|---|---|
| Python 3.12 | The pinned `numpy==1.26.4` does not support newer Python releases. Check with `python --version`. |
| Git | `git --version` |
| GitHub account | To host the code |
| Render account | Sign up at render.com with your GitHub login |

## 3. Run it locally first

**Windows (PowerShell)**

```powershell
cd path\to\project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt
python train_model.py --no-show
python app.py
```

**macOS / Linux**

```bash
cd path/to/project
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
python train_model.py --no-show
python app.py
```

Open http://127.0.0.1:5000, click **Fill a higher-risk example**, then **Estimate risk**. If a result appears, the app works.

> If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

Training creates `model/heart_model.joblib`. **Train in the same environment you will deploy with** (Python 3.12 and the versions in `requirements.txt`). A model saved with one scikit-learn version can fail or misbehave when loaded with another.

## 4. Pre-deployment checklist

- [ ] `model/heart_model.joblib` and `model/model_metadata.json` exist
- [ ] `python app.py` works locally
- [ ] `requirements.txt` lists exactly the versions you trained with: run `pip freeze | findstr -i "scikit-learn numpy pandas joblib"` (Windows) or `pip freeze | grep -iE "scikit-learn|numpy|pandas|joblib"` (macOS/Linux) and compare
- [ ] No secrets in the code (there are none by default)
- [ ] You have read the disclaimer text in `templates/index.html` and are happy with it. Keep it.

Note: gunicorn does not run on Windows. That is fine: use `python app.py` locally, and Render runs gunicorn on Linux.

## 5. Push the project to GitHub

1. On github.com click **New repository**. Suggested name: `cardiovascular-disease-risk-assessment`. Leave it empty (no README, no .gitignore), because the project already has both.
2. In the project folder:

```bash
git init
git add .
git commit -m "Cardiovascular Disease Risk Assessment & Prediction Model"
git branch -M main
git remote add origin https://github.com/<your-username>/cardiovascular-disease-risk-assessment.git
git push -u origin main
```

3. Refresh the repository page and confirm that `model/heart_model.joblib` is listed. If it is missing, run `git add -f model/heart_model.joblib model/model_metadata.json`, commit and push again.

## 6. Deploy on Render

Render's free web-service tier is enough for this app. Free instances currently have limits you should know about (check Render's pricing page for the latest): they go to sleep after about 15 minutes without traffic, so the first visit after a pause takes noticeably longer, and they have 512 MB of RAM. This app is small and fits comfortably.

### Option A: Blueprint (uses `render.yaml`)

1. In the Render dashboard choose **New > Blueprint**.
2. Connect your GitHub account and pick the repository.
3. Render reads `render.yaml`, shows the service it will create, and asks you to confirm.
4. Click **Apply** and wait for the build to finish.

### Option B: Manual setup

1. **New > Web Service**, then connect the GitHub repository.
2. Fill in:

| Setting | Value |
|---|---|
| Name | `cardiovascular-risk-assessment` (becomes part of your URL) |
| Language | Python 3 |
| Branch | `main` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 60` |
| Instance Type | Free |

3. Under **Advanced**, add an environment variable: `PYTHON_VERSION` = `3.12.8`. This keeps Render's Python in line with the pinned packages.
4. Set **Health Check Path** to `/health`.
5. Click **Create Web Service**.

### After the build

- Watch the logs. Success looks like `Listening at: http://0.0.0.0:...` followed by `Booting worker`.
- Your site is live at `https://<service-name>.onrender.com`.
- Test it: open the page, run an example, and open `/health` (should show `{"status":"ok"}`).
- Paste the URL into the **Live demo** line of `README.md` and push.

Every `git push` to `main` redeploys automatically.

## 7. Alternative: PythonAnywhere (free account)

1. Create an account at pythonanywhere.com and open a **Bash console**.
2. Clone and install:

```bash
git clone https://github.com/<your-username>/cardiovascular-disease-risk-assessment.git
cd cardiovascular-disease-risk-assessment
python3.12 -m venv ~/.virtualenvs/cvd      # use the closest Python version your account offers
source ~/.virtualenvs/cvd/bin/activate
pip install -r requirements.txt
```

3. On the **Web** tab choose **Add a new web app > Manual configuration** and pick the same Python version.
4. Set **Virtualenv** to `/home/<username>/.virtualenvs/cvd`.
5. Open the **WSGI configuration file** and replace its contents with:

```python
import sys
path = "/home/<username>/cardiovascular-disease-risk-assessment"
if path not in sys.path:
    sys.path.insert(0, path)

from app import app as application
```

6. Under **Static files**, map URL `/static/` to `/home/<username>/cardiovascular-disease-risk-assessment/static`.
7. Click **Reload**. Your site is at `<username>.pythonanywhere.com`.

If PythonAnywhere does not offer Python 3.12, retrain the model in a virtual environment that uses the version it does offer (and adjust the pinned versions to ones that support it), so the saved model matches.

## 8. Custom domain (optional)

On Render: service **Settings > Custom Domains > Add**, enter your domain, then create the DNS record Render shows you (usually a CNAME) at your domain registrar. Render issues the HTTPS certificate automatically once DNS resolves.

## 9. Updating the site

- **Change the page or API:** edit the files, commit, push. Render redeploys.
- **Retrain the model:** run `python train_model.py` locally in the pinned environment, then commit the new `model/` files and push:

```bash
git add model/
git commit -m "Retrain model"
git push
```

- **Upgrade packages:** change the versions in `requirements.txt`, retrain with those versions, commit the model and requirements together.

## 10. Security, privacy and responsible use

- **Do not store or log health values.** The app keeps nothing today. If you add logging, analytics or a database, you take on privacy duties (consent, retention, local law). Do not add them casually.
- **Keep the disclaimer** visible. The estimate is not a diagnosis.
- **Rate limiting:** a public API can be hammered. For a portfolio demo this is optional; if you want it, add `Flask-Limiter`.
- **Debug mode is off** in production. Never set `FLASK_DEBUG=1` on the server.
- **HTTPS** is provided by Render and PythonAnywhere. Do not collect data over plain HTTP.

## 11. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Build fails with `No matching distribution found for numpy==1.26.4` | Render is using a newer Python. Set `PYTHON_VERSION=3.12.8` and redeploy. |
| `RuntimeError: Model files not found in ./model` | The model was not committed. Run `git add -f model/*` , commit, push. |
| Log shows `InconsistentVersionWarning` or the app errors on load | The model was trained with a different scikit-learn version than the server installs. Retrain with the versions in `requirements.txt` and push the new model. |
| `ModuleNotFoundError: No module named 'flask'` | Build command did not run `pip install -r requirements.txt`, or the file is missing from the repo. |
| Deploy hangs on "no open ports detected" | Start command must bind to `$PORT`: `gunicorn app:app --bind 0.0.0.0:$PORT`. |
| Worker killed or out-of-memory | Keep `--workers 1`. Do not add more workers on the free plan. |
| First page load takes 30 to 60 seconds | Free instance was asleep. Normal on the free tier; upgrading removes it. |
| Page has no styling | `static/` folder not pushed, or (PythonAnywhere) the static files mapping is missing. |
| Fonts look different from the screenshot | The page loads Google Fonts. Offline or blocked networks fall back to system fonts; the layout still works. |
| API returns 422 | One of the 13 fields is missing or out of range. The response lists each one. |
| `Activate.ps1 cannot be loaded` (Windows) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |

## 12. Quick reference

| Task | Command |
|---|---|
| Train | `python train_model.py --no-show` |
| Run locally | `python app.py` |
| Production start command | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 60` |
| Health check | `GET /health` |
| Model details | `GET /api/model-info` |
