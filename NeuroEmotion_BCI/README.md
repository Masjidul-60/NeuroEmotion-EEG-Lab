# NeuroEmotion BCI Project

An educational EEG signal-processing demo with a polished Streamlit dashboard. It generates synthetic EEG-like signals, calculates Theta, Alpha, Beta, and Gamma band power, and displays a simple alpha-versus-beta state estimate.

> **Important:** The signal is simulated, and the state estimate is a basic heuristic. This project is not a validated medical or emotion-recognition system.

## Run on Windows

Open PowerShell in the `NeuroEmotion_BCI` project folder, then run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app\dashboard.py
```

If PowerShell blocks environment activation, either allow scripts for the current terminal with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again, or skip activation and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app\dashboard.py
```

If PowerShell opens outside the project folder, enter it first (quote the path because it contains spaces):

```powershell
cd "G:\My Research\NeuroEmotion_BCI_Project\NeuroEmotion_BCI"
```

## Run on macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

## Dashboard controls

- **Run live simulation** refreshes the generated signal once per second. Turn it off to stop updates.
- **Signal profile** selects random, alpha-dominant (relaxed-style), or beta-dominant (focused-style) synthetic activity.
- **Sampling rate** and **Signal window** adjust the generated data.
- **Generate EEG sample** creates a single sample when live mode is off.
- The dashboard shows the signal, band powers, alpha/beta heuristic, and recent power trend.
- **Learning practice** demonstrates a synthetic beginner-to-expert progression for a focus (Beta) or calm (Alpha) target, alongside a Bengali explainer for EEG and BCI calibration, modeling, and feedback.

The beginner-to-expert display only illustrates how signal-to-noise might be varied in a teaching example. It is not a trained model, a real learning curve, measured EEG, or a claim that experts share one brain-signal pattern. The live controls use Streamlit fragments, so install the declared Streamlit version (1.37.0 or later).
