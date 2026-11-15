"""Run a real Chromium upload/download and SHAP smoke test against Streamlit.

Usage: python tests/browser_smoke.py
Prerequisite: python -m playwright install chromium
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess import CATEGORICAL_VALUES, load_artifact, predict_batch, predict_probabilities


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    artifact = load_artifact()
    raw: dict[str, object] = {
        name: values[0] for name, values in CATEGORICAL_VALUES.items()
    }
    raw.update(PhoneService="Yes", SeniorCitizen=0, tenure=12,
               MonthlyCharges=70.0, TotalCharges=2000.0)
    probability = predict_probabilities(artifact, pd.DataFrame([raw]))[0]
    with tempfile.TemporaryDirectory(prefix="churn-browser-") as temporary:
        log_path = reports / "browser_smoke_streamlit.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                 "--server.port", str(port), "--server.address", "127.0.0.1",
                 "--server.headless", "true", "--server.fileWatcherType", "none"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text(encoding="utf-8"))
                    try:
                        with urlopen(url + "/_stcore/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(0.3)
                else:
                    raise TimeoutError("Streamlit did not start within 40 seconds")
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    page = browser.new_page(viewport={"width": 1280, "height": 1000})
                    page.set_default_timeout(40000)
                    page.goto(url)
                    page.get_by_role("button", name="Predict churn", exact=True).click()
                    expect(page.get_by_text(f"Predicted churn: {'Yes' if probability >= artifact['threshold'] else 'No'}", exact=True)).to_be_visible()
                    expect(page.get_by_test_id("stMetricValue")).to_have_text(f"{probability:.2%}")
                    page.get_by_role("button", name="Explain prediction", exact=True).click()
                    plot = page.get_by_test_id("stImage").locator("img")
                    expect(plot).to_be_visible()
                    assert plot.evaluate("img => img.complete && img.naturalWidth > 0")
                    plot.scroll_into_view_if_needed()
                    page.screenshot(path=str(reports / "app_single.png"))
                    assert page.get_by_test_id("stException").count() == 0

                    page.get_by_text("Batch Prediction", exact=True).click()
                    uploader = page.locator('input[type="file"]')
                    expect(page.get_by_role("button", name="Download sample CSV")).to_be_visible()
                    rows = pd.read_csv(ROOT / "data" / "telecom.csv").head(5).drop(columns="Churn")
                    rows["TotalCharges"] = rows["TotalCharges"].astype(object)
                    rows.loc[1, "TotalCharges"] = ""
                    rows.loc[2, "MonthlyCharges"] = -1
                    expected = predict_batch(artifact, rows)
                    uploader.set_input_files({"name": "customers.csv", "mimeType": "text/csv",
                                              "buffer": rows.to_csv(index=False).encode()})
                    expect(page.get_by_text("Predicted 4 of 5 records.", exact=True)).to_be_visible()
                    with page.expect_download() as download_event:
                        page.get_by_role("button", name="Download predictions", exact=True).click()
                    download = download_event.value
                    downloaded = pd.read_csv(download.path(), keep_default_na=False)
                    assert len(downloaded) == len(rows)
                    assert downloaded["Predicted_Churn"].tolist() == expected["Predicted_Churn"].tolist()
                    assert downloaded["Validation_Error"].tolist() == expected["Validation_Error"].tolist()
                    valid = expected["Validation_Error"].eq("")
                    np.testing.assert_allclose(downloaded.loc[valid, "Churn_Probability"].astype(float),
                                               expected.loc[valid, "Churn_Probability"], atol=1e-15, rtol=0)
                    assert downloaded.loc[2, "Churn_Probability"] == ""
                    page.screenshot(path=str(reports / "app_batch.png"))
                    assert page.get_by_test_id("stException").count() == 0

                    uploader.set_input_files({"name": "missing_column.csv", "mimeType": "text/csv",
                                              "buffer": rows.drop(columns="PaymentMethod").to_csv(index=False).encode()})
                    expect(page.get_by_text("Missing required columns: PaymentMethod", exact=True)).to_be_visible()
                    uploader.set_input_files({"name": "malformed.csv", "mimeType": "text/csv",
                                              "buffer": b"gender,gender\nFemale,Male\n"})
                    expect(page.get_by_text("Could not read customer CSV: Duplicate CSV column names are not allowed.", exact=True)).to_be_visible()
                    assert page.get_by_test_id("stException").count() == 0
                    browser.close()
                evidence = {
                    "model_id": artifact["model_id"], "browser": "Chromium (Playwright)",
                    "single_prediction_matches_backend": True, "shap_image_loaded": True,
                    "batch_upload_rows": 5, "batch_valid_predictions": 4, "batch_invalid_rows": 1,
                    "download_matches_backend": True, "missing_column_error": True,
                    "duplicate_header_error": True,
                }
                (reports / "browser_smoke.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(evidence, indent=2))
            finally:
                if os.name == "nt":
                    # Stop Streamlit and its child processes on Windows.
                    if process.poll() is None:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                            timeout=15,
                        )
                elif process.poll() is None:
                    process.terminate()

                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
