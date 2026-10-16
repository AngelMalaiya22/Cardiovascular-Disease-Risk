(function () {
  "use strict";

  const form = document.getElementById("assessment");
  const submitBtn = document.getElementById("submit");
  const result = document.querySelector(".result");
  const states = {
    empty: document.getElementById("state-empty"),
    loading: document.getElementById("state-loading"),
    error: document.getElementById("state-error"),
    result: document.getElementById("state-result"),
  };

  function show(name) {
    Object.entries(states).forEach(([key, el]) => { el.hidden = key !== name; });
  }

  function clearFieldErrors() {
    form.querySelectorAll(".field").forEach((f) => {
      f.classList.remove("invalid");
      f.querySelector(".msg").textContent = "";
      const control = f.querySelector("input, select");
      control.removeAttribute("aria-invalid");
    });
  }

  function setFieldError(name, message) {
    const field = form.querySelector(`.field[data-field="${name}"]`);
    if (!field) return;
    field.classList.add("invalid");
    field.querySelector(".msg").textContent = message;
    field.querySelector("input, select").setAttribute("aria-invalid", "true");
  }

  // Client-side check mirrors the server rules so people get instant feedback.
  function validateClient() {
    let firstInvalid = null;
    form.querySelectorAll(".field").forEach((f) => {
      const control = f.querySelector("input, select");
      const name = f.dataset.field;
      let message = "";
      if (control.value === "") {
        message = "This field is required.";
      } else if (control.type === "number") {
        const v = Number(control.value);
        const min = Number(control.min), max = Number(control.max);
        if (Number.isNaN(v) || v < min || v > max) message = `Enter a value between ${min} and ${max}.`;
      }
      if (message) {
        setFieldError(name, message);
        if (!firstInvalid) firstInvalid = control;
      }
    });
    if (firstInvalid) firstInvalid.focus();
    return !firstInvalid;
  }

  function collect() {
    const data = {};
    new FormData(form).forEach((value, key) => { data[key] = value; });
    return data;
  }

  function render(data) {
    const { probability, band, band_label, cutoffs, warnings, model } = data;

    result.dataset.band = band;
    document.getElementById("percent").textContent = probability.toFixed(1);
    document.getElementById("band").textContent = band_label;

    // gauge zones and ticks follow the server-side cut-offs
    const gauge = document.getElementById("gauge");
    gauge.style.setProperty("--low-w", cutoffs.low);
    gauge.style.setProperty("--mid-w", cutoffs.high - cutoffs.low);
    gauge.style.setProperty("--high-w", 100 - cutoffs.high);
    const tickLow = document.getElementById("tick-low");
    const tickHigh = document.getElementById("tick-high");
    tickLow.textContent = cutoffs.low + "%";
    tickLow.style.left = cutoffs.low + "%";
    tickHigh.textContent = cutoffs.high + "%";
    tickHigh.style.left = cutoffs.high + "%";
    gauge.setAttribute("aria-label", `Estimated probability ${probability.toFixed(1)} percent, ${band_label}`);
    document.getElementById("marker").style.left = Math.min(Math.max(probability, 0), 100) + "%";

    const messages = {
      low: "The values you entered resemble patients in the dataset who did not have heart disease. This is not a clearance: a low estimate does not rule out heart problems.",
      moderate: "The values you entered are a mixed picture: some resemble patients with heart disease and some do not. If you have symptoms or concerns, talk to a doctor.",
      high: "The values you entered resemble patients in the dataset who had heart disease. Discuss these results with a doctor; this tool cannot diagnose.",
    };
    document.getElementById("band-message").textContent = messages[band];

    const list = document.getElementById("warnings");
    list.innerHTML = "";
    warnings.forEach((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      list.appendChild(li);
    });

    document.getElementById("model-note").textContent =
      `Model: ${model.name}. Cross-validated accuracy ${model.cv_accuracy}%, ` +
      `accuracy ${model.test_accuracy}% on a held-out test set of ${model.n_test} patients.`;

    show("result");
    if (window.matchMedia("(max-width: 900px)").matches) {
      result.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearFieldErrors();
    if (!validateClient()) return;

    submitBtn.disabled = true;
    show("loading");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collect()),
      });
      const data = await response.json();

      if (response.status === 422 && data.errors) {
        Object.entries(data.errors).forEach(([name, msg]) => setFieldError(name, msg));
        document.getElementById("error-text").textContent = "Some values need fixing. Check the highlighted fields.";
        show("error");
      } else if (!response.ok) {
        throw new Error(data.error || "Request failed");
      } else {
        render(data);
      }
    } catch (err) {
      document.getElementById("error-text").textContent =
        "The estimate could not be calculated. Check your connection and try again.";
      show("error");
    } finally {
      submitBtn.disabled = false;
    }
  });

  form.addEventListener("reset", () => {
    clearFieldErrors();
    delete result.dataset.band;
    show("empty");
  });

  document.querySelectorAll("[data-example]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const example = window.EXAMPLES[btn.dataset.example];
      clearFieldErrors();
      Object.entries(example).forEach(([name, value]) => {
        const control = form.elements[name];
        if (control) control.value = String(value);
      });
      show("empty");
    });
  });
})();
