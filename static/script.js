// Sloth Study front-end. Plain JavaScript, no frameworks.
// Every piece of AI text is inserted with textContent (never innerHTML),
// so nothing the model writes can inject HTML or scripts into the page.

(() => {
  "use strict";

  const MAX_BYTES = 10 * 1024 * 1024;
  const ALLOWED = [".pdf", ".txt"];
  const LETTERS = ["A", "B", "C", "D"];

  const $ = (id) => document.getElementById(id);
  const els = {
    status: $("model-status"),
    statusText: $("status-text"),
    setupHint: $("setup-hint"),
    errorPanel: $("error-panel"),
    errorMessage: $("error-message"),
    errorDismiss: $("error-dismiss"),
    uploadPanel: $("upload-panel"),
    uploadForm: $("upload-form"),
    uploadMessage: $("upload-message"),
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    fileChip: $("file-chip"),
    generateBtn: $("generate-btn"),
    sampleBtn: $("sample-btn"),
    loadingPanel: $("loading-panel"),
    elapsed: $("elapsed"),
    results: $("results"),
    resultsTitle: $("results-title"),
    sourceLabel: $("source-label"),
    sourceMeta: $("source-meta"),
    newUploadBtn: $("new-upload-btn"),
    overview: $("summary-overview"),
    points: $("summary-points"),
    definitionsBlock: $("definitions-block"),
    definitions: $("summary-definitions"),
    takeaways: $("summary-takeaways"),
    flashcard: $("flashcard"),
    cardFront: $("card-front"),
    cardBack: $("card-back"),
    cardQuestion: $("card-question"),
    cardAnswer: $("card-answer"),
    cardCounter: $("card-counter"),
    prevCard: $("prev-card"),
    nextCard: $("next-card"),
    quizForm: $("quiz-form"),
    quizQuestions: $("quiz-questions"),
    quizMessage: $("quiz-message"),
    quizSubmit: $("quiz-submit"),
    quizRetake: $("quiz-retake"),
    quizResult: $("quiz-result"),
    scoreValue: $("score-value"),
    scorePercent: $("score-percent"),
    scoreComment: $("score-comment"),
    aiCredit: $("ai-credit"),
  };
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));

  const state = { file: null, cards: [], cardIndex: 0, flipped: false, quiz: null, timer: null };

  // ---------- small helpers ----------

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function extensionOf(name) {
    const dot = name.lastIndexOf(".");
    return dot >= 0 ? name.slice(dot).toLowerCase() : "";
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  // fetch() wrapper that always gives back a friendly error message.
  async function requestJSON(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch {
      throw new Error("Couldn't reach the Sloth Study server. Is the terminal running app.py still open?");
    }
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    if (!response.ok) {
      throw new Error((data && data.error) || `The server answered with an error (${response.status}).`);
    }
    if (!data) throw new Error("The server sent an unexpected response. Please try again.");
    return data;
  }

 function showError(message) {
    const errorText = String(message || "An unexpected error occurred.");
    els.errorMessage.textContent = errorText;
    els.errorPanel.hidden = false;
    els.errorPanel.scrollIntoView({ block: "nearest" });
}



  // ---------- local AI status ----------

  async function checkHealth() {
    try {
      const data = await requestJSON("/api/health");
      setStatus(data);
    } catch (err) {
      setStatus({ ready: false, message: err.message });
    }
  }

  function setStatus(data) {
    els.status.classList.toggle("is-ready", Boolean(data.ready));
    els.status.classList.toggle("is-warn", !data.ready);
    els.statusText.textContent = data.ready ? `Local AI ready · ${data.model}` : "Local AI not ready";
    els.status.title = data.message || "";
    els.setupHint.textContent = data.ready ? "" : data.message || "";
    els.setupHint.hidden = Boolean(data.ready) || !data.message;
  }

  // ---------- choosing a file ----------

  function chooseFile(file) {
    els.uploadMessage.textContent = "";
    state.file = null;
    els.fileChip.hidden = true;
    if (!file) return;

    if (!ALLOWED.includes(extensionOf(file.name))) {
      els.uploadMessage.textContent = "That file type isn't supported. Please choose a PDF or TXT file.";
      return;
    }
    if (file.size === 0) {
      els.uploadMessage.textContent = "That file is empty. Please choose a file with some notes in it.";
      return;
    }
    if (file.size > MAX_BYTES) {
      els.uploadMessage.textContent = "That file is larger than 10 MB. Try uploading one chapter at a time.";
      return;
    }
    state.file = file;
    els.fileChip.textContent = `${file.name} · ${formatBytes(file.size)}`;
    els.fileChip.hidden = false;
  }

  els.fileInput.addEventListener("change", () => chooseFile(els.fileInput.files[0]));

  ["dragenter", "dragover"].forEach((type) =>
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("is-dragging");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("is-dragging");
    })
  );
  els.dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    chooseFile(file);
  });
  // Stop the browser from opening a file dropped just outside the drop zone.
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  // ---------- generating ----------

  els.uploadForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!state.file) {
      els.uploadMessage.textContent = "Choose a PDF or TXT file first, or try the demo notes.";
      return;
    }
    const formData = new FormData();
    formData.append("file", state.file);
    generate(formData);
  });

  els.sampleBtn.addEventListener("click", () => {
    const formData = new FormData();
    formData.append("use_sample", "1");
    generate(formData);
  });

  function startTimer() {
    const started = Date.now();
    els.elapsed.textContent = "0:00";
    state.timer = setInterval(() => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      els.elapsed.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    }, 1000);
  }

  function stopTimer() {
    clearInterval(state.timer);
    state.timer = null;
  }

  async function generate(formData) {
    hideError();
    els.generateBtn.disabled = true;
    els.sampleBtn.disabled = true;
    els.uploadPanel.hidden = true;
    els.results.hidden = true;
    els.loadingPanel.hidden = false;
    startTimer();
    try {
      const data = await requestJSON("/api/generate", { method: "POST", body: formData });
      renderResults(data);
      els.loadingPanel.hidden = true;
      els.results.hidden = false;
      selectTab("summary");
      els.resultsTitle.focus();
    } catch (err) {
    console.error("Sloth Study generation error:", err);

    els.loadingPanel.hidden = true;
    els.uploadPanel.hidden = false;

    showError(err?.message || "An unexpected error occurred.");
    checkHealth();
} finally {
      stopTimer();
      els.generateBtn.disabled = false;
      els.sampleBtn.disabled = false;
    }
  }

  els.errorDismiss.addEventListener("click", hideError);

  els.newUploadBtn.addEventListener("click", () => {
    els.results.hidden = true;
    els.uploadPanel.hidden = false;
    els.fileInput.value = "";
    chooseFile(null);
    hideError();
    window.scrollTo({ top: 0 });
    els.fileInput.focus();
  });

  // ---------- results ----------

  function renderResults(data) {
    const source = data.source;
    els.sourceLabel.textContent = source.is_sample ? "Demo · built-in sample notes" : "From your notes";
    let meta = `${source.filename} · ${source.characters.toLocaleString()} characters used`;
    if (source.truncated) meta += " (long file: only the first part was sent to the model)";
    els.sourceMeta.textContent = meta;

    renderSummary(data.summary);

    state.cards = data.flashcards;
    state.cardIndex = 0;
    renderCard();

    state.quiz = data.quiz;
    renderQuiz();

    els.aiCredit.textContent = `Generated by ${data.model} running locally through Ollama in ${Math.round(data.seconds)} seconds.`;
  }

  function renderSummary(summary) {
    els.overview.textContent = summary.overview;

    els.points.replaceChildren(...summary.key_points.map((point) => make("li", null, point)));

    els.definitions.replaceChildren(
      ...summary.definitions.map((item) => {
        const wrap = make("div");
        wrap.append(make("dt", null, item.term), make("dd", null, item.definition));
        return wrap;
      })
    );
    els.definitionsBlock.hidden = summary.definitions.length === 0;

    els.takeaways.replaceChildren(...summary.takeaways.map((item) => make("li", null, item)));
  }

  // ---------- tabs ----------

  function selectTab(name) {
    tabs.forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      $(tab.getAttribute("aria-controls")).hidden = !active;
    });
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      let next = null;
      if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
      if (event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
      if (next) {
        event.preventDefault();
        selectTab(next.dataset.tab);
        next.focus();
      }
    });
  });

  document.querySelectorAll("[data-goto]").forEach((button) =>
    button.addEventListener("click", () => {
      selectTab(button.dataset.goto);
      const tab = tabs.find((t) => t.dataset.tab === button.dataset.goto);
      tab.focus();
      tab.scrollIntoView({ block: "nearest" });
    })
  );

  // ---------- flashcards ----------

  function setFlipped(flipped) {
    state.flipped = flipped;
    els.flashcard.classList.toggle("is-flipped", flipped);
    els.cardFront.setAttribute("aria-hidden", String(flipped));
    els.cardBack.setAttribute("aria-hidden", String(!flipped));
    const card = state.cards[state.cardIndex];
    if (card) {
      els.flashcard.setAttribute(
        "aria-label",
        flipped ? `Answer: ${card.answer}. Press to see the question again.` : `Question: ${card.question}. Press to reveal the answer.`
      );
    }
  }

  function renderCard() {
    const total = state.cards.length;
    const card = state.cards[state.cardIndex];
    els.cardQuestion.textContent = card.question;
    els.cardAnswer.textContent = card.answer;
    els.cardCounter.textContent = `Card ${state.cardIndex + 1} of ${total}`;
    els.prevCard.disabled = state.cardIndex === 0;
    els.nextCard.disabled = state.cardIndex === total - 1;
    setFlipped(false);
  }

  function moveCard(step) {
    const next = state.cardIndex + step;
    if (next < 0 || next >= state.cards.length) return;
    state.cardIndex = next;
    renderCard();
  }

  els.flashcard.addEventListener("click", () => setFlipped(!state.flipped));
  els.prevCard.addEventListener("click", () => moveCard(-1));
  els.nextCard.addEventListener("click", () => moveCard(1));

  document.addEventListener("keydown", (event) => {
    const panelOpen = !els.results.hidden && !$("panel-flashcards").hidden;
    const target = event.target;
    if (!panelOpen || target.matches("input, textarea, [role='tab']")) return;
    if (event.key === "ArrowRight") moveCard(1);
    if (event.key === "ArrowLeft") moveCard(-1);
  });

  // ---------- quiz ----------

  function renderQuiz() {
    els.quizResult.hidden = true;
    els.quizMessage.textContent = "";
    els.quizSubmit.hidden = false;
    els.quizRetake.hidden = true;
    els.quizForm.classList.remove("quiz-done");

    const total = state.quiz.questions.length;
    els.quizQuestions.replaceChildren(
      ...state.quiz.questions.map((question, qIndex) => {
        const fieldset = make("fieldset", "quiz-q");
        const legend = make("legend");
        legend.append(make("span", "q-number", `Question ${qIndex + 1} of ${total}`), document.createTextNode(question.question));
        const options = make("div", "options");
        question.options.forEach((text, oIndex) => {
          const label = make("label", "option");
          const input = document.createElement("input");
          input.type = "radio";
          input.name = `q${qIndex}`;
          input.value = String(oIndex);
          label.append(input, make("span", "option-letter", LETTERS[oIndex]), make("span", "option-text", text));
          options.append(label);
        });
        const explanation = make("p", "explanation");
        explanation.hidden = true;
        fieldset.append(legend, options, explanation);
        return fieldset;
      })
    );
  }

  function collectAnswers() {
    return state.quiz.questions.map((_, qIndex) => {
      const checked = els.quizForm.querySelector(`input[name="q${qIndex}"]:checked`);
      return checked ? Number(checked.value) : null;
    });
  }

  els.quizForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const answers = collectAnswers();
    const missing = answers.filter((a) => a === null).length;
    if (missing > 0) {
      els.quizMessage.textContent = `You still have ${missing} unanswered question${missing === 1 ? "" : "s"}.`;
      return;
    }
    els.quizMessage.textContent = "";
    els.quizSubmit.disabled = true;
    try {
      const result = await requestJSON(`/api/quiz/${encodeURIComponent(state.quiz.id)}/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers }),
      });
      showQuizResult(result);
    } catch (err) {
      els.quizMessage.textContent = err.message;
    } finally {
      els.quizSubmit.disabled = false;
    }
  });

  function showQuizResult(result) {
    const fieldsets = els.quizQuestions.querySelectorAll(".quiz-q");
    result.results.forEach((item, qIndex) => {
      const fieldset = fieldsets[qIndex];
      const labels = fieldset.querySelectorAll(".option");
      labels.forEach((label, oIndex) => {
        label.querySelector("input").disabled = true;
        if (oIndex === item.correct_index) {
          label.classList.add("is-correct");
          label.append(make("span", "tag", "Correct answer"));
        } else if (oIndex === item.selected) {
          label.classList.add("is-wrong");
          label.append(make("span", "tag", "Your answer"));
        }
      });
      const explanation = fieldset.querySelector(".explanation");
      const verdict = item.is_correct ? "Correct! " : `Not quite. The answer is ${LETTERS[item.correct_index]}. `;
      explanation.textContent = verdict + item.explanation;
      explanation.hidden = false;
    });

    els.quizForm.classList.add("quiz-done");
    els.scoreValue.textContent = `${result.score} / ${result.total}`;
    els.scorePercent.textContent = `${result.percent}% correct`;
    els.scoreComment.textContent =
      result.score === result.total
        ? "Perfect score. Even a sloth would sprint for that."
        : result.score >= 3
        ? "Nice work! Review the explanations below, then try the flashcards again."
        : "Good start. Go back over the summary and flashcards, then retake the quiz.";
    els.quizResult.hidden = false;
    els.quizSubmit.hidden = true;
    els.quizRetake.hidden = false;
    els.quizResult.focus();
    els.quizResult.scrollIntoView({ block: "start" });
  }

  els.quizRetake.addEventListener("click", () => {
    renderQuiz();
    els.quizQuestions.querySelector("input").focus();
  });

  checkHealth();
})();
