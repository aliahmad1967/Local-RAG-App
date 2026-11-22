/* LocalRAG Phase 1 UI — no build step, no external assets. */
(() => {
  "use strict";

  const I18N = {
    ar: {
      brandTag: "مكتبتك، على جهازك — بلا اتصال",
      checking: "جارٍ الفحص", ollamaUp: "Ollama جاهز", ollamaDown: "Ollama غير مشغّل",
      dropTitle: "أفلت الكتب هنا", dropHint: "ملفات PDF — تُقرأ محليًا، صفحة بصفحة",
      choose: "اختر ملفات PDF", ocrEngine: "محرك التعريف", library: "المكتبة",
      activeFolio: "الملف النشط", done: "مكتملة", flagged: "تحتاج مراجعة", failed: "فاشلة",
      pause: "إيقاف مؤقت", resume: "متابعة", cancel: "إلغاء", retry: "إعادة الصفحات المتعثرة",
      pages: "الصفحات", search: "بحث", benchmark: "مقارنة المحركات", events: "السجل",
      pageList: "قائمة الصفحات", preview: "معاينة الصفحة", noPreview: "لا توجد صورة بعد",
      searchPlaceholder: "ابحث في الكتب المفهرسة…", runSearch: "ابحث",
      searchScopeCurrent: "هذا الكتاب", searchScopeAll: "كل الكتب",
      pageRange: "صفحة", topK: "النتائج", noSearch: "اكتب سؤالًا لعرض ترتيب البحث والتقريب.",
      searching: "جارٍ البحث…", denseRank: "كثيف", bm25Rank: "BM25",
      fusedRank: "مدمج", rerankScore: "إعادة ترتيب", parentWindow: "النافذة الأكبر",
      searchStats: "نتائج: {dense} كثيف · {bm25} BM25 · {fused} مدمج · {final} نهائية",
      benchIntro: "اختر حتى ١٠ صفحات، وشغّل محركات التعريف عليها.",
      benchHelp: "تُعرض النتائج جنبًا إلى جنب مع الزمن لكل صفحة. يُفرَّغ المحرك من الذاكرة بين كل محرك وآخر.",
      runBenchmark: "شغّل المقارنة", addPages: "أضف", pageNumbers: "أرقام الصفحات (مثال: 3, 8, 15)",
      emptyLine1: "اختر كتابًا لمتابعة التعرف على حروفه.",
      emptyLine2: "تُحفظ كل صفحة فورًا؛ لا تخشَ إغلاق التطبيق.",
      imported: "تم استيراد {name}", benchmarkDone: "انتهت المقارنة", retryQueued: "تمت جدولة إعادة الصفحات",
      useEngine: "استخدم هذا المحرك", noBench: "لم تُجرَ مقارنة بعد.",
      eta: "الوقت المتبقي", page: "صفحة", noPages: "لا صفحات بعد", noEvents: "لا أحداث بعد",
      chat: "المحادثة", memory: "الذاكرة", settings: "الإعدادات",
      newConversation: "محادثة جديدة", conversation: "محادثة",
      modeChat: "سؤال وجواب", modeSummarize: "تلخيص نطاق", modeCompare: "مقارنة الكتب",
      chatPlaceholder: "اسأل عن الكتب المفهرسة…", send: "أرسل", sending: "جارٍ التوليد…",
      noChat: "ابدأ محادثة؛ كل إجابة ستكون مستندة إلى صفحات الكتب.",
      you: "أنت", assistant: "المساعد", citations: "المصادر", warnings: "تنبيهات",
      citationOpen: "افتح الاقتباس", fact: "حقيقة", preference: "تفضيل",
      memoryDraftPlaceholder: "حقيقة تؤكد حفظها لهذه المحادثة (اختياري)",
      memoryPlaceholder: "حقيقة أو تفضيل مؤكد من المستخدم…", addMemory: "أضف للذاكرة",
      memoryNote: "تُحفظ الحقائق المؤكدة فقط؛ لا يكتب المحرك ذاكرة تلقائيًا.",
      noMemories: "لا توجد حقائق محفوظة بعد.", edit: "تعديل", delete: "حذف",
      memorySaved: "تم حفظ الذاكرة", memoryUpdated: "تم تحديث الذاكرة", memoryDeleted: "تم حذف الذاكرة",
      removeAll: "حذف الكل", deleteBook: "حذف المصدر", deleteConfirm: "حذف «{name}» وجميع بياناته؟",
      removeAllConfirm: "حذف جميع المصادر والبيانات المرتبطة بها؟", bookDeleted: "تم حذف المصدر",
      allBooksDeleted: "تم حذف جميع المصادر",
      llmModel: "نموذج اللغة", temperature: "الحرارة", contextTokens: "حد السياق",
      historyTurns: "الذاكرة القريبة", saveSettings: "حفظ الإعدادات", settingsSaved: "تم حفظ الإعدادات",
      noModels: "لا توجد نماذج مثبتة", closeCitation: "إغلاق",
      sourceOverview: "ملخص المصدر", analyzingSource: "جارٍ تحليل النص المضمّن…",
      overviewReady: "جاهز", overviewPending: "بانتظار OCR", overviewFallback: "ملخص مبدئي",
    },
    en: {
      brandTag: "Your library, on your machine — offline",
      checking: "Checking", ollamaUp: "Ollama ready", ollamaDown: "Ollama is down",
      dropTitle: "Drop your books here", dropHint: "PDF files — read locally, page by page",
      choose: "Choose PDF files", ocrEngine: "OCR engine", library: "Library",
      activeFolio: "Active folio", done: "done", flagged: "needs review", failed: "failed",
      pause: "Pause", resume: "Resume", cancel: "Cancel", retry: "Retry problem pages",
      pages: "Pages", search: "Search", benchmark: "OCR benchmark", events: "Activity",
      pageList: "Page list", preview: "Page preview", noPreview: "No image yet",
      searchPlaceholder: "Search the indexed books…", runSearch: "Search",
      searchScopeCurrent: "This book", searchScopeAll: "All books",
      pageRange: "Page", topK: "Results", noSearch: "Ask a question to see dense, BM25, fused and rerank scores.",
      searching: "Searching…", denseRank: "dense", bm25Rank: "BM25",
      fusedRank: "fused", rerankScore: "rerank", parentWindow: "Parent window",
      searchStats: "candidates: {dense} dense · {bm25} BM25 · {fused} fused · {final} final",
      benchIntro: "Pick up to 10 pages and run every OCR engine on them.",
      benchHelp: "Results appear side by side with per-page timing. Each model is unloaded from VRAM before the next runs.",
      runBenchmark: "Run benchmark", addPages: "Add", pageNumbers: "Page numbers (e.g. 3, 8, 15)",
      emptyLine1: "Select a book to watch its pages being read.",
      emptyLine2: "Every page is saved immediately — closing the app is safe.",
      imported: "Imported {name}", benchmarkDone: "Benchmark finished", retryQueued: "Retry queued",
      useEngine: "Use this engine", noBench: "No benchmark run yet.",
      eta: "ETA", page: "page", noPages: "No pages yet", noEvents: "No events yet",
      chat: "Chat", memory: "Memory", settings: "Settings",
      newConversation: "New conversation", conversation: "Conversation",
      modeChat: "Q&A", modeSummarize: "Summarize range", modeCompare: "Compare books",
      chatPlaceholder: "Ask about the indexed books…", send: "Send", sending: "Generating…",
      noChat: "Start a conversation; every answer is grounded in book pages.",
      you: "You", assistant: "Assistant", citations: "Sources", warnings: "Warnings",
      citationOpen: "Open citation", fact: "Fact", preference: "Preference",
      memoryDraftPlaceholder: "A confirmed fact for this conversation (optional)",
      memoryPlaceholder: "A user-confirmed fact or preference…", addMemory: "Add memory",
      memoryNote: "Only confirmed facts are stored. The assistant never writes memory automatically.",
      noMemories: "No saved memories yet.", edit: "Edit", delete: "Delete",
      memorySaved: "Memory saved", memoryUpdated: "Memory updated", memoryDeleted: "Memory deleted",
      removeAll: "Remove all", deleteBook: "Remove source", deleteConfirm: "Remove “{name}” and all its data?",
      removeAllConfirm: "Remove all sources and their associated data?", bookDeleted: "Source removed",
      allBooksDeleted: "All sources removed",
      llmModel: "LLM model", temperature: "Temperature", contextTokens: "Context budget",
      historyTurns: "Recent turns", saveSettings: "Save settings", settingsSaved: "Settings saved",
      noModels: "No installed models", closeCitation: "Close",
      sourceOverview: "Source overview", analyzingSource: "Analyzing embedded text…",
      overviewReady: "Ready", overviewPending: "Awaiting OCR", overviewFallback: "Basic overview",
    },
  };

  const $ = (id) => document.getElementById(id);
  let lang = localStorage.getItem("localrag.lang") || "ar";
  let theme = localStorage.getItem("localrag.theme") || "dark";
  const S = {
    books: [], bookId: null, detail: null, pages: [], pageNo: null,
    benchmark: null, searchResult: null, activeTab: "pages", tick: 0, busy: false, engines: [],
    conversations: [], conversationId: null, chatMessages: [], chatBusy: false,
    lastSources: {}, memories: [], models: { installed: [], current: "" }, settings: {},
    overview: null,
  };

  const t = (key, vars = {}) => {
    let value = (I18N[lang] && I18N[lang][key]) || I18N.en[key] || key;
    for (const [k, v] of Object.entries(vars)) value = value.replace(`{${k}}`, v);
    return value;
  };

  function applyLang() {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
    $("langBtn").textContent = lang === "ar" ? "EN" : "عربي";
    if (S.benchmark) renderBenchmark();
    if (S.searchResult) renderSearchResults();
    if (S.chatMessages.length) renderChatLog();
    if (S.memories.length) renderMemories();
  }

  function applyTheme() {
    document.documentElement.dataset.theme = theme;
    $("themeBtn").textContent = theme === "dark" ? "☀" : "☾";
  }

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    let data = {};
    try { data = await response.json(); } catch (_) { /* empty body */ }
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    return data;
  }

  function toast(message, kind = "") {
    const node = document.createElement("div");
    node.className = `toast ${kind}`;
    node.textContent = message;
    $("toasts").appendChild(node);
    setTimeout(() => node.remove(), 6000);
  }

  // -- health / engines ---------------------------------------------------
  async function refreshHealth() {
    try {
      const health = await api("/api/health");
      const pill = $("ollamaPill");
      pill.className = `pill ${health.ollama.running ? "pill-on" : "pill-off"}`;
      pill.lastElementChild.textContent = health.ollama.running ? t("ollamaUp") : t("ollamaDown");
      S.engines = health.ocr_engines;
      const select = $("engineSelect");
      const previous = select.value || health.default_engine;
      select.innerHTML = "";
      health.ocr_engines.forEach((engine) => {
        const option = document.createElement("option");
        option.value = engine.name;
        option.textContent = engine.name + (engine.available ? "" : "  ✕");
        select.appendChild(option);
      });
      select.value = previous || health.default_engine;
    } catch (error) {
      toast(error.message, "error");
    }
  }

  // -- library ------------------------------------------------------------
  function languageCode(language) {
    return language === "ar" ? "AR" : language === "en" ? "EN" : language === "mixed" ? "AR/EN" : "··";
  }

  function renderBooks() {
    const list = $("bookList");
    list.innerHTML = "";
    $("bookCount").textContent = String(S.books.length);
    S.books.forEach((book) => {
      const progress = book.progress || {};
      const card = document.createElement("button");
      card.type = "button";
      card.className = "book-card" + (book.id === S.bookId ? " selected" : "");
      card.innerHTML = `
        <span class="spine"><span class="spine-code">${languageCode(book.language)}</span></span>
        <span class="book-info">
          <span class="book-title"></span>
          <span class="book-sub">${book.page_count} ${t("page")}${book.page_count === 1 ? "" : "s"} · ${book.ocr_engine || "—"}</span>
          <span class="mini-bar"><i style="width:${progress.percent || 0}%"></i></span>
          <span class="mini-meta"><span>${progress.done || 0}/${book.page_count}</span><span>${progress.job_status || "—"}</span></span>
        </span>`;
      card.querySelector(".book-title").textContent = book.title;
      card.addEventListener("click", () => selectBook(book.id));
      list.appendChild(card);
    });
  }

  async function refreshBooks() {
    S.books = await api("/api/books");
    if (!S.bookId && S.books.length) S.bookId = S.books[0].id;
    if (S.bookId && !S.books.some((b) => b.id === S.bookId)) S.bookId = S.books[0]?.id || null;
    renderBooks();
    return S.bookId;
  }

  async function deleteCurrentBook() {
    if (!S.bookId || !S.detail) return;
    const name = S.detail.book.title;
    if (!window.confirm(t("deleteConfirm", { name }))) return;
    await api(`/api/books/${encodeURIComponent(S.bookId)}`, { method: "DELETE" });
    toast(t("bookDeleted"), "ok");
    S.bookId = null;
    S.detail = null;
    S.pages = [];
    await refreshBooks();
    renderDetail();
  }

  async function deleteAllBooks() {
    if (!S.books.length) return;
    if (!window.confirm(t("removeAllConfirm"))) return;
    await api("/api/books", { method: "DELETE" });
    toast(t("allBooksDeleted"), "ok");
    S.bookId = null;
    S.detail = null;
    S.pages = [];
    await refreshBooks();
    renderDetail();
  }

  // -- detail -------------------------------------------------------------
  function renderDetail() {
    if (!S.detail) {
      $("emptyState").hidden = false;
      $("detail").hidden = true;
      return;
    }
    $("emptyState").hidden = true;
    $("detail").hidden = false;
    const book = S.detail.book;
    const progress = S.detail.progress;
    const status = progress.job_status || book.status || "—";
    $("bookTitle").textContent = book.title;
    $("bookMeta").textContent = `${book.page_count} pages · ${book.language} · ${book.file_name}`;
    $("statusChip").textContent = status;
    $("statusChip").className = `chip ${status === "done" ? "active" : status === "running" ? "warn" : ""}`;
    $("progressPct").textContent = String(progress.percent);
    $("progressFill").style.width = `${progress.percent}%`;
    $("progressText").textContent = `${progress.current_page}/${progress.page_count}`;
    $("progressEta").textContent = progress.eta_s ? `${t("eta")} ${Math.ceil(progress.eta_s)}s` : "";
    $("statDone").textContent = String(progress.done);
    $("statFlagged").textContent = String(progress.flagged);
    $("statFailed").textContent = String(progress.failed);
  }

  async function refreshDetail() {
    if (!S.bookId) return;
    S.detail = await api(`/api/books/${encodeURIComponent(S.bookId)}`);
    renderDetail();
    if (S.activeTab === "events") await loadEvents();
  }

  function renderPages() {
    const list = $("pageList");
    list.innerHTML = "";
    if (!S.pages.length) {
      const empty = document.createElement("div");
      empty.className = "page-row muted";
      empty.textContent = t("noPages");
      list.appendChild(empty);
      return;
    }
    S.pages.forEach((page) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "page-row" + (page.page_no === S.pageNo ? " selected" : "");
      row.innerHTML = `
        <span class="page-no">${String(page.page_no).padStart(4, "0")}</span>
        <span class="dot dot-${page.status}"></span>
        <span>${page.source === "embedded" ? "text" : page.source === "ocr" ? "OCR" : "—"}</span>
        <span class="page-chars">${page.char_count || 0}</span>`;
      row.addEventListener("click", () => selectPage(page.page_no));
      list.appendChild(row);
    });
  }

  async function loadPages() {
    if (!S.bookId) return;
    S.pages = await api(`/api/books/${encodeURIComponent(S.bookId)}/pages`);
    renderPages();
  }

  async function selectPage(pageNo) {
    if (!S.bookId) return;
    S.pageNo = pageNo;
    renderPages();
    const page = await api(`/api/books/${encodeURIComponent(S.bookId)}/pages/${pageNo}`);
    $("previewTitle").textContent = `${t("preview")} · ${pageNo}`;
    $("previewConfidence").textContent = page.confidence == null ? "—" : String(page.confidence);
    $("previewText").textContent = page.text_clean || page.error || "";
    const box = $("previewImageBox");
    box.innerHTML = "";
    if (page.raster_path) {
      const image = document.createElement("img");
      image.alt = `page ${pageNo}`;
      image.src = `/api/books/${encodeURIComponent(S.bookId)}/pages/${pageNo}/image`;
      box.appendChild(image);
    } else {
      const hint = document.createElement("span");
      hint.className = "muted";
      hint.textContent = t("noPreview");
      box.appendChild(hint);
    }
  }

  // -- search debug panel -------------------------------------------------
  function rankBadge(label, value) {
    const badge = document.createElement("span");
    badge.className = "rank-badge" + (value == null ? " empty" : "");
    badge.textContent = `${label}: ${value == null ? "—" : (typeof value === "number" ? value.toFixed(4) : value)}`;
    return badge;
  }

  function renderSearchResults() {
    const holder = $("searchResult");
    if (!holder) return;
    holder.innerHTML = "";
    if (!S.searchResult) {
      holder.textContent = t("noSearch");
      return;
    }
    const result = S.searchResult;
    const summary = document.createElement("div");
    summary.className = "search-summary";
    summary.textContent = `${result.query} · ${t("searchStats", result.counts)}`;
    holder.appendChild(summary);
    if (result.models) {
      const models = document.createElement("div");
      models.className = "search-models muted";
      models.textContent = `${result.models.embedding || "—"} · ${result.models.reranker || "—"}`;
      holder.appendChild(models);
    }
    result.hits.forEach((hit, index) => {
      const card = document.createElement("article");
      card.className = "search-hit";
      const head = document.createElement("div");
      head.className = "search-hit-head";
      const title = document.createElement("strong");
      title.textContent = `${index + 1}. ${hit.book_title || hit.book_id} · ${hit.page_start}-${hit.page_end}`;
      const badges = document.createElement("div");
      badges.className = "rank-badges";
      badges.append(
        rankBadge(t("denseRank"), hit.dense_rank),
        rankBadge(t("bm25Rank"), hit.bm25_rank),
        rankBadge(t("fusedRank"), hit.fused_rank),
        rankBadge(t("rerankScore"), hit.rerank_score),
      );
      head.append(title, badges);
      const text = document.createElement("div");
      text.className = "search-text";
      text.dir = "auto";
      text.textContent = hit.text || "";
      card.append(head, text);
      if (hit.parent_text) {
        const details = document.createElement("details");
        const caption = document.createElement("summary");
        caption.textContent = t("parentWindow");
        const parent = document.createElement("div");
        parent.className = "search-text parent";
        parent.dir = "auto";
        parent.textContent = hit.parent_text;
        details.append(caption, parent);
        card.appendChild(details);
      }
      holder.appendChild(card);
    });
  }

  async function runSearch() {
    const query = $("searchQuery").value.trim();
    if (!query) { toast(lang === "ar" ? "اكتب سؤالًا أولًا" : "Type a question first", "error"); return; }
    const body = {
      query,
      top_k: Number($("searchTopK").value) || 8,
      explain: true,
    };
    if ($("searchScope").value === "current" && S.bookId) body.book_ids = [S.bookId];
    const from = Number($("searchPageFrom").value);
    const to = Number($("searchPageTo").value);
    if (from > 0) body.page_from = from;
    if (to > 0) body.page_to = to;
    $("searchResult").textContent = t("searching");
    try {
      S.searchResult = await api("/api/retrieve", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      renderSearchResults();
    } catch (error) {
      S.searchResult = null;
      $("searchResult").textContent = "";
      toast(error.message, "error");
    }
  }

  // -- grounded chat, citations and memory -------------------------------
  function renderConversations() {
    const select = $("conversationSelect");
    if (!select) return;
    select.innerHTML = "";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = t("newConversation");
    select.appendChild(blank);
    S.conversations.forEach((conversation) => {
      const option = document.createElement("option");
      option.value = conversation.id;
      option.textContent = conversation.title || t("conversation");
      select.appendChild(option);
    });
    select.value = S.conversationId || "";
  }

  async function loadConversations() {
    S.conversations = await api("/api/conversations");
    renderConversations();
  }

  async function loadMessages(conversationId) {
    if (!conversationId) {
      S.chatMessages = [];
      renderChatLog();
      return;
    }
    S.chatMessages = await api(`/api/conversations/${encodeURIComponent(conversationId)}/messages`);
    renderChatLog();
  }

  function renderMessageText(node, text, sources = []) {
    node.innerHTML = "";
    const byLabel = new Map((sources || []).map((source) => [source.label, source]));
    const pattern = /\[([^\]\n]+?,\s*p\.\d+(?:[-–]\d+)?)\]/g;
    let last = 0;
    let match;
    while ((match = pattern.exec(text || "")) !== null) {
      if (match.index > last) {
        node.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = "citation";
      button.textContent = match[0];
      button.title = t("citationOpen");
      const source = byLabel.get(match[1]);
      button.addEventListener("click", () => openCitation(match[1], source));
      node.appendChild(button);
      last = pattern.lastIndex;
    }
    if (last < (text || "").length) {
      node.appendChild(document.createTextNode(text.slice(last)));
    }
  }

  function appendChatMessage(role, text, sources = [], language = "auto") {
    const log = $("chatLog");
    if (!log.querySelector(".chat-message")) log.innerHTML = "";
    const article = document.createElement("article");
    article.className = `chat-message ${role}`;
    const meta = document.createElement("div");
    meta.className = "chat-meta";
    meta.textContent = role === "user" ? t("you") : t("assistant");
    const content = document.createElement("div");
    content.className = "chat-content";
    content.dir = language === "ar" ? "rtl" : language === "en" ? "ltr" : "auto";
    renderMessageText(content, text, sources);
    article.append(meta, content);
    log.appendChild(article);
    log.scrollTop = log.scrollHeight;
    return { node: article, content };
  }

  function renderChatLog() {
    const log = $("chatLog");
    if (!log) return;
    log.innerHTML = "";
    if (!S.chatMessages.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = t("noChat");
      log.appendChild(empty);
      return;
    }
    S.chatMessages.forEach((message) => {
      appendChatMessage(
        message.role, message.content, message.sources || [], message.language || "auto"
      );
    });
  }

  function normalizedText(value) {
    return String(value || "").replace(/\s+/g, " ").trim();
  }

  function highlightInto(node, raw, quote) {
    node.innerHTML = "";
    const full = normalizedText(raw);
    const needle = normalizedText(quote).slice(0, 180);
    const index = needle ? full.indexOf(needle) : -1;
    if (index < 0) {
      node.textContent = full;
      return;
    }
    node.appendChild(document.createTextNode(full.slice(0, index)));
    const mark = document.createElement("mark");
    mark.textContent = full.slice(index, index + needle.length);
    node.appendChild(mark);
    node.appendChild(document.createTextNode(full.slice(index + needle.length)));
  }

  async function openCitation(label, source) {
    const cited = source || S.lastSources[label];
    if (!cited) { toast(t("citationOpen"), "error"); return; }
    S.lastCitation = cited;
    $("citationPanel").hidden = false;
    $("citationTitle").textContent = label;
    const imageBox = $("citationImage");
    imageBox.innerHTML = "";
    if (cited.book_id && cited.page_start) {
      const image = document.createElement("img");
      image.alt = label;
      image.src = `/api/books/${encodeURIComponent(cited.book_id)}/pages/${cited.page_start}/image`;
      imageBox.appendChild(image);
      try {
        const page = await api(
          `/api/books/${encodeURIComponent(cited.book_id)}/pages/${cited.page_start}`
        );
        if (S.lastCitation !== cited) return;
        const rangeText = cited.page_end > cited.page_start ? cited.text : "";
        highlightInto(
          $("citationText"),
          rangeText || page.text_clean || page.text_original || cited.text,
          cited.text
        );
      } catch (error) {
        $("citationText").textContent = cited.text || "";
      }
    } else {
      $("citationText").textContent = cited.text || "";
    }
  }

  async function newConversation() {
    try {
      const conversation = await api("/api/conversations", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: t("newConversation"), book_ids: S.bookId ? [S.bookId] : [] }),
      });
      S.conversationId = conversation.id;
      await loadConversations();
      await loadMessages(conversation.id);
    } catch (error) { toast(error.message, "error"); }
  }

  async function sendChat() {
    if (S.chatBusy) return;
    const query = $("chatQuery").value.trim();
    if (!query) { toast(lang === "ar" ? "اكتب سؤالًا أولًا" : "Type a question first", "error"); return; }
    const body = {
      query,
      mode: $("chatMode").value,
      language: "auto",
    };
    if (S.conversationId) body.conversation_id = S.conversationId;
    if ($("chatScope").value === "current" && S.bookId) body.book_ids = [S.bookId];
    const from = Number($("chatPageFrom").value);
    const to = Number($("chatPageTo").value);
    if (from > 0) body.page_from = from;
    if (to > 0) body.page_to = to;
    const memoryDraft = $("memoryDraft").value.trim();
    if (memoryDraft) {
      body.memory_content = memoryDraft;
      body.memory_kind = $("memoryDraftKind").value;
    }

    S.chatBusy = true;
    $("chatSendBtn").disabled = true;
    $("chatSendBtn").textContent = t("sending");
    appendChatMessage("user", query);
    const answer = appendChatMessage("assistant", "");
    let raw = "";
    let finalEvent = null;
    const log = $("chatLog");

    const handleEvent = (event) => {
      if (event.type === "delta") {
        raw += String(event.text || "");
        answer.content.textContent += String(event.text || "");
        log.scrollTop = log.scrollHeight;
      } else if (event.type === "done") {
        finalEvent = event;
        raw = String(event.answer || raw);
        S.lastSources = Object.fromEntries((event.sources || []).map((source) => [source.label, source]));
        renderMessageText(answer.content, raw, event.sources || []);
        if (event.conversation_id) S.conversationId = event.conversation_id;
        if ((event.warnings || []).length) {
          const note = document.createElement("div");
          note.className = "chat-warnings";
          note.textContent = `${t("warnings")}: ${event.warnings.join(" · ")}`;
          answer.node.appendChild(note);
        }
      } else if (event.type === "memory_saved") {
        $("memoryDraft").value = "";
        toast(t("memorySaved"), "ok");
        loadMemories().catch(() => {});
      } else if (event.type === "error") {
        toast(event.message || "Ollama error", "error");
        answer.content.textContent += `\n${event.message || ""}`;
      }
    };

    try {
      const response = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try { detail = (await response.json()).detail || detail; } catch (_) { /* keep HTTP error */ }
        throw new Error(detail);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      const consume = () => {
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        parts.forEach((part) => {
          const line = part.split("\n").find((item) => item.startsWith("data:"));
          if (!line) return;
          try { handleEvent(JSON.parse(line.slice(5).trim())); } catch (_) { /* ignore keep-alive */ }
        });
      };
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");
        consume();
      }
      buffer += decoder.decode();
      consume();
      if (raw) {
        S.chatMessages.push({
          role: "user", content: query, sources: [], language: "auto",
        });
        S.chatMessages.push({
          role: "assistant", content: raw,
          sources: (finalEvent && finalEvent.sources) || [],
          language: (finalEvent && finalEvent.language) || "auto",
        });
      }
      $("chatQuery").value = "";
      await loadConversations();
      if (S.conversationId) await loadMessages(S.conversationId);
    } catch (error) {
      toast(error.message, "error");
      answer.content.textContent += `\n${error.message}`;
    } finally {
      S.chatBusy = false;
      $("chatSendBtn").disabled = false;
      $("chatSendBtn").textContent = t("send");
      log.scrollTop = log.scrollHeight;
    }
  }

  async function loadMemories() {
    S.memories = await api("/api/memories");
    renderMemories();
  }

  function renderMemories() {
    const holder = $("memoryList");
    if (!holder) return;
    holder.innerHTML = "";
    if (!S.memories.length) {
      holder.textContent = t("noMemories");
      return;
    }
    S.memories.forEach((memory) => {
      const row = document.createElement("article");
      row.className = "memory-row";
      const meta = document.createElement("div");
      meta.className = "memory-meta";
      const chip = document.createElement("span");
      chip.className = "chip subtle";
      chip.textContent = memory.kind === "preference" ? t("preference") : t("fact");
      meta.appendChild(chip);
      const actions = document.createElement("span");
      const edit = document.createElement("button");
      edit.type = "button"; edit.className = "ghost small"; edit.textContent = t("edit");
      edit.addEventListener("click", async () => {
        const next = window.prompt(t("edit"), memory.content);
        if (!next || !next.trim()) return;
        try {
          await api(`/api/memories/${encodeURIComponent(memory.id)}`, {
            method: "PATCH", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: next.trim(), kind: memory.kind }),
          });
          toast(t("memoryUpdated"), "ok");
          await loadMemories();
        } catch (error) { toast(error.message, "error"); }
      });
      const remove = document.createElement("button");
      remove.type = "button"; remove.className = "ghost small danger"; remove.textContent = t("delete");
      remove.addEventListener("click", async () => {
        try {
          await api(`/api/memories/${encodeURIComponent(memory.id)}`, { method: "DELETE" });
          toast(t("memoryDeleted"), "ok");
          await loadMemories();
        } catch (error) { toast(error.message, "error"); }
      });
      actions.append(edit, remove);
      meta.appendChild(actions);
      const text = document.createElement("div");
      text.className = "memory-content";
      text.dir = "auto";
      text.textContent = memory.content;
      row.append(meta, text);
      holder.appendChild(row);
    });
  }

  async function loadModelsAndSettings() {
    let models;
    try {
      models = await api("/api/models");
    } catch (error) {
      models = { installed: [], current: "" };
      toast(error.message, "error");
    }
    const settings = await api("/api/config/generation");
    S.models = models;
    S.settings = settings;
    const select = $("modelSelect");
    select.innerHTML = "";
    const installed = models.installed || [];
    if (!installed.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = t("noModels");
      select.appendChild(option);
    }
    installed.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    });
    const current = settings.model || models.current || "";
    if (current && !installed.includes(current)) {
      const option = document.createElement("option");
      option.value = current;
      option.textContent = current;
      select.appendChild(option);
    }
    select.value = current;
    $("settingTemperature").value = settings.temperature ?? 0.2;
    $("settingContext").value = settings.context_tokens ?? 8192;
    $("settingHistory").value = settings.history_turns ?? 6;
    $("settingsInfo").textContent = `${t("llmModel")}: ${current || "—"} · ${installed.length}`;
  }

  async function saveSettings() {
    const body = {
      model: $("modelSelect").value || null,
      temperature: Number($("settingTemperature").value),
      context_tokens: Number($("settingContext").value),
      history_turns: Number($("settingHistory").value),
    };
    try {
      await api("/api/config/generation", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      toast(t("settingsSaved"), "ok");
      await loadModelsAndSettings();
    } catch (error) { toast(error.message, "error"); }
  }

  // -- benchmark ----------------------------------------------------------
  function renderBenchPageChips() {
    const holder = $("benchPages");
    holder.innerHTML = "";
    const input = document.createElement("input");
    input.id = "benchPageInput";
    input.placeholder = t("pageNumbers");
    input.style.cssText = "background:var(--surface-2);border:1px solid var(--line);border-radius:6px;padding:7px 9px;min-width:210px";
    const add = document.createElement("button");
    add.type = "button";
    add.className = "ghost small";
    add.textContent = t("addPages");
    add.addEventListener("click", () => {
      (input.value.match(/\d+/g) || []).forEach((value) => {
        const number = Number(value);
        if (number >= 1 && number <= (S.detail?.book.page_count || 0)) S.benchSelection.add(number);
      });
      input.value = "";
      if (S.benchSelection.size > 10) {
        const keep = [...S.benchSelection].slice(0, 10);
        S.benchSelection = new Set(keep);
        toast(lang === "ar" ? "الحد الأقصى ١٠ صفحات" : "Maximum 10 pages", "error");
      }
      renderBenchPageChips();
    });
    holder.append(input, add);
    [...S.benchSelection].sort((a, b) => a - b).forEach((number) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "page-chip on";
      chip.textContent = String(number);
      chip.addEventListener("click", () => { S.benchSelection.delete(number); renderBenchPageChips(); });
      holder.appendChild(chip);
    });
  }

  function renderBenchmark() {
    const holder = $("benchResult");
    holder.innerHTML = "";
    if (!S.benchmark || !S.benchmark.engines) {
      holder.textContent = t("noBench");
      return;
    }
    const engines = Object.values(S.benchmark.engines);
    const table = document.createElement("table");
    table.className = "bench-table";
    const head = document.createElement("tr");
    head.appendChild(document.createElement("th")).textContent = t("page");
    engines.forEach((engine) => {
      const th = document.createElement("th");
      th.textContent = `${engine.engine} · ${engine.mean_duration_s ?? "—"}s · ${engine.total_chars ?? 0} ch`;
      if (engine.available) {
        const use = document.createElement("button");
        use.type = "button";
        use.className = "solid small";
        use.style.marginInlineStart = "8px";
        use.textContent = t("useEngine");
        use.addEventListener("click", async () => {
          try {
            await api(`/api/books/${encodeURIComponent(S.bookId)}/ocr-engine`, {
              method: "PUT", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ engine: engine.engine }),
            });
            toast(`${t("useEngine")}: ${engine.engine}`, "ok");
            await refreshBooks(); await refreshDetail();
          } catch (error) { toast(error.message, "error"); }
        });
        th.appendChild(use);
      }
      head.appendChild(th);
    });
    table.appendChild(head);
    S.benchmark.page_numbers.forEach((number) => {
      const row = document.createElement("tr");
      row.appendChild(document.createElement("td")).textContent = String(number);
      engines.forEach((engine) => {
        const cell = document.createElement("td");
        const entry = engine.pages[String(number)];
        if (!entry || entry.error) {
          cell.textContent = entry?.error || engine.error || "—";
          cell.className = "muted";
        } else {
          const time = document.createElement("div");
          time.className = "time";
          time.textContent = `${entry.duration_s}s · ${entry.confidence ?? "—"} · ${entry.chars} ch`;
          const sample = document.createElement("div");
          sample.className = "sample";
          sample.dir = "auto";
          sample.textContent = entry.text;
          cell.append(time, sample);
        }
        row.appendChild(cell);
      });
      table.appendChild(row);
    });
    holder.appendChild(table);
  }

  async function loadBenchmark() {
    if (!S.bookId) return;
    S.benchmark = await api(`/api/ocr/benchmark/latest?book_id=${encodeURIComponent(S.bookId)}`);
    renderBenchmark();
  }

  async function loadEvents() {
    const job = S.detail?.jobs?.[0];
    if (!job) { $("eventList").textContent = t("noEvents"); return; }
    const data = await api(`/api/jobs/${encodeURIComponent(job.id)}`);
    const holder = $("eventList");
    holder.innerHTML = "";
    if (!data.events.length) holder.textContent = t("noEvents");
    data.events.slice().reverse().forEach((event) => {
      const row = document.createElement("div");
      row.className = `event-row ${event.level}`;
      const time = document.createElement("time");
      time.textContent = event.ts.replace("T", " ").slice(0, 19);
      const message = document.createElement("span");
      message.className = "message";
      message.textContent = event.message;
      row.append(time, message);
      holder.appendChild(row);
    });
  }

  // -- events wiring ------------------------------------------------------
  function selectBook(id) {
    S.bookId = id;
    S.pageNo = null;
    S.benchSelection = new Set();
    S.benchmark = null;
    S.searchResult = null;
    S.overview = null;
    loadOverview();
    if ($("searchResult")) renderSearchResults();
    renderBooks();
    $("previewText").textContent = "";
    $("previewImageBox").innerHTML = "";
    Promise.all([refreshDetail(), loadPages(), loadBenchmark()]).catch((e) => toast(e.message, "error"));
  }

  function renderOverview() {
    const card = $("sourceOverview");
    if (!card) return;
    if (!S.overview) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    $("overviewSummary").textContent = S.overview.summary || "";
    $("overviewSummary").dir = S.overview.language === "ar" ? "rtl" : "ltr";
    $("overviewStatus").textContent = S.overview.status === "ocr_pending"
      ? t("overviewPending")
      : S.overview.status === "fallback" ? t("overviewFallback") : t("overviewReady");
    const holder = $("overviewQuestions");
    holder.innerHTML = "";
    (S.overview.questions || []).forEach((question) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "overview-question";
      button.textContent = question;
      button.dir = S.overview.language === "ar" ? "rtl" : "ltr";
      button.addEventListener("click", () => {
        S.activeTab = "chat";
        document.querySelectorAll(".tab").forEach((tab) => {
          tab.classList.toggle("active", tab.dataset.tab === "chat");
        });
        document.querySelectorAll(".tab-panel").forEach((panel) => {
          panel.hidden = panel.id !== "tab-chat";
        });
        $("chatQuery").value = question;
        $("chatQuery").focus();
      });
      holder.appendChild(button);
    });
  }

  async function loadOverview() {
    if (!S.bookId) return;
    const card = $("sourceOverview");
    if (card) {
      card.hidden = false;
      $("overviewSummary").textContent = t("analyzingSource");
      $("overviewStatus").textContent = "…";
      $("overviewQuestions").innerHTML = "";
    }
    try {
      S.overview = await api(`/api/books/${encodeURIComponent(S.bookId)}/overview`);
      renderOverview();
    } catch (error) {
      if (card) card.hidden = true;
      toast(error.message, "error");
    }
  }

  async function uploadFiles(files) {
    for (const file of files) {
      const body = new FormData();
      body.append("file", file);
      body.append("engine", $("engineSelect").value);
      try {
        const result = await api("/api/books/import", { method: "POST", body });
        S.bookId = result.book.id;
        toast(t("imported", { name: file.name }), "ok");
      } catch (error) { toast(error.message, "error"); }
    }
    await refreshBooks();
    if (S.bookId) await selectBook(S.bookId);
    $("fileInput").value = "";
  }

  function wire() {
    $("langBtn").addEventListener("click", () => {
      lang = lang === "ar" ? "en" : "ar";
      localStorage.setItem("localrag.lang", lang);
      applyLang(); renderBooks(); renderBenchmark();
      api("/api/config/ui", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: lang }),
      }).catch(() => {});
    });
    $("themeBtn").addEventListener("click", () => {
      theme = theme === "dark" ? "light" : "dark";
      localStorage.setItem("localrag.theme", theme);
      applyTheme();
      api("/api/config/ui", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ theme }),
      }).catch(() => {});
    });

    const dropzone = $("dropzone");
    ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
      event.preventDefault(); dropzone.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
      event.preventDefault(); dropzone.classList.remove("dragover");
    }));
    dropzone.addEventListener("drop", (event) => uploadFiles([...event.dataTransfer.files]));
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") $("fileInput").click();
    });
    $("chooseBtn").addEventListener("click", (event) => { event.stopPropagation(); $("fileInput").click(); });
    $("fileInput").addEventListener("change", (event) => uploadFiles([...event.target.files]));

    $("pauseBtn").addEventListener("click", () => action("pause"));
    $("resumeBtn").addEventListener("click", () => action("resume"));
    $("cancelBtn").addEventListener("click", () => action("cancel"));
    $("deleteBookBtn").addEventListener("click", () => deleteCurrentBook().catch((e) => toast(e.message, "error")));
    $("removeAllBtn").addEventListener("click", () => deleteAllBooks().catch((e) => toast(e.message, "error")));
    $("retryBtn").addEventListener("click", async () => {
      try {
        const result = await api(`/api/books/${encodeURIComponent(S.bookId)}/retry`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ statuses: ["failed", "flagged"] }),
        });
        toast(result.message || t("retryQueued"), result.ok ? "ok" : "");
        await refreshBooks(); await refreshDetail();
      } catch (error) { toast(error.message, "error"); }
    });

    $("searchBtn").addEventListener("click", () => { runSearch().catch((e) => toast(e.message, "error")); });
    $("searchQuery").addEventListener("keydown", (event) => {
      if (event.key === "Enter") runSearch().catch((e) => toast(e.message, "error"));
    });

    $("chatSendBtn").addEventListener("click", () => { sendChat().catch((e) => toast(e.message, "error")); });
    $("chatQuery").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendChat().catch((e) => toast(e.message, "error"));
      }
    });
    $("newConversationBtn").addEventListener("click", () => { newConversation(); });
    $("conversationSelect").addEventListener("change", (event) => {
      S.conversationId = event.target.value || null;
      loadMessages(S.conversationId).catch((e) => toast(e.message, "error"));
    });
    $("citationClose").addEventListener("click", () => {
      $("citationPanel").hidden = true;
      S.lastCitation = null;
    });
    $("memoryAddBtn").addEventListener("click", async () => {
      const content = $("memoryContent").value.trim();
      if (!content) return;
      try {
        await api("/api/memories", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content, kind: $("memoryKind").value }),
        });
        $("memoryContent").value = "";
        toast(t("memorySaved"), "ok");
        await loadMemories();
      } catch (error) { toast(error.message, "error"); }
    });
    $("saveSettingsBtn").addEventListener("click", () => { saveSettings(); });

    $("benchBtn").addEventListener("click", async () => {
      if (!S.benchSelection.size) { toast(lang === "ar" ? "اختر صفحة أولًا" : "Pick a page first", "error"); return; }
      try {
        const result = await api("/api/ocr/benchmark", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ book_id: S.bookId, pages: [...S.benchSelection] }),
        });
        S.benchmark = result;
        renderBenchmark();
        toast(t("benchmarkDone"), "ok");
      } catch (error) { toast(error.message, "error"); }
    });

    document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
      S.activeTab = tab.dataset.tab;
      document.querySelectorAll(".tab").forEach((node) => node.classList.toggle("active", node === tab));
      document.querySelectorAll(".tab-panel").forEach((panel) => { panel.hidden = panel.id !== `tab-${S.activeTab}`; });
      if (S.activeTab === "events") loadEvents().catch((e) => toast(e.message, "error"));
      if (S.activeTab === "search") renderSearchResults();
      if (S.activeTab === "chat") {
        loadConversations()
          .then(() => loadMessages(S.conversationId))
          .catch((e) => toast(e.message, "error"));
      }
      if (S.activeTab === "memory") loadMemories().catch((e) => toast(e.message, "error"));
      if (S.activeTab === "settings") loadModelsAndSettings().catch((e) => toast(e.message, "error"));
    }));
  }

  async function action(name) {
    const job = S.detail?.jobs?.[0];
    if (!job) return;
    try {
      await api(`/api/jobs/${encodeURIComponent(job.id)}/${name}`, { method: "POST" });
      await refreshBooks(); await refreshDetail();
    } catch (error) { toast(error.message, "error"); }
  }

  async function tick() {
    if (S.busy) return;
    S.busy = true;
    try {
      const id = await refreshBooks();
      if (id) {
        await refreshDetail();
        if (++S.tick % 3 === 0) await loadPages();
        if (
          S.detail?.progress?.job_status === "done"
          && S.overview?.status === "ocr_pending"
          && S.tick % 5 === 0
        ) {
          await loadOverview();
        }
      }
    } catch (error) { /* transient network hiccup; next tick retries */ }
    finally { S.busy = false; }
  }

  applyLang();
  applyTheme();
  wire();
  refreshHealth();
  refreshBooks().then((id) => { if (id) selectBook(id); }).catch((e) => toast(e.message, "error"));
  setInterval(tick, 1500);
  setInterval(refreshHealth, 15000);
})();
