/* ==========================================================================
   Pharma Blister Digital Twin - showcase behaviour
   Zero dependencies. Every interactive control is pointer-, touch- and
   keyboard-operable, and degrades to static content when JS is unavailable.
   ========================================================================== */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ------------------------------------------------------------------ */
  /* Hero video: honour prefers-reduced-motion with a manual play control */
  /* ------------------------------------------------------------------ */
  (function heroVideo() {
    var video = document.getElementById("heroVideo");
    var play = document.getElementById("heroPlay");
    if (!video || !play) return;

    function showButton(label) {
      play.setAttribute("data-show", "1");
      play.querySelector("span").textContent = label;
    }
    function hideButton() {
      play.removeAttribute("data-show");
    }

    if (reduceMotion) {
      video.autoplay = false;
      video.loop = false;
      video.pause();
      showButton("Play recorded run");
    } else {
      // Some browsers refuse autoplay even when muted; offer a control if so.
      var attempt = video.play();
      if (attempt && typeof attempt.catch === "function") {
        attempt.catch(function () { showButton("Play recorded run"); });
      }
    }

    play.addEventListener("click", function () {
      if (video.paused) {
        video.play();
        hideButton();
      } else {
        video.pause();
        showButton("Play recorded run");
      }
    });

    video.addEventListener("playing", hideButton);
    video.addEventListener("pause", function () {
      if (!video.ended) showButton("Resume recorded run");
    });
  })();

  /* ------------------------------------------------------------------ */
  /* Asset comparison slider                                             */
  /* ------------------------------------------------------------------ */
  (function assetCompare() {
    var box = document.getElementById("assetCompare");
    var handle = document.getElementById("cmpHandle");
    var base = document.getElementById("cmpBase");
    var overlay = document.getElementById("cmpOverlay");
    var btnCrop = document.getElementById("viewCrop");
    var btnFull = document.getElementById("viewFull");
    if (!box || !handle || !base || !overlay) return;

    var pos = 50;
    var dragging = false;

    function clamp(n) { return n < 0 ? 0 : (n > 100 ? 100 : n); }

    function setPos(next) {
      pos = clamp(next);
      var pct = pos.toFixed(2) + "%";
      box.style.setProperty("--pos", pct);
      handle.setAttribute("aria-valuenow", String(Math.round(pos)));
      handle.setAttribute("aria-valuetext", Math.round(pos) + " percent asset v1");
    }

    function posFromClientX(clientX) {
      var rect = box.getBoundingClientRect();
      if (!rect.width) return pos;
      return ((clientX - rect.left) / rect.width) * 100;
    }

    function onPointerMove(ev) {
      if (!dragging) return;
      ev.preventDefault();
      setPos(posFromClientX(ev.clientX));
    }

    function endDrag(ev) {
      if (!dragging) return;
      dragging = false;
      if (handle.releasePointerCapture && ev && ev.pointerId !== undefined) {
        try { handle.releasePointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
      }
    }

    handle.addEventListener("pointerdown", function (ev) {
      dragging = true;
      if (handle.setPointerCapture) {
        try { handle.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
      }
      ev.preventDefault();
      handle.focus();
    });
    handle.addEventListener("pointermove", onPointerMove);
    handle.addEventListener("pointerup", endDrag);
    handle.addEventListener("pointercancel", endDrag);
    window.addEventListener("pointerup", endDrag);

    // Click anywhere on the frame to jump the divider there.
    box.addEventListener("pointerdown", function (ev) {
      if (ev.target === handle || handle.contains(ev.target)) return;
      setPos(posFromClientX(ev.clientX));
    });

    // Touch fallback for engines without Pointer Events.
    if (!window.PointerEvent) {
      handle.addEventListener("touchstart", function (ev) {
        dragging = true;
        ev.preventDefault();
      }, { passive: false });
      handle.addEventListener("touchmove", function (ev) {
        if (!dragging || !ev.touches.length) return;
        ev.preventDefault();
        setPos(posFromClientX(ev.touches[0].clientX));
      }, { passive: false });
      handle.addEventListener("touchend", function () { dragging = false; });
    }

    handle.addEventListener("keydown", function (ev) {
      var step = ev.shiftKey ? 10 : 2;
      var handled = true;
      switch (ev.key) {
        case "ArrowLeft":
        case "ArrowDown": setPos(pos - step); break;
        case "ArrowRight":
        case "ArrowUp": setPos(pos + step); break;
        case "Home": setPos(0); break;
        case "End": setPos(100); break;
        case "PageDown": setPos(pos - 20); break;
        case "PageUp": setPos(pos + 20); break;
        default: handled = false;
      }
      if (handled) ev.preventDefault();
    });

    /* crop <-> full render toggle */
    function setView(full) {
      base.src = full ? "./media/asset_v2.jpg" : "./media/asset_v2_crop.jpg";
      overlay.src = full ? "./media/asset_v1.jpg" : "./media/asset_v1_crop.jpg";
      box.classList.toggle("is-full", full);
      if (btnCrop) btnCrop.setAttribute("aria-pressed", full ? "false" : "true");
      if (btnFull) btnFull.setAttribute("aria-pressed", full ? "true" : "false");
    }
    if (btnCrop) btnCrop.addEventListener("click", function () { setView(false); });
    if (btnFull) btnFull.addEventListener("click", function () { setView(true); });

    setPos(50);
  })();

  /* ------------------------------------------------------------------ */
  /* Generic tablist: roving tabindex + arrow-key navigation             */
  /* ------------------------------------------------------------------ */
  function wireTablist(tabs, activate) {
    function select(index) {
      tabs.forEach(function (tab, i) {
        var on = i === index;
        tab.setAttribute("aria-selected", on ? "true" : "false");
        if (on) { tab.removeAttribute("tabindex"); } else { tab.setAttribute("tabindex", "-1"); }
      });
      activate(index, tabs[index]);
    }

    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () { select(i); });
      tab.addEventListener("keydown", function (ev) {
        var next = null;
        if (ev.key === "ArrowRight" || ev.key === "ArrowDown") next = (i + 1) % tabs.length;
        else if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") next = (i - 1 + tabs.length) % tabs.length;
        else if (ev.key === "Home") next = 0;
        else if (ev.key === "End") next = tabs.length - 1;
        if (next === null) return;
        ev.preventDefault();
        select(next);
        tabs[next].focus();
      });
    });

    return select;
  }

  /* ------------------------------------------------------------------ */
  /* Defect inspector tabs                                               */
  /* ------------------------------------------------------------------ */
  (function inspector() {
    var tabs = Array.prototype.slice.call(
      document.querySelectorAll('.tabs [role="tab"]')
    );
    if (!tabs.length) return;

    wireTablist(tabs, function (index) {
      tabs.forEach(function (tab, i) {
        var panel = document.getElementById(tab.getAttribute("aria-controls"));
        if (panel) panel.hidden = i !== index;
      });
    });
  })();

  /* ------------------------------------------------------------------ */
  /* JSON syntax highlighter (hand written, no dependencies)             */
  /* ------------------------------------------------------------------ */

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function jsonString(value) {
    // JSON.stringify handles escaping of quotes, backslashes and control chars.
    return JSON.stringify(value);
  }

  function isPrimitive(value) {
    return value === null || typeof value !== "object";
  }

  function allPrimitive(list) {
    for (var i = 0; i < list.length; i++) {
      if (!isPrimitive(list[i])) return false;
    }
    return true;
  }

  /**
   * Render a parsed JSON value as highlighted HTML.
   * Arrays whose members are all primitives stay on one line, which keeps the
   * detection rows readable instead of 60 lines of single numbers.
   */
  function renderValue(value, indent) {
    var pad = new Array(indent + 1).join("  ");
    var padIn = new Array(indent + 2).join("  ");
    var i, parts;

    if (value === null) {
      return '<span class="j-null">null</span>';
    }
    if (typeof value === "boolean") {
      return '<span class="j-bool">' + value + "</span>";
    }
    if (typeof value === "number") {
      return '<span class="j-num">' + escapeHtml(String(value)) + "</span>";
    }
    if (typeof value === "string") {
      return '<span class="j-str">' + escapeHtml(jsonString(value)) + "</span>";
    }

    if (Array.isArray(value)) {
      if (!value.length) return '<span class="j-punct">[]</span>';
      if (allPrimitive(value)) {
        parts = value.map(function (item) { return renderValue(item, 0); });
        return '<span class="j-punct">[</span>' + parts.join('<span class="j-punct">, </span>') +
          '<span class="j-punct">]</span>';
      }
      parts = [];
      for (i = 0; i < value.length; i++) {
        parts.push(padIn + renderValue(value[i], indent + 1));
      }
      return '<span class="j-punct">[</span>\n' + parts.join('<span class="j-punct">,</span>\n') +
        '\n' + pad + '<span class="j-punct">]</span>';
    }

    var keys = Object.keys(value);
    if (!keys.length) return '<span class="j-punct">{}</span>';
    parts = [];
    for (i = 0; i < keys.length; i++) {
      parts.push(
        padIn +
        '<span class="j-key">' + escapeHtml(jsonString(keys[i])) + "</span>" +
        '<span class="j-punct">: </span>' +
        renderValue(value[keys[i]], indent + 1)
      );
    }
    return '<span class="j-punct">{</span>\n' + parts.join('<span class="j-punct">,</span>\n') +
      '\n' + pad + '<span class="j-punct">}</span>';
  }

  /* ------------------------------------------------------------------ */
  /* Audit record viewer                                                 */
  /* ------------------------------------------------------------------ */
  (function auditViewer() {
    var holder = document.getElementById("audit-records");
    var out = document.getElementById("auditOut");
    if (!holder || !out) return;

    var records = holder.textContent
      .split("\n")
      .map(function (line) { return line.trim(); })
      .filter(function (line) { return line.length > 0; })
      .map(function (line) {
        try { return JSON.parse(line); } catch (err) { return null; }
      })
      .filter(function (record) { return record !== null; });

    if (!records.length) {
      out.innerHTML = "<code>No audit records available.</code>";
      return;
    }

    function show(index) {
      var record = records[Math.min(index, records.length - 1)];
      out.innerHTML = "<code>" + renderValue(record, 0) + "</code>";
    }

    var tabs = Array.prototype.slice.call(
      document.querySelectorAll('.recordbar [role="tab"]')
    );

    if (tabs.length) {
      var select = wireTablist(tabs, function (index, tab) {
        show(parseInt(tab.getAttribute("data-rec"), 10) || 0);
      });
      // Start on the blister.audit/1 record - the one with the verdict in it.
      var initial = 0;
      tabs.forEach(function (tab, i) {
        if (tab.getAttribute("aria-selected") === "true") initial = i;
      });
      select(initial);
    } else {
      show(0);
    }
  })();

  /* ------------------------------------------------------------------ */
  /* Nav: mark the section currently in view                             */
  /* ------------------------------------------------------------------ */
  (function activeSection() {
    if (!("IntersectionObserver" in window)) return;
    var links = {};
    Array.prototype.forEach.call(
      document.querySelectorAll(".navlinks a"),
      function (a) {
        var id = a.getAttribute("href").slice(1);
        if (id) links[id] = a;
      }
    );

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var link = links[entry.target.id];
        if (!link) return;
        if (entry.isIntersecting) {
          Object.keys(links).forEach(function (key) {
            links[key].style.color = "";
          });
          link.style.color = "var(--accent)";
        }
      });
    }, { rootMargin: "-56px 0px -70% 0px", threshold: 0 });

    Object.keys(links).forEach(function (id) {
      var section = document.getElementById(id);
      if (section) observer.observe(section);
    });
  })();
})();
