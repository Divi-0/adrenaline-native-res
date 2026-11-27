// Draws the list pages from data.js: <main data-page="games"> as a card per game,
// <main data-page="releases"> as a timeline of releases.
(function () {
  const NR = window.NR, main = document.querySelector("main[data-page]");
  if (!NR || !main) return;
  const LABEL = { tested: "Tested", partial: "Partial", broken: "Broken" }, RANK = { tested: 0, partial: 1, broken: 2 };
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const dash = '<span aria-hidden="true">–</span><span class="sr-only">not measured</span>';
  const share = (v, of) => v == null ? 0 : Math.min(1, v / of);
  const color = s => s >= .97 ? "#4cc673" : s >= .6 ? "#e0ae3a" : "#f07070";
  const day = d => new Date(d + "T00:00:00").toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });

  // a frame-rate bar: filled to the average, a tick at the 1% low
  function bar(g, where) {
    if (g.fps == null) return `<div class="fpsbar" role="img" aria-label="Frame rate not measured"></div>`;
    const s = share(g.fps, g.of);
    return `<div class="fpsbar" role="img" aria-label="${esc(g.fps)} of ${esc(g.of)} frames a second${where ? ", " + esc(where) : ""}" style="--st: ${color(s)}">
      <i style="width: ${100 * s}%"></i>${g.low != null ? `<span class="low" style="left: ${100 * share(g.low, g.of)}%"></span>` : ""}</div>`;
  }

  /* ---------------------------------------------------------------- games */
  // a card per game: its state and title ID, its name, the frame-rate bar, the two figures, where it stands
  function games() {
    const box = document.getElementById("gt"), q = document.getElementById("q"), empty = document.getElementById("gt-empty");
    function render() {
      const terms = q.value.toLowerCase().split(/\s+/).filter(Boolean);
      const rows = NR.games.filter(g => terms.every(t => `${g.title} ${g.id} ${g.region} ${LABEL[g.status]} ${g.note}`.toLowerCase().includes(t)))
        .sort((x, y) => RANK[x.status] - RANK[y.status] || x.title.localeCompare(y.title));
      box.innerHTML = `<div class="grid">${rows.map(g => `<article class="card m-nolift span-4 gcard">
          <p class="gcard__top"><span class="state" data-state="${esc(g.status)}">${LABEL[g.status]}</span><span class="id">${esc(g.id)}</span></p>
          <h2 class="gcard__name">${esc(g.title)}</h2>
          ${bar(g)}
          <p class="fpsrow"><span>Avg FPS <b>${g.fps == null ? "–" : esc(g.fps)}</b> of ${esc(g.of)}</span><span>1% low ${g.low == null ? dash : `<b>${esc(g.low)}</b>`}</span></p>
          <p class="gcard__note">${esc(g.note)}</p>
        </article>`).join("")}</div>`;
      empty.hidden = rows.length > 0;
    }
    q.addEventListener("input", render);
    render();
  }

  /* ------------------------------------------------------------- releases */
  // a timeline: version, date and download in a rail on the left, the release beside it, its changes as points
  function releases() {
    const host = document.getElementById("releases");
    const word = r => r.was ? (r.was === r.now ? LABEL[r.now] : `${LABEL[r.was]}, now ${LABEL[r.now].toLowerCase()}`)
      : (r.now === "broken" ? "New, not playable yet" : `New, ${LABEL[r.now].toLowerCase()}`);
    // a game's pictures: img/<x>.jpg is its tile; img/<x>-native.jpg and img/<x>-sony.jpg the full-size pair
    const pair = (img, side) => img.replace(/\.jpg$/, `-${side}.jpg`);
    function card(r) {
      const g = NR.games.find(x => x.id === r.id) || { title: r.id };
      return `<article class="card card--row card--open m-nolift">
        ${g.img ? `<button class="panel panel--top zoom" type="button" data-native="${esc(pair(g.img, "native"))}" data-sony="${esc(pair(g.img, "sony"))}" data-title="${esc(g.title)}" aria-label="Show ${esc(g.title)} at full size"><img class="frame__img" src="${esc(g.img)}" alt="${esc(g.title)}, drawn by NativeRes" loading="lazy" data-source="client" data-subject="${esc(g.title)}"></button>` : ""}
        <h3>${esc(g.title)}</h3>
        <p><span class="state" data-state="${esc(r.now)}">${word(r)}</span></p>
        ${r.fps ? bar(r, r.scene) + `<p class="fpsrow"><span>Avg FPS <b>${esc(r.fps)}</b> of ${esc(r.of)}</span><span>${esc(r.scene || "")}</span></p>` : ""}
      </article>`;
    }

    host.outerHTML = NR.releases.map((r, n) => `<section class="section flow rel" id="${esc(r.tag)}"><div class="wrap">
        <div class="grid grid--top">
          <div class="span-3 tl-rail">
            <p class="rel-meta"><span class="ver">${esc(r.tag)}</span><time datetime="${esc(r.date)}">${day(r.date)}</time></p>
            <a class="link" href="${NR.repo}/releases/tag/${encodeURIComponent(r.tag)}">Get ${esc(r.tag)}</a>
          </div>
          <div class="span-9">
            <${n ? "h2" : "h1"} class="rel-title">${esc(r.title)}</${n ? "h2" : "h1"}>
            <p class="rel-sum">${esc(r.summary)}</p>
            <div class="tl-cards">${r.games.map(card).join("")}</div>
            <ul class="rel-points">${r.changes.map(c => `<li>${esc(c)}</li>`).join("")}</ul>
          </div>
        </div>
      </div></section>`).join("");
    if (location.hash) { const t = document.getElementById(decodeURIComponent(location.hash.slice(1))); if (t) t.scrollIntoView(); }

    // a click on a game's picture opens it at its full 960x544 with the line to drag between Stock
    // Adrenaline's picture and ours; the cross, Esc or a click beside the picture closes it
    const shot = document.createElement("dialog");
    shot.className = "shot";
    shot.innerHTML = `<div class="shot__pic" tabindex="-1" autofocus>
        <img class="shot__new" alt="" width="960" height="544">
        <div class="cmp-clip"><img class="shot__old" alt="" width="960" height="544"></div>
        <input class="cmp-range" type="range" min="0" max="100" value="50" aria-label="Move the line between Stock Adrenaline's picture and NativeRes's">
        <span class="cmp-line" aria-hidden="true"></span>
        <span class="cmp-knob" aria-hidden="true"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 5l-4 5 4 5M13 5l4 5-4 5"/></svg></span>
        <button class="shot__close" type="button" aria-label="Close the picture"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 4l10 10M14 4L4 14"/></svg></button>
      </div>
      <p class="shot__sides"><span>Stock Adrenaline · 480×272</span><span>NativeRes · 960×544</span></p>`;
    document.body.append(shot);
    const pic = shot.querySelector(".shot__pic"), range = shot.querySelector(".cmp-range");
    const fresh = shot.querySelector(".shot__new"), old = shot.querySelector(".shot__old");
    old.addEventListener("error", () => shot.classList.add("shot--plain"));   // no pair for this game: the plain picture
    let next = null;
    range.addEventListener("input", () => {
      if (next === null) requestAnimationFrame(() => { pic.style.setProperty("--pos", next + "%"); next = null; });
      next = range.value;
    });
    document.addEventListener("click", e => {
      const z = e.target.closest(".zoom");
      if (z) {
        shot.classList.remove("shot--plain");
        fresh.src = z.dataset.native; fresh.alt = `${z.dataset.title}, drawn by NativeRes at 960×544`;
        old.src = z.dataset.sony; old.alt = `${z.dataset.title}, the same frame in Stock Adrenaline, 480×272 scaled up`;
        range.value = 50; pic.style.setProperty("--pos", "50%");
        shot.showModal();
      } else if (e.target.closest(".shot__close") || e.target === shot) shot.close();
    });
  }

  (main.dataset.page === "games" ? games : releases)();
})();
