"use strict";

const els = {
  board: document.querySelector("#board"),
  loadSample: document.querySelector("#loadSample"),
  fileInput: document.querySelector("#fileInput"),
  prev: document.querySelector("#prev"),
  play: document.querySelector("#play"),
  next: document.querySelector("#next"),
  slider: document.querySelector("#plySlider"),
  plyText: document.querySelector("#plyText"),
  players: document.querySelector("#players"),
  rules: document.querySelector("#rules"),
  status: document.querySelector("#status"),
  moveText: document.querySelector("#moveText")
};

const coords = makeBoardCoords();
let game = emptyGame();
let ply = 0;
let timer = null;

function makeBoardCoords() {
  const map = new Map();
  const add = (q, r) => map.set(`${q},${r}`, { q, r });

  for (let q = -4; q <= 4; q += 1) {
    for (let r = -4; r <= 4; r += 1) {
      const s = -q - r;
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) <= 4) add(q, r);
    }
  }

  for (let d = 1; d <= 4; d += 1) {
    for (let r = -4; r <= -d; r += 1) add(4 + d, r);
    for (let r = d; r <= 4; r += 1) add(-4 - d, r);
    for (let q = -4; q <= -d; q += 1) add(q, 4 + d);
    for (let q = d; q <= 4; q += 1) add(q, -4 - d);
    for (let q = -4; q <= -d; q += 1) add(q, -4 - d - q);
    for (let q = d; q <= 4; q += 1) add(q, 4 + d - q);
  }

  return Array.from(map.values()).sort((a, b) => a.r - b.r || a.q - b.q);
}

function emptyGame() {
  return {
    start: null,
    moves: [],
    end: null,
    states: [Array(121).fill(".")]
  };
}

function parseJsonl(text) {
  const records = text
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => JSON.parse(line));

  const parsed = emptyGame();
  const start = records.find(record => record.type === "game_start");
  if (!start || typeof start.initial_cells !== "string") {
    throw new Error("Replay JSONL needs a game_start record with initial_cells.");
  }
  if (start.initial_cells.length !== coords.length) {
    throw new Error(`Expected ${coords.length} cells, found ${start.initial_cells.length}.`);
  }

  parsed.start = start;
  parsed.states = [start.initial_cells.split("")];
  for (const record of records) {
    if (record.type === "move") {
      const next = parsed.states[parsed.states.length - 1].slice();
      next[record.from] = ".";
      next[record.to] = String(record.player);
      parsed.moves.push(record);
      parsed.states.push(next);
    } else if (record.type === "game_end") {
      parsed.end = record;
    }
  }
  return parsed;
}

function boardPoint(coord) {
  const gap = 18;
  return {
    x: (coord.q + coord.r / 2) * gap,
    y: coord.r * gap * 0.8660254
  };
}

function renderBoard() {
  const points = coords.map(boardPoint);
  const minX = Math.min(...points.map(p => p.x)) - 14;
  const maxX = Math.max(...points.map(p => p.x)) + 14;
  const minY = Math.min(...points.map(p => p.y)) - 14;
  const maxY = Math.max(...points.map(p => p.y)) + 14;
  const state = game.states[ply] || game.states[0];
  const last = ply > 0 ? game.moves[ply - 1] : null;
  const path = last?.path?.map(id => points[id]).filter(Boolean) || [];

  els.board.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`);
  els.board.innerHTML = "";

  if (path.length > 1) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("class", "move-path");
    line.setAttribute("points", path.map(p => `${p.x},${p.y}`).join(" "));
    els.board.append(line);
  }

  points.forEach((point, id) => {
    const hole = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    const occupant = state[id];
    const classes = ["hole"];
    if (last?.from === id) classes.push("last-from");
    if (last?.to === id) classes.push("last-to");
    if (occupant === "0" || occupant === "1") classes.push(`piece-${occupant}`);
    hole.setAttribute("class", classes.join(" "));
    hole.setAttribute("cx", point.x);
    hole.setAttribute("cy", point.y);
    hole.setAttribute("r", occupant === "0" || occupant === "1" ? 6.4 : 4.2);
    hole.dataset.id = String(id);
    els.board.append(hole);
  });
}

function renderMeta() {
  const total = Math.max(0, game.states.length - 1);
  const move = ply > 0 ? game.moves[ply - 1] : null;
  els.slider.max = String(total);
  els.slider.value = String(ply);
  els.plyText.textContent = `${ply} / ${total}`;
  els.players.textContent = game.start ? `${game.start.p0} vs ${game.start.p1}` : "-";
  els.rules.textContent = game.start?.rule_profile || "-";
  els.status.textContent = game.end
    ? `${game.end.draw ? "Draw" : `Winner: ${game.end.winner}`} (${game.end.reason})`
    : game.start
      ? "Loaded"
      : "No game loaded";
  els.moveText.textContent = move ? `${move.ply}. P${move.player} ${move.move}` : "Initial position";
}

function render() {
  renderBoard();
  renderMeta();
}

function setGame(nextGame) {
  stop();
  game = nextGame;
  ply = 0;
  render();
}

async function loadSample() {
  const response = await fetch("sample-game.jsonl");
  if (!response.ok) throw new Error(`Could not load sample: ${response.status}`);
  setGame(parseJsonl(await response.text()));
}

function step(delta) {
  const max = game.states.length - 1;
  ply = Math.max(0, Math.min(max, ply + delta));
  render();
  if (ply >= max) stop();
}

function stop() {
  if (timer) window.clearInterval(timer);
  timer = null;
  els.play.textContent = "Play";
}

function togglePlay() {
  if (timer) {
    stop();
    return;
  }
  if (ply >= game.states.length - 1) ply = 0;
  els.play.textContent = "Pause";
  timer = window.setInterval(() => step(1), 700);
}

els.loadSample.addEventListener("click", () => {
  loadSample().catch(error => {
    els.status.textContent = error.message;
  });
});

els.fileInput.addEventListener("change", async event => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    setGame(parseJsonl(await file.text()));
  } catch (error) {
    els.status.textContent = error.message;
  }
});

els.prev.addEventListener("click", () => step(-1));
els.next.addEventListener("click", () => step(1));
els.play.addEventListener("click", togglePlay);
els.slider.addEventListener("input", event => {
  ply = Number(event.target.value);
  render();
});

render();
loadSample().catch(() => {});
