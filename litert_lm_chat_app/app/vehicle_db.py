"""Simulated CAN/OBD vehicle state + reminder/calendar persistence.

SQLite (stdlib, runs on-device) — NOT a vector DB: everything here is
structured data with exact lookups; semantic recall stays in the diary/mem0
layer. One file at runtime/vehicle.db, opened per call (cheap, thread-safe).

`execute_tool(name, args)` is the single entry point the SchemaTool stubs
dispatch through: vehicle_command WRITES state, vehicle_query READS it back
(so "set temp to 22" then "what's the temperature?" closes the loop),
reminder_command / calendar_query get real CRUD. Model-emitted args can be
garbage (e.g. trigger=':') — handlers tolerate and store best-effort.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger("litert_app.vehicle_db")

DB_PATH = Path(__file__).resolve().parent.parent / "runtime" / "vehicle.db"

# Simulated CAN/OBD defaults — seeded once, then mutated by vehicle_command.
_ZONES = ("driver", "passenger", "rear")

_DEFAULT_STATE: dict[str, Any] = {
    "cabin_temp_c": 21.0, "hvac_mode": "auto", "fan_level": 2,
    # Per-zone climate: on/off + target temp (zone "all" fans out).
    "hvac_zones": {z: "off" for z in _ZONES},
    "zone_temp_c": {z: 21.0 for z in _ZONES},
    "sunroof": "closed", "windows": "closed", "defrost": "off",
    "lights": "auto", "media": "off", "media_content": "",
    "volume": 40, "phone_connected": True,
    # OBD-ish read-only signals (vehicle_query targets)
    "fuel_level_pct": 68, "range_km": 412, "battery_soc_pct": 81,
    "odometer_km": 24310, "engine_status": "ok", "coolant_temp_c": 89,
    "tire_pressure_psi": {"fl": 34, "fr": 34, "rl": 33, "rr": 34},
    "dtc_codes": [],
}


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=5)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS vehicle_state(
            key TEXT PRIMARY KEY, value TEXT NOT NULL,
            updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS reminders(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL, trigger TEXT DEFAULT '',
            trigger_type TEXT DEFAULT '', done INTEGER DEFAULT 0,
            created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS calendar_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL, start_ts TEXT NOT NULL,
            end_ts TEXT DEFAULT '', created_at TEXT NOT NULL);
        """)
        try:  # migration: resolved absolute due time for reminders
            c.execute("ALTER TABLE reminders ADD COLUMN due_ts TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
        now = _now()
        for k, v in _DEFAULT_STATE.items():
            c.execute("INSERT OR IGNORE INTO vehicle_state VALUES (?,?,?)",
                      (k, json.dumps(v), now))
        # Seed demo calendar with REAL host-clock dates (tomorrow).
        if not c.execute("SELECT 1 FROM calendar_events LIMIT 1").fetchone():
            import datetime as _dt
            tmw = _dt.date.today() + _dt.timedelta(days=1)
            for title, hhmm in (("Team standup", "09:30"),
                                ("Elevatex AI review", "16:00")):
                c.execute("INSERT INTO calendar_events(title,start_ts,created_at)"
                          " VALUES (?,?,?)", (title, f"{tmw}T{hhmm}:00", now))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _set_state(c: sqlite3.Connection, key: str, value: Any) -> None:
    c.execute("INSERT INTO vehicle_state VALUES (?,?,?) ON CONFLICT(key) "
              "DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
              (key, json.dumps(value), _now()))


def get_state() -> dict[str, Any]:
    with _conn() as c:
        return {r["key"]: json.loads(r["value"])
                for r in c.execute("SELECT key,value FROM vehicle_state")}


def get_reminders(include_done: bool = False) -> list[dict]:
    q = "SELECT * FROM reminders" + ("" if include_done else " WHERE done=0")
    with _conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY id DESC LIMIT 20")]


def get_events() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM calendar_events ORDER BY id LIMIT 20")]


# ---------- tool dispatch ----------

def _zones_for(args: dict) -> list[str]:
    z = str(args.get("zone") or args.get("target") or "all").lower()
    return list(_ZONES) if z in ("all", "cabin", "") else \
        [z] if z in _ZONES else list(_ZONES)


def _hvac_interlock(c: sqlite3.Connection, state: dict) -> list[str]:
    """Real-vehicle rule: whenever any HVAC zone is running, sunroof and
    windows must be (and stay) closed. Returns what was auto-closed."""
    closed = []
    if any(v == "on" for v in state["hvac_zones"].values()):
        for opening in ("sunroof", "windows"):
            if state.get(opening) not in ("closed", None):
                _set_state(c, opening, "closed")
                closed.append(opening)
    return closed


def _vehicle_command(args: dict) -> dict:
    action = str(args.get("action", ""))
    # Any off-ish token in the args means "turn it off" (models phrase this
    # as operation/state/mode/adjustment_type inconsistently).
    tokens = " ".join(str(v).lower() for v in args.values())
    turning_off = any(t in tokens for t in ("off", "stop", "disable"))
    extra: dict[str, Any] = {}
    with _conn() as c:
        state = get_state()
        if action == "set_temperature":
            t = args.get("target_temperature") or args.get("value") or args.get("level")
            zones = _zones_for(args)
            zt, hz = state["zone_temp_c"], state["hvac_zones"]
            for z in zones:
                hz[z] = "on"
                if isinstance(t, (int, float)) and 14 <= float(t) <= 30:
                    zt[z] = float(t)
            _set_state(c, "zone_temp_c", zt)
            _set_state(c, "hvac_zones", hz)
            if isinstance(t, (int, float)) and 14 <= float(t) <= 30 and len(zones) == 3:
                _set_state(c, "cabin_temp_c", float(t))
            state["hvac_zones"] = hz
            extra["auto_closed"] = _hvac_interlock(c, state)
        elif action in ("set_ac", "climate", "hvac", "climate_control", "ac"):
            hz = state["hvac_zones"]
            for z in _zones_for(args):
                hz[z] = "off" if turning_off else "on"
            _set_state(c, "hvac_zones", hz)
            _set_state(c, "hvac_mode", "off" if all(
                v == "off" for v in hz.values()) else "auto")
            state["hvac_zones"] = hz
            extra["auto_closed"] = _hvac_interlock(c, state)
        elif action in ("sunroof", "windows"):
            # Models often emit no open/close word at all (e.g. only
            # adjustment_type='position') — the user asked to actuate it,
            # so default to OPEN unless a closing word appears.
            close_words = ["close", "shut"]
            if action == "windows":
                close_words.append("up")     # "roll the windows up"
            closing = turning_off or any(t in tokens for t in close_words)
            wants_open = not closing
            if wants_open and any(v == "on" for v in state["hvac_zones"].values()):
                # Interlock refuses: climate is running.
                return {"status": "blocked", "reason":
                        f"{action} stays closed while climate control is on",
                        "applied": args}
            _set_state(c, action, "open" if wants_open else "closed")
        elif action in ("defrost", "lights_and_mirrors"):
            key = "lights" if action.startswith("lights") else action
            val = "off" if turning_off else (
                args.get("operation") or args.get("adjustment_type")
                or args.get("state") or "on")
            _set_state(c, key, str(val))
        else:
            _set_state(c, f"last_{action or 'command'}", args)
    out = {"status": "success", "applied": args}
    if extra.get("auto_closed"):
        out["auto_closed"] = extra["auto_closed"]
        out["note"] = ("sunroof/windows were closed automatically because "
                       "climate control is on")
    return out


def _vehicle_query(args: dict) -> dict:
    state = get_state()
    metric = str(args.get("metric") or args.get("query_type")
                 or args.get("action") or "")
    hits = {k: v for k, v in state.items() if metric and metric.split("_")[0] in k}
    return {"status": "success",
            "result": hits or {k: state[k] for k in
                               ("fuel_level_pct", "range_km", "battery_soc_pct",
                                "cabin_temp_c", "engine_status")}}


def _resolve_when(*phrases: str) -> str:
    """Resolve loose natural-language time hints against the HOST CLOCK to
    an absolute ISO timestamp. Handles today/tomorrow/tonight and times
    like '4pm', '16:00', '11:30am'. Returns '' when nothing parses."""
    import datetime as dt
    import re as _re
    text = " ".join(p.lower() for p in phrases)
    day = dt.date.today()
    if "tomorrow" in text:
        day += dt.timedelta(days=1)
    m = _re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    if not m and "tomorrow" not in text and "today" not in text \
            and "tonight" not in text:
        return ""
    hh, mm = (int(m.group(1)), int(m.group(2) or 0)) if m else (9, 0)
    if m and m.group(3) == "pm" and hh < 12:
        hh += 12
    if "tonight" in text and hh < 12:
        hh += 12
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return ""
    when = dt.datetime.combine(day, dt.time(hh, mm))
    if when < dt.datetime.now() and "today" not in text:
        when += dt.timedelta(days=1)  # past time w/o a day → next occurrence
    return when.strftime("%Y-%m-%dT%H:%M:%S")


def _reminder_command(args: dict) -> dict:
    action = str(args.get("action", "")).lower()
    text = str(args.get("reminder_text") or args.get("text") or "").strip()
    with _conn() as c:
        if any(t in action for t in ("delete", "cancel", "remove", "clear")):
            hit = c.execute(
                "SELECT id,text FROM reminders WHERE done=0 AND text LIKE ? "
                "ORDER BY id DESC LIMIT 1", (f"%{text[:40]}%",)).fetchone() \
                if text else None
            if hit:
                c.execute("UPDATE reminders SET done=1 WHERE id=?", (hit["id"],))
                c.commit()  # get_reminders() opens a fresh connection
                return {"status": "success", "deleted": hit["text"],
                        "reminders": get_reminders()}
            c.execute("UPDATE reminders SET done=1")
            return {"status": "success", "deleted": "all",
                    "reminders": []}
        # "list/show my reminders" often arrives as set_reminder with
        # listy text — treat query-ish input as a read.
        if action != "set_reminder" or not text or "reminders" in text.lower():
            return {"status": "success", "now": _now(),
                    "reminders": get_reminders()}
        due = _resolve_when(text, str(args.get("trigger", "")))
        c.execute("INSERT INTO reminders(text,trigger,trigger_type,due_ts,"
                  "created_at) VALUES (?,?,?,?,?)",
                  (text, str(args.get("trigger", "")),
                   str(args.get("trigger_type", "")), due, _now()))
    return {"status": "success", "saved": text, "due_ts": due or "unspecified",
            "reminders_count": len(get_reminders())}


def _calendar_query(args: dict) -> dict:
    # calendar_query is nominally read-only, but the model routes "delete
    # my events" here too — honor it so the spoken confirmation is REAL
    # (it previously narrated deletions that never happened).
    tokens = " ".join(str(v).lower() for v in args.values())
    if any(t in tokens for t in ("delete", "remove", "clear", "cancel")):
        with _conn() as c:
            title = str(args.get("event") or args.get("title") or "").strip()
            if title:
                c.execute("DELETE FROM calendar_events WHERE title LIKE ?",
                          (f"%{title[:40]}%",))
            else:
                c.execute("DELETE FROM calendar_events")
            c.commit()
        return {"status": "success", "deleted": title or "all",
                "now": _now(), "events": get_events()}
    # Include the host clock so the model can ground "today"/"tomorrow".
    return {"status": "success", "now": _now(), "events": get_events()}


def _media_command(args: dict) -> dict:
    content = str(args.get("content", ""))
    with _conn() as c:
        if content.lower() in ("stop", "pause", "off"):
            _set_state(c, "media", "off")
            _set_state(c, "media_content", "")
        else:
            _set_state(c, "media", str(args.get("media_type", "music")))
            _set_state(c, "media_content", content)
    return {"status": "success", "applied": args}


_HANDLERS = {
    "vehicle_command": _vehicle_command,
    "vehicle_query": _vehicle_query,
    "reminder_command": _reminder_command,
    "calendar_query": _calendar_query,
    "media_command": _media_command,
}


def execute_tool(name: str, args: Mapping[str, Any]) -> dict | None:
    """Run a tool against the store. Returns the result payload, or None if
    the tool has no persistence handler (caller falls back to the stub)."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return None
    try:
        init_db()
        out = handler(dict(args))
        log.info("vehicle_db: %s -> %s", name, str(out)[:160])
        return out
    except Exception:
        log.exception("vehicle_db handler failed for %s; falling back to stub", name)
        return None
