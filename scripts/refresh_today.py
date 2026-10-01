"""Rebuild the league-wide 2025-2026 training table and score unfinished games.

    python scripts/refresh_today.py

Completed games only, from 2025-03-01 through games that are final when this
process checks the MLB schedule. America/Denver decides the cutoff date.
Pre-2025 rows are not kept and are not copied forward. The two-game wild-card
file is only a slice of that table. The remaining-games file lists every game
on the Denver date that is not final.

Starter FIP, K%, BB%, lineup wRC+ versus the starter's hand, and park factor
are taken from pybaseball. A missing field is left out of the model. Box scores
supply rest, last-start innings, 14-day bullpen ERA, yesterday's bullpen
innings, 14-day runs scored and allowed, and the Game 1 winner.
"""

from __future__ import annotations

import io
import json
import re
import sys
import threading
import time
import traceback
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "cache"
BOX_CACHE = CACHE / "boxscores"
HAND_CACHE = CACHE / "hands.json"
TABLE_PATH = ROOT / "artifacts" / "training_table.csv"
PRED_PATH = ROOT / "artifacts" / "wc_2026-09-30_predictions.csv"
ALL_TEAMS_PATH = ROOT / "artifacts" / "mlb_2026-09-30_all_teams.csv"
REMAINING_PATH = ROOT / "artifacts" / "mlb_2026-09-30_remaining_games.csv"
NOTES_PATH = ROOT / "NOTES.md"
SLICE_MATCHUPS = (
    ("Boston Red Sox", "New York Yankees"),
    ("Chicago Cubs", "San Diego Padres"),
)
GAME_COLUMNS = (
    "date",
    "away",
    "home",
    "starter_away",
    "starter_home",
    "home_win_prob",
    "predicted_away_runs",
    "predicted_home_runs",
    "model_side",
    "data_cutoff",
)
TEAM_COLUMNS = (
    "team",
    "last_14_runs_scored",
    "last_14_runs_allowed",
    "bullpen_era_14d",
    "data_min",
    "data_cutoff",
)
EXPECTED_TEAMS = (
    "Arizona Diamondbacks",
    "Athletics",
    "Atlanta Braves",
    "Baltimore Orioles",
    "Boston Red Sox",
    "Chicago Cubs",
    "Chicago White Sox",
    "Cincinnati Reds",
    "Cleveland Guardians",
    "Colorado Rockies",
    "Detroit Tigers",
    "Houston Astros",
    "Kansas City Royals",
    "Los Angeles Angels",
    "Los Angeles Dodgers",
    "Miami Marlins",
    "Milwaukee Brewers",
    "Minnesota Twins",
    "New York Mets",
    "New York Yankees",
    "Philadelphia Phillies",
    "Pittsburgh Pirates",
    "San Diego Padres",
    "San Francisco Giants",
    "Seattle Mariners",
    "St. Louis Cardinals",
    "Tampa Bay Rays",
    "Texas Rangers",
    "Toronto Blue Jays",
    "Washington Nationals",
)

START = date(2025, 3, 1)
TRAIN_END = date(2026, 9, 1)
GAME_TYPES = {"R", "F", "D", "L", "W"}
SCHEMA = 1
API = "https://statsapi.mlb.com/api/v1"
WORKERS = 8
NAMED_ABSENCES = ("Aaron Judge", "Roman Anthony", "Jeremiah Estrada")

SIDE_STATS = (
    "starter_rest_days",
    "starter_last_ip",
    "bullpen_era_14",
    "bullpen_ip_yesterday",
    "rs_14",
    "ra_14",
)
GAME_STATS = ("home_won_game1",)
PYB_SIDE = (
    "starter_fip",
    "starter_xfip",
    "starter_k_pct",
    "starter_bb_pct",
    "lineup_ops_vs_hand",
    "lineup_wrc_vs_hand",
)
SAVANT_PITCH_URL = (
    "https://baseballsavant.mlb.com/leaderboard/custom"
    "?year={year}&type=pitcher&filter=&sort=1&sortDir=desc&min=1"
    "&selections=k_percent,bb_percent,fip,xfip&csv=true"
)
SAVANT_PARK_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
    "?type=year&year=2025&batSide=&stat=index_woba"
)
PYB_GAME = ("park_factor",)
BUGS_FIXED = (
    "README pointed at baseball_dataset.py, baseball_model.py, and baseball_prediction.py in the repo root. Those files live in legacy/. The README now points at legacy/ and at scripts/refresh_today.py.",
    "legacy/baseball_prediction.py hardcoded the 2024 season in the date filter, the schedule end date, and the team lookup. Those now use datetime.now().year.",
    "legacy/baseball_dataset.py described a 2000-2024 build and its year loop stopped at 2025. The loop now runs through the current year, and the retry loop runs only when the file is executed as a script.",
    "legacy/baseball_model.py and legacy/baseball_prediction.py read stats.csv and the pickles from the working directory at import. They now look next to the script, then in the working directory, and a missing csv exits instead of failing at import.",
    "Unused imports were removed from the three legacy scripts. The Ridge Classifier and the linear run model were not rewritten.",
    "mlb_predictor/data.py constructed a Supabase client at import, and importing the module required the supabase package. The import and the client now happen only when the client is used. No key was added.",
    "pip install -r requirements.txt failed on numpy==2.0.0 because Python 3.13 has no wheel for that pin and this machine has no C compiler. numpy, pandas, scikit-learn, and scipy were pinned to the installed wheels (2.5.3, 3.0.6, 1.9.1, 1.18.1). The retry succeeded.",
)

COUNT_KEYS = ("outs", "er", "k", "bb", "h", "hr", "hbp", "bf", "ab", "doubles", "triples", "sf", "pitches")
LOGS: list[str] = []
PRINT_LOCK = threading.Lock()
TLS = threading.local()


def note(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)
        LOGS.append(msg)


def denver_now() -> datetime:
    try:
        return datetime.now(ZoneInfo("America/Denver"))
    except Exception as exc:
        note(f"America/Denver unavailable ({exc}); using MDT UTC-6")
        return datetime.now(timezone(timedelta(hours=-6)))


def http() -> requests.Session:
    session = getattr(TLS, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "mlb-game-predictor-refresh/1.0"
        TLS.session = session
    return session


def get_json(url: str, params: dict | None = None, tries: int = 5) -> dict:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            response = http().get(url, params=params, timeout=45)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.2 * (attempt + 1))
                last = RuntimeError(f"{response.status_code} {url}")
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"GET failed {url} params={params} err={last}")


def parse_iso(value: str) -> date:
    return date.fromisoformat(value[:10])


def is_before(rec_date: date, rec_number: int, game_date: date, game_number: int) -> bool:
    if rec_date < game_date:
        return True
    return rec_date == game_date and rec_number < game_number


def outs_to_ip(outs: int) -> float:
    return outs / 3.0


def fip_from_counts(outs: int, hr: int, bb: int, hbp: int, k: int, const: float) -> float | None:
    if outs < 3:
        return None
    ip = outs_to_ip(outs)
    return (13 * hr + 3 * (bb + hbp) - 2 * k) / ip + const


def self_check() -> None:
    assert is_before(date(2026, 9, 29), 1, date(2026, 9, 30), 1)
    assert not is_before(date(2026, 9, 30), 1, date(2026, 9, 30), 1)
    assert is_before(date(2026, 9, 30), 1, date(2026, 9, 30), 2)
    raw = (13 * 1 + 3 * 2 - 2 * 9) / 9 + 3.1
    got = fip_from_counts(27, 1, 2, 0, 9, 3.1)
    assert got is not None and abs(got - raw) < 1e-9
    assert abs(outs_to_ip(19) - (19 / 3)) < 1e-9


def pitching_line(stats: dict) -> dict:
    def num(*keys: str) -> int:
        for key in keys:
            if stats.get(key) is not None:
                try:
                    return int(stats.get(key) or 0)
                except (TypeError, ValueError):
                    continue
        return 0

    return {
        "outs": num("outs"),
        "pitches": num("numberOfPitches", "pitchesThrown"),
        "er": num("earnedRuns"),
        "h": num("hits"),
        "bb": num("baseOnBalls"),
        "hbp": num("hitByPitch", "hitBatsmen"),
        "k": num("strikeOuts"),
        "hr": num("homeRuns"),
        "bf": num("battersFaced"),
        "ab": num("atBats"),
        "doubles": num("doubles"),
        "triples": num("triples"),
        "sf": num("sacFlies"),
        "pitches_alt": num("pitchesThrown"),
        "gs": num("gamesStarted"),
        "saves": num("saves"),
        "holds": num("holds"),
        "blown": num("blownSaves"),
        "gf": num("gamesFinished"),
        "note": stats.get("note") or "",
    }


def sub_line(staff: dict, starter: dict) -> dict:
    out = {}
    for key in ("outs", "er", "k", "bb", "h", "hr", "hbp", "bf"):
        out[key] = max(0, int(staff.get(key) or 0) - int(starter.get(key) or 0))
    return out


def sum_lines(lines: list[dict]) -> dict:
    acc = {key: 0 for key in ("outs", "er", "k", "bb", "h", "hr", "hbp", "bf")}
    for line in lines:
        for key in acc:
            acc[key] += int(line.get(key) or 0)
    return acc


def high_lev_outs(relievers: list[dict]) -> int:
    # Closer and highest-leverage relievers: save, hold, blown save, or the finisher.
    chosen = []
    for line in relievers:
        note_text = line.get("note") or ""
        if line["saves"] or line["holds"] or line["blown"] or line["gf"]:
            chosen.append(line)
        elif re.search(r"\((?:H|S|BS)\b", note_text):
            chosen.append(line)
    if not chosen and relievers:
        chosen = [relievers[-1]]
    return sum(int(line["outs"]) for line in chosen)


def pack_side(block: dict) -> dict:
    players = block.get("players") or {}
    lines = []
    for pid in block.get("pitchers") or []:
        player = players.get(f"ID{pid}") or {}
        stats = (player.get("stats") or {}).get("pitching")
        if not stats:
            continue
        line = pitching_line(stats)
        line["id"] = int(pid)
        line["name"] = (player.get("person") or {}).get("fullName") or ""
        lines.append(line)
    starter = next((line for line in lines if line["gs"] >= 1), None)
    if starter is None and lines:
        starter = lines[0]
    relievers = [line for line in lines if starter is None or line["id"] != starter["id"]]
    staff_stats = (block.get("teamStats") or {}).get("pitching") or {}
    staff = pitching_line(staff_stats) if staff_stats else None
    if staff and starter:
        bullpen = sub_line(staff, starter)
    else:
        bullpen = sum_lines(relievers)
    return {
        "starter": starter,
        "bullpen": bullpen,
        "staff": staff or sum_lines(([starter] if starter else []) + relievers),
        "highlev_outs": high_lev_outs(relievers),
    }


def month_spans(start: date, end: date):
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=9))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def parse_schedule_game(game: dict, date_fallback: str | None) -> dict | None:
    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_team = home.get("team") or {}
    away_team = away.get("team") or {}
    if not home_team.get("id") or not away_team.get("id"):
        return None
    status = game.get("status") or {}
    official = game.get("officialDate") or date_fallback
    if not official:
        return None
    venue = game.get("venue") or {}
    home_probable = home.get("probablePitcher") or {}
    away_probable = away.get("probablePitcher") or {}
    return {
        "game_pk": int(game["gamePk"]),
        "game_date": parse_iso(official),
        "game_number": int(game.get("gameNumber") or 1),
        "game_type": game.get("gameType") or "",
        "abstract": status.get("abstractGameState") or "",
        "detailed": status.get("detailedState") or "",
        "venue_id": venue.get("id"),
        "series_description": game.get("seriesDescription") or "",
        "series_game_number": int(game.get("seriesGameNumber") or 1),
        "home_id": int(home_team["id"]),
        "away_id": int(away_team["id"]),
        "home_name": home_team.get("name") or "",
        "away_name": away_team.get("name") or "",
        "home_runs": home.get("score"),
        "away_runs": away.get("score"),
        "home_probable_id": home_probable.get("id"),
        "home_probable_name": home_probable.get("fullName"),
        "away_probable_id": away_probable.get("id"),
        "away_probable_name": away_probable.get("fullName"),
    }


def fetch_range(start: date, end: date, depth: int = 0) -> list[dict]:
    payload = get_json(
        f"{API}/schedule",
        {
            "sportId": 1,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "hydrate": "probablePitcher,venue",
        },
    )
    games = []
    for bucket in payload.get("dates") or []:
        for game in bucket.get("games") or []:
            parsed = parse_schedule_game(game, bucket.get("date"))
            if parsed:
                games.append(parsed)
    reported = payload.get("totalGames")
    span = (end - start).days
    if reported is not None and reported > len(games) and span > 0 and depth < 8:
        mid = start + timedelta(days=max(span // 2, 0))
        if start <= mid < end:
            left = fetch_range(start, mid, depth + 1)
            right = fetch_range(mid + timedelta(days=1), end, depth + 1)
            return left + right
    return games


def fetch_schedule(start: date, end: date) -> list[dict]:
    found: dict[int, dict] = {}
    for chunk_start, chunk_end in month_spans(start, end):
        for game in fetch_range(chunk_start, chunk_end):
            if game["game_type"] not in GAME_TYPES:
                continue
            if game["game_date"] < START or game["game_date"] > end:
                continue
            found[game["game_pk"]] = game
    games = list(found.values())
    games.sort(key=lambda game: (game["game_date"], game["game_number"], game["game_pk"]))
    return games


def finals_only(games: list[dict], end: date) -> list[dict]:
    rows = []
    for game in games:
        if game["abstract"] != "Final":
            continue
        if game["detailed"] in {"Postponed", "Cancelled", "Suspended"}:
            continue
        if game["game_date"] < START or game["game_date"] > end:
            continue
        if game["home_runs"] is None or game["away_runs"] is None:
            continue
        rows.append(game)
    return rows


def cache_path(game_pk: int) -> Path:
    return BOX_CACHE / f"{game_pk}.json"


def load_cached(game_pk: int) -> dict | None:
    path = cache_path(game_pk)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != SCHEMA:
        return None
    payload["game_date"] = parse_iso(payload["game_date"])
    return payload


def save_cached(record: dict) -> None:
    path = cache_path(record["game_pk"])
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(record)
    body["schema"] = SCHEMA
    body["game_date"] = record["game_date"].isoformat()
    path.write_text(json.dumps(body), encoding="utf-8")


def record_from_box(meta: dict, box: dict) -> dict:
    teams = box.get("teams") or {}
    packs = {}
    for side in ("home", "away"):
        block = teams.get(side) or {}
        team = block.get("team") or {}
        if not team.get("id"):
            raise RuntimeError(f"boxscore missing team {meta['game_pk']}")
        packs[int(team["id"])] = pack_side(block)
    if meta["home_id"] not in packs or meta["away_id"] not in packs:
        raise RuntimeError(f"boxscore teams do not match schedule {meta['game_pk']}")
    home = packs[meta["home_id"]]
    away = packs[meta["away_id"]]
    home_runs = int(meta["home_runs"])
    away_runs = int(meta["away_runs"])
    return {
        "schema": SCHEMA,
        "game_pk": meta["game_pk"],
        "game_date": meta["game_date"],
        "game_number": meta["game_number"],
        "game_type": meta["game_type"],
        "venue_id": meta["venue_id"] or meta["home_id"],
        "series_description": meta["series_description"],
        "series_game_number": meta["series_game_number"],
        "home_id": meta["home_id"],
        "away_id": meta["away_id"],
        "home_name": meta["home_name"],
        "away_name": meta["away_name"],
        "home_runs": home_runs,
        "away_runs": away_runs,
        "home_starter": home["starter"],
        "away_starter": away["starter"],
        "home_bullpen": home["bullpen"],
        "away_bullpen": away["bullpen"],
        "home_staff": home["staff"],
        "away_staff": away["staff"],
        "home_highlev_outs": home["highlev_outs"],
        "away_highlev_outs": away["highlev_outs"],
    }


def fetch_box_record(meta: dict) -> dict:
    cached = load_cached(meta["game_pk"])
    if cached is not None:
        return cached
    box = get_json(f"{API}/game/{meta['game_pk']}/boxscore")
    record = record_from_box(meta, box)
    save_cached(record)
    return record


def load_records(finals: list[dict]) -> list[dict]:
    BOX_CACHE.mkdir(parents=True, exist_ok=True)
    records: list[dict | None] = [None] * len(finals)
    failed: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_map = {pool.submit(fetch_box_record, meta): index for index, meta in enumerate(finals)}
        for future in as_completed(future_map):
            index = future_map[future]
            meta = finals[index]
            try:
                records[index] = future.result()
            except Exception as exc:
                failed.append(f"{meta['game_pk']} {meta['game_date']} {exc}")
            done += 1
            if done % 250 == 0 or done == len(finals):
                note(f"boxscores {done}/{len(finals)} failed {len(failed)}")
    if failed:
        for line in failed[:20]:
            note(f"boxscore failed {line}")
        raise RuntimeError(f"{len(failed)} final games had no boxscore")
    return [record for record in records if record is not None]


def download_transactions(start: date, end: date) -> list[dict]:
    rows = []
    for chunk_start, chunk_end in month_spans(start, end):
        try:
            payload = get_json(
                f"{API}/transactions",
                {"sportId": 1, "startDate": chunk_start.isoformat(), "endDate": chunk_end.isoformat()},
            )
        except Exception as exc:
            note(f"MLB transactions failed {chunk_start}..{chunk_end}: {exc}")
            continue
        for item in payload.get("transactions") or []:
            desc = item.get("description") or ""
            raw_date = item.get("date") or item.get("effectiveDate") or ""
            if not desc or not raw_date:
                continue
            rows.append({"date": parse_iso(raw_date), "description": desc})
    note(f"MLB transactions kept {len(rows)}")
    return rows


def injured_code(status: dict | None) -> bool:
    if not status:
        return False
    code = status.get("code") or ""
    desc = (status.get("description") or "").lower()
    if len(code) > 1 and code[0] == "D" and code[1].isdigit():
        return True
    return "injur" in desc


def fetch_named_roster() -> dict[str, dict]:
    found = {}
    for name in NAMED_ABSENCES:
        try:
            searched = get_json(f"{API}/people/search", {"names": name, "sportId": 1})
            people = searched.get("people") or []
            if not people:
                note(f"MLB people search missed {name}")
                continue
            pid = people[0]["id"]
            payload = get_json(
                f"{API}/people",
                {"personIds": pid, "hydrate": "currentTeam,rosterEntries"},
            )
            person = (payload.get("people") or [None])[0]
            if not person:
                note(f"MLB people hydrate missed {name}")
                continue
            entries = [
                entry
                for entry in person.get("rosterEntries") or []
                if entry.get("isActive") and not entry.get("endDate")
            ]
            entries.sort(key=lambda entry: entry.get("statusDate") or "")
            current = entries[-1] if entries else None
            status = (current or {}).get("status") or {}
            status_date = None
            if current and current.get("statusDate"):
                status_date = parse_iso(current["statusDate"])
            team_name = ((current or {}).get("team") or person.get("currentTeam") or {}).get("name")
            found[name] = {
                "id": pid,
                "team": team_name,
                "status_code": status.get("code"),
                "status_desc": status.get("description"),
                "status_date": status_date,
                "injured": injured_code(status),
            }
            note(
                f"injury roster {name} team={team_name} "
                f"status={status.get('code')}:{status.get('description')} "
                f"status_date={status_date} injured={injured_code(status)}"
            )
        except Exception as exc:
            note(f"MLB injury roster failed for {name}: {exc}")
    return found


PLACED_RE = re.compile(
    r"placed\s+(?:[A-Z0-9]{1,3}\s+)?(.+?)\s+on the (\d+)-day injured list"
    r"(?: retroactive to ([A-Za-z]+ \d{1,2}, \d{4}))?",
    re.I,
)
ACTIVATED_RE = re.compile(
    r"activated\s+(?:[A-Z0-9]{1,3}\s+)?(.+?)\s+from the (\d+)-day injured list",
    re.I,
)


def team_in_text(desc: str, names: set[str]) -> str | None:
    hits = [name for name in names if name and name in desc]
    if not hits:
        return None
    return max(hits, key=len)


def parse_absences(transactions: list[dict], team_names: set[str], roster: dict[str, dict]) -> dict:
    events = []
    blurbs = []
    for item in transactions:
        desc = item["description"]
        if any(name in desc for name in NAMED_ABSENCES):
            blurbs.append(f"{item['date']} {desc}")
        placed = PLACED_RE.search(desc)
        activated = ACTIVATED_RE.search(desc)
        if placed:
            raw_name = " ".join(placed.group(1).split())
            if raw_name not in NAMED_ABSENCES:
                continue
            start = item["date"]
            if placed.group(3):
                try:
                    start = datetime.strptime(placed.group(3), "%B %d, %Y").date()
                except ValueError:
                    start = item["date"]
            team = team_in_text(desc, team_names)
            events.append((start, 0, "place", raw_name, team, desc))
        elif activated:
            raw_name = " ".join(activated.group(1).split())
            if raw_name not in NAMED_ABSENCES:
                continue
            team = team_in_text(desc, team_names)
            events.append((item["date"], 1, "activate", raw_name, team, desc))
    events.sort(key=lambda item: (item[0], item[1]))
    open_stints: dict[str, list[dict]] = defaultdict(list)
    closed: list[dict] = []
    for start, _order, kind, person, team, desc in events:
        if kind == "place":
            if not team:
                note(f"injury placement without a team, skipped: {desc}")
                continue
            open_stints[person].append({"person": person, "team": team, "start": start, "end": None})
        else:
            stack = open_stints.get(person) or []
            if not stack:
                continue
            stint = stack.pop()
            stint["end"] = start
            closed.append(stint)
    for person, stack in open_stints.items():
        info = roster.get(person) or {}
        for stint in stack:
            if info and not info.get("injured"):
                stint["end"] = info.get("status_date") or stint["start"]
                note(
                    f"injury stint closed from roster {person} end={stint['end']} "
                    f"status={info.get('status_code')}"
                )
            elif info and info.get("injured"):
                note(f"injury stint still open {person} since {stint['start']} ({info.get('status_desc')})")
            closed.append(stint)
    for line in blurbs:
        note(f"injury text {line}")
    questionable = {
        name: any("questionable" in line.lower() and name in line for line in blurbs)
        for name in NAMED_ABSENCES
    }
    for name, flagged in questionable.items():
        if not flagged:
            note(f"injury feed does not list {name} as questionable")
    return {"stints": closed, "questionable": questionable, "roster": roster}


def absence_flag(book: dict, person: str, team_name: str, game_date: date) -> int:
    for stint in book.get("stints") or []:
        if stint["person"] != person or stint["team"] != team_name:
            continue
        end = stint["end"]
        if stint["start"] <= game_date and (end is None or game_date < end):
            return 1
    return 0


def norm_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = text.encode("ascii", "ignore").decode()
    text = text.lower().replace(".", "").replace("'", "")
    return " ".join(text.split())


def canon_team(name: str) -> str:
    aliases = {
        "Oakland Athletics": "Athletics",
        "Cleveland Indians": "Cleveland Guardians",
    }
    return aliases.get(name, name)


ABBR = {
    "ARI": "Arizona Diamondbacks",
    "AZ": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "ATH": "Athletics",
    "OAK": "Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SDP": "San Diego Padres",
    "SF": "San Francisco Giants",
    "SFG": "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
    "WSN": "Washington Nationals",
}


def map_team_value(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if text in ABBR:
        return ABBR[text]
    return canon_team(text)


def first_col(frame: pd.DataFrame, options: tuple[str, ...]) -> str | None:
    lookup = {str(col).lower(): col for col in frame.columns}
    for option in options:
        if option.lower() in lookup:
            return lookup[option.lower()]
    return None


def call_table(label: str, fn, variants: list[tuple]) -> pd.DataFrame | None:
    last = None
    for args, kwargs in variants:
        try:
            frame = fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            note(f"{label} failed args={args} kwargs={kwargs}: {exc}")
            continue
        if frame is None or len(frame) == 0:
            note(f"{label} returned no rows args={args}")
            continue
        note(f"{label} rows={len(frame)} cols={list(frame.columns)[:12]}")
        return frame
    note(f"{label} unavailable: {last}")
    return None


def empty_ext() -> dict:
    return {
        "pitcher_prior": {},
        "lineup_wrc": {},
        "lineup_ops": {},
        "park": {},
        "used": [],
        "skipped": [],
        "have": set(),
    }


def skip_field(ext: dict, field: str, reason: str) -> None:
    line = f"{field}: {reason}"
    ext["skipped"].append(line)
    note(f"skipped {line}")


def store_pitcher_table(ext: dict, frame: pd.DataFrame, source: str) -> None:
    id_col = first_col(frame, ("mlbID", "MLBAMID", "key_mlbam", "player_id", "mlb_id"))
    name_col = first_col(frame, ("Name", "Player", "NameASCII"))
    columns = {
        "FIP": ("fip", first_col(frame, ("FIP", "fip", "p_fip"))),
        "xFIP": ("xfip", first_col(frame, ("xFIP", "xfip", "p_xfip"))),
        "K%": ("k_pct", first_col(frame, ("K%", "K_pct", "k_percent", "p_k_percent"))),
        "BB%": ("bb_pct", first_col(frame, ("BB%", "BB_pct", "bb_percent", "p_bb_percent"))),
    }
    present = {field: pair for field, pair in columns.items() if pair[1]}
    for field, (_key, _col) in columns.items():
        if field not in present:
            skip_field(ext, field, f"{source} has no {field} column")
    if not present:
        return
    name_index: dict[str, list] = defaultdict(list)
    stored = {field: 0 for field in present}
    for idx, row in frame.iterrows():
        pid = None
        if id_col and pd.notna(row[id_col]):
            try:
                pid = int(row[id_col])
            except (TypeError, ValueError):
                pid = None
        if pid is None and name_col and pd.notna(row[name_col]):
            name_index[norm_name(str(row[name_col]))].append(idx)
            continue
        if pid is None:
            continue
        slot = ext["pitcher_prior"].setdefault(pid, {})
        for field, (key, col) in present.items():
            if pd.isna(row[col]):
                continue
            try:
                slot[key] = float(row[col])
            except (TypeError, ValueError):
                continue
            stored[field] += 1
    for key, indexes in name_index.items():
        if len(indexes) != 1:
            continue
        row = frame.loc[indexes[0]]
        slot = ext["pitcher_prior"].setdefault(("name", key), {})
        for field, (store_key, col) in present.items():
            if pd.isna(row[col]):
                continue
            try:
                value = float(row[col])
            except (TypeError, ValueError):
                continue
            slot.setdefault(store_key, value)
            stored[field] += 1
    for field, count in stored.items():
        if count:
            ext["have"].add(field)
            ext["used"].append(f"{source} {field} n={count}")
            note(f"{source} {field} stored for {count} pitchers; joined only onto 2026 games")
        else:
            skip_field(ext, field, f"{source} {field} column was empty")


def store_hand_wrc(ext: dict, frame: pd.DataFrame, source: str) -> None:
    team_col = first_col(frame, ("Team", "team", "Tm", "name"))
    hand_cols = {
        "L": first_col(frame, ("wRC+ vs L", "wRC+ vs LHP", "wRC+_vs_LHP", "wrc_plus_vs_l")),
        "R": first_col(frame, ("wRC+ vs R", "wRC+ vs RHP", "wRC+_vs_RHP", "wrc_plus_vs_r")),
    }
    if not team_col or not hand_cols["L"] or not hand_cols["R"]:
        skip_field(
            ext,
            "lineup wRC+ versus starter hand",
            f"{source} has no wRC+ split versus pitcher hand",
        )
        return
    found = 0
    for _, row in frame.iterrows():
        team = map_team_value(row[team_col])
        if not team:
            continue
        for hand, col in hand_cols.items():
            if pd.isna(row[col]):
                continue
            ext["lineup_wrc"][(team, hand)] = float(row[col])
            found += 1
    covered = {team for team, _hand in ext["lineup_wrc"]}
    if found and set(EXPECTED_TEAMS) <= covered:
        ext["have"].add("wRC+_vs_hand")
        ext["used"].append(f"{source} wRC+ vs hand n={found}")
        note(f"{source} lineup wRC+ versus hand stored for {found} team-hands; joined only onto 2026 games")
        return
    missing = [team for team in EXPECTED_TEAMS if team not in covered]
    ext["lineup_wrc"].clear()
    skip_field(
        ext,
        "lineup wRC+ versus starter hand",
        f"{source} split did not cover every team ({', '.join(missing)})",
    )


def store_park_table(ext: dict, frame: pd.DataFrame, source: str, year: int) -> bool:
    year_col = first_col(frame, ("yearID", "year", "Season"))
    team_col = first_col(frame, ("name", "Team", "team", "teamName", "Tm"))
    pf_col = first_col(frame, ("BPF", "park_factor", "Park Factor", "PF", "basic_pf"))
    if not team_col or not pf_col:
        return False
    found: dict[str, float] = {}
    for _, row in frame.iterrows():
        if year_col:
            try:
                if int(row[year_col]) != year:
                    continue
            except (TypeError, ValueError):
                continue
        if pd.isna(row[pf_col]):
            continue
        team = map_team_value(row[team_col])
        if team:
            found[team] = float(row[pf_col])
    if set(EXPECTED_TEAMS) <= set(found):
        ext["park"] = found
        ext["have"].add("park_factor")
        ext["used"].append(f"{source} park factor {year} n={len(found)}")
        note(f"{source} {year} park factor stored for {len(found)} teams; joined only onto 2026 games")
        return True
    return False


def fetch_text(url: str) -> str:
    response = http().get(url, timeout=60)
    response.raise_for_status()
    return response.text


def parse_rate(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.startswith("."):
        text = "0" + text
    try:
        return float(text)
    except ValueError:
        return None


def mlb_team_names(season: int) -> dict[int, str]:
    payload = get_json(f"{API}/teams", {"sportId": 1, "season": season})
    names = {}
    for team in payload.get("teams") or []:
        name = canon_team(team.get("name") or "")
        if team.get("id") and name in EXPECTED_TEAMS:
            names[int(team["id"])] = name
    return names


def pull_savant_pitching(ext: dict) -> None:
    note("FanGraphs is not called. The last run returned HTTP 403.")
    note("pybaseball Savant wrappers publish expected stats and percentile ranks, not the K% rate, BB% rate, FIP, or xFIP")
    try:
        frame = pd.read_csv(io.StringIO(fetch_text(SAVANT_PITCH_URL.format(year=2025))))
    except Exception as exc:
        for field in ("starter FIP or xFIP", "K%", "BB%"):
            skip_field(ext, field, f"Baseball Savant custom leaderboard failed ({exc})")
        return
    if frame is None or len(frame) == 0:
        for field in ("starter FIP or xFIP", "K%", "BB%"):
            skip_field(ext, field, "Baseball Savant custom leaderboard returned no rows")
        return
    note(f"Baseball Savant custom leaderboard 2025 rows={len(frame)} cols={list(frame.columns)}")
    store_pitcher_table(ext, frame, "Baseball Savant custom leaderboard 2025")
    if "FIP" not in ext["have"] and "xFIP" not in ext["have"]:
        skip_field(
            ext,
            "starter FIP or xFIP",
            "Baseball Savant returned empty FIP and xFIP columns, and the MLB Stats API pitching line has no FIP or xFIP field",
        )


def pull_lineup_ops(ext: dict) -> None:
    try:
        names = mlb_team_names(2025)
    except Exception as exc:
        skip_field(ext, "lineup OPS or wRC+ versus starter hand", f"MLB team list failed ({exc})")
        return
    found: dict[tuple[str, str], float] = {}
    for team_id, name in names.items():
        try:
            payload = get_json(
                f"{API}/teams/{team_id}/stats",
                {"stats": "statSplits", "group": "hitting", "season": 2025, "sitCodes": "vl,vr"},
            )
        except Exception as exc:
            note(f"MLB hitting splits failed for {name}: {exc}")
            continue
        for group in payload.get("stats") or []:
            for split in group.get("splits") or []:
                code = (split.get("split") or {}).get("code")
                hand = {"vl": "L", "vr": "R"}.get(code)
                ops = parse_rate((split.get("stat") or {}).get("ops"))
                if hand and ops is not None:
                    found[(name, hand)] = ops
    covered = {team for team in EXPECTED_TEAMS if (team, "L") in found and (team, "R") in found}
    if set(EXPECTED_TEAMS) <= covered:
        ext["lineup_ops"] = found
        ext["have"].add("OPS_vs_hand")
        ext["used"].append("MLB Stats API 2025 team OPS versus LHP and RHP")
        note("MLB Stats API 2025 OPS versus hand stored for 30 teams; joined only onto 2026 games")
        note("wRC+ versus hand was not in the MLB Stats API split. OPS is stored instead of a wRC+ label")
        return
    missing = [team for team in EXPECTED_TEAMS if team not in covered]
    skip_field(
        ext,
        "lineup OPS or wRC+ versus starter hand",
        f"MLB Stats API splits did not cover every team ({', '.join(missing)})",
    )


def pull_savant_park(ext: dict) -> None:
    try:
        text = fetch_text(SAVANT_PARK_URL)
        start = text.find('[{"grouping_venue_conditions"')
        if start < 0:
            raise ValueError("park-factor page had no data array")
        rows, _end = json.JSONDecoder().raw_decode(text[start:])
        names = mlb_team_names(2025)
    except Exception as exc:
        skip_field(ext, "park factor", f"Baseball Savant park factors failed ({exc})")
        return
    found: dict[str, float] = {}
    for row in rows:
        if str(row.get("key_year")) != "2025" or str(row.get("is_sport_mlb")) != "1":
            continue
        if row.get("key_bat_side") != "All" or row.get("grouping_venue_conditions") != "All":
            continue
        try:
            team_id = int(row.get("main_team_id"))
        except (TypeError, ValueError):
            continue
        name = names.get(team_id)
        value = parse_rate(row.get("index_woba"))
        if name and value is not None:
            found[name] = value
    if set(EXPECTED_TEAMS) <= set(found):
        ext["park"] = found
        ext["have"].add("park_factor")
        ext["used"].append("Baseball Savant 2025 park factor index_woba, 3-year rolling window published as 2023-2025")
        note("Baseball Savant park factor index_woba stored for 30 teams; joined only onto 2026 games")
        return
    missing = [team for team in EXPECTED_TEAMS if team not in found]
    skip_field(ext, "park factor", f"Baseball Savant park factors missed {', '.join(missing)}")


def pull_pybaseball(years: tuple[int, ...]) -> dict:
    """Savant and MLB Stats API only. FanGraphs is not called."""
    del years
    ext = empty_ext()
    pull_savant_pitching(ext)
    pull_lineup_ops(ext)
    pull_savant_park(ext)
    return ext


def load_hands(pitcher_ids: set[int]) -> dict[int, str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached: dict[int, str] = {}
    if HAND_CACHE.exists():
        try:
            raw = json.loads(HAND_CACHE.read_text(encoding="utf-8"))
            cached = {int(key): value for key, value in raw.items()}
        except (OSError, json.JSONDecodeError, ValueError):
            cached = {}
    missing = [pid for pid in sorted(pitcher_ids) if pid not in cached]
    for offset in range(0, len(missing), 80):
        chunk = missing[offset : offset + 80]
        try:
            payload = get_json(f"{API}/people", {"personIds": ",".join(str(pid) for pid in chunk)})
        except Exception as exc:
            note(f"pitcher hand lookup failed: {exc}")
            continue
        for person in payload.get("people") or []:
            hand = (person.get("pitchHand") or {}).get("code")
            if person.get("id") and hand:
                cached[int(person["id"])] = hand
    HAND_CACHE.write_text(json.dumps({str(key): value for key, value in cached.items()}), encoding="utf-8")
    return cached


def attach_hands(records: list[dict], hands: dict[int, str]) -> None:
    for record in records:
        for side in ("home_starter", "away_starter"):
            starter = record.get(side)
            if not starter:
                continue
            starter["hand"] = hands.get(int(starter["id"]))


class History:
    def __init__(self) -> None:
        self.pitcher = defaultdict(list)
        self.team = defaultdict(list)
        self.bullpen = defaultdict(list)
        self.series: dict = {}
        self.park_runs = defaultdict(int)
        self.park_games = defaultdict(int)
        self.league_runs = 0
        self.league_games = 0
        self.league = {"outs": 0, "er": 0, "hr": 0, "bb": 0, "hbp": 0, "k": 0}

    def fip_c(self) -> float:
        outs = self.league["outs"]
        if outs < 900:
            return 3.10
        ip = outs_to_ip(outs)
        raw = (13 * self.league["hr"] + 3 * (self.league["bb"] + self.league["hbp"]) - 2 * self.league["k"]) / ip
        era = self.league["er"] * 9 / ip
        return era - raw


def add_counts(bucket: dict, line: dict | None) -> None:
    if not line:
        return
    for key in ("outs", "er", "hr", "bb", "hbp", "k"):
        bucket[key] += int(line.get(key) or 0)


def series_key(record: dict):
    return (
        record["game_date"].year,
        frozenset((record["home_id"], record["away_id"])),
        record["series_description"],
    )


def update_history(hist: History, record: dict) -> None:
    key = series_key(record)
    if record["series_game_number"] <= 1 or key not in hist.series:
        hist.series[key] = {"games": {}, "game1_winner": None}
    state = hist.series[key]
    if record["home_runs"] > record["away_runs"]:
        winner = record["home_id"]
    elif record["away_runs"] > record["home_runs"]:
        winner = record["away_id"]
    else:
        winner = None
    state["games"][record["series_game_number"]] = {
        "winner": winner,
        "pen_outs": {
            record["home_id"]: int(record["home_highlev_outs"]),
            record["away_id"]: int(record["away_highlev_outs"]),
        },
    }
    if record["series_game_number"] == 1:
        state["game1_winner"] = winner

    order = (record["game_date"], record["game_number"], record["game_pk"])
    for side, opp in (("home", "away"), ("away", "home")):
        starter = record.get(f"{side}_starter")
        if starter:
            hist.pitcher[int(starter["id"])].append(
                {
                    "date": record["game_date"],
                    "game_number": record["game_number"],
                    "game_pk": record["game_pk"],
                    "is_home": side == "home",
                    "outs": int(starter.get("outs") or 0),
                    "pitches": int(starter.get("pitches") or 0),
                    "er": int(starter.get("er") or 0),
                    "h": int(starter.get("h") or 0),
                    "bb": int(starter.get("bb") or 0),
                    "hbp": int(starter.get("hbp") or 0),
                    "k": int(starter.get("k") or 0),
                    "hr": int(starter.get("hr") or 0),
                    "bf": int(starter.get("bf") or 0),
                }
            )
        opp_starter = record.get(f"{opp}_starter") or {}
        hist.team[record[f"{side}_id"]].append(
            {
                "date": record["game_date"],
                "game_number": record["game_number"],
                "game_pk": record["game_pk"],
                "runs_scored": int(record[f"{side}_runs"]),
                "runs_allowed": int(record[f"{opp}_runs"]),
                "opp_hand": opp_starter.get("hand"),
                "ab": int(opp_starter.get("ab") or 0),
                "h": int(opp_starter.get("h") or 0),
                "bb": int(opp_starter.get("bb") or 0),
                "hbp": int(opp_starter.get("hbp") or 0),
                "sf": int(opp_starter.get("sf") or 0),
                "doubles": int(opp_starter.get("doubles") or 0),
                "triples": int(opp_starter.get("triples") or 0),
                "hr": int(opp_starter.get("hr") or 0),
                "order": order,
            }
        )
        bullpen = record.get(f"{side}_bullpen") or {}
        hist.bullpen[record[f"{side}_id"]].append(
            {
                "date": record["game_date"],
                "game_number": record["game_number"],
                "outs": int(bullpen.get("outs") or 0),
                "er": int(bullpen.get("er") or 0),
                "k": int(bullpen.get("k") or 0),
                "order": order,
            }
        )
        add_counts(hist.league, record.get(f"{side}_staff"))
    hist.park_runs[record["venue_id"]] += int(record["home_runs"]) + int(record["away_runs"])
    hist.park_games[record["venue_id"]] += 1
    hist.league_runs += int(record["home_runs"]) + int(record["away_runs"])
    hist.league_games += 1


def prior_rows(rows: list[dict], game_date: date, game_number: int, days: int | None = None, year: int | None = None):
    picked = []
    for row in rows:
        if not is_before(row["date"], row["game_number"], game_date, game_number):
            continue
        if year is not None and row["date"].year != year:
            continue
        if days is not None and row["date"] < game_date - timedelta(days=days):
            continue
        picked.append(row)
    return picked


def fip_of(starts: list[dict], const: float) -> float | None:
    if not starts:
        return None
    return fip_from_counts(
        sum(item["outs"] for item in starts),
        sum(item["hr"] for item in starts),
        sum(item["bb"] for item in starts),
        sum(item["hbp"] for item in starts),
        sum(item["k"] for item in starts),
        const,
    )


def rate(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def bullpen_ip_yesterday(hist: History, team_id: int, game_date: date) -> float:
    yesterday = game_date - timedelta(days=1)
    outs = 0
    for item in hist.bullpen[team_id]:
        if item["date"] == yesterday:
            outs += int(item["outs"])
    return outs_to_ip(outs)


def side_features(hist: History, record: dict, side: str) -> dict:
    team_id = record[f"{side}_id"]
    starter = record.get(f"{side}_starter") or {}
    starter_id = starter.get("id")
    game_date = record["game_date"]
    game_number = record["game_number"]
    starts = []
    if starter_id is not None:
        starts = prior_rows(hist.pitcher[int(starter_id)], game_date, game_number)
    last = starts[-1] if starts else None
    team_14 = prior_rows(hist.team[team_id], game_date, game_number, days=14)
    pen = prior_rows(hist.bullpen[team_id], game_date, game_number, days=14)
    pen_outs = sum(item["outs"] for item in pen)
    return {
        "starter_last_ip": outs_to_ip(last["outs"]) if last else None,
        "starter_rest_days": (game_date - last["date"]).days if last else None,
        "bullpen_era_14": rate(sum(item["er"] for item in pen) * 27, pen_outs),
        "bullpen_ip_yesterday": bullpen_ip_yesterday(hist, team_id, game_date),
        "rs_14": float(sum(item["runs_scored"] for item in team_14)),
        "ra_14": float(sum(item["runs_allowed"] for item in team_14)),
    }


def game_context(hist: History, record: dict) -> dict:
    state = hist.series.get(series_key(record))
    number = int(record["series_game_number"])
    if number <= 1:
        game1_played = 0
        home_won_game1 = 0
    else:
        game1_played = 1 if state and state.get("game1_winner") else 0
        home_won_game1 = 1 if state and state.get("game1_winner") == record["home_id"] else 0
    return {
        "series_game_number": number,
        "game1_played": game1_played,
        "home_won_game1": home_won_game1,
    }


def prior_lookup(ext: dict, starter: dict | None) -> dict:
    if not starter or not ext:
        return {}
    found = dict((ext.get("pitcher_prior") or {}).get(int(starter["id"]), {}))
    by_name = (ext.get("pitcher_prior") or {}).get(("name", norm_name(starter.get("name") or "")))
    if by_name:
        for key, value in by_name.items():
            found.setdefault(key, value)
    return found


def attach_pybaseball(row: dict, record: dict, ext: dict) -> None:
    if record["game_date"].year < 2026:
        return
    have = ext.get("have") or set()
    if {"FIP", "xFIP", "K%", "BB%"} & set(have):
        home_prior = prior_lookup(ext, record.get("home_starter"))
        away_prior = prior_lookup(ext, record.get("away_starter"))
        if "FIP" in have:
            row["home_starter_fip"] = home_prior.get("fip")
            row["away_starter_fip"] = away_prior.get("fip")
        if "K%" in have:
            row["home_starter_k_pct"] = home_prior.get("k_pct")
            row["away_starter_k_pct"] = away_prior.get("k_pct")
        if "BB%" in have:
            row["home_starter_bb_pct"] = home_prior.get("bb_pct")
            row["away_starter_bb_pct"] = away_prior.get("bb_pct")
        if "xFIP" in have:
            row["home_starter_xfip"] = home_prior.get("xfip")
            row["away_starter_xfip"] = away_prior.get("xfip")
    if "OPS_vs_hand" in have:
        home_hand = (record.get("home_starter") or {}).get("hand")
        away_hand = (record.get("away_starter") or {}).get("hand")
        ops = ext.get("lineup_ops") or {}
        row["home_lineup_ops_vs_hand"] = ops.get((canon_team(record["home_name"]), away_hand))
        row["away_lineup_ops_vs_hand"] = ops.get((canon_team(record["away_name"]), home_hand))
    if "wRC+_vs_hand" in have:
        home_hand = (record.get("home_starter") or {}).get("hand")
        away_hand = (record.get("away_starter") or {}).get("hand")
        wrc = ext.get("lineup_wrc") or {}
        row["home_lineup_wrc_vs_hand"] = wrc.get((canon_team(record["home_name"]), away_hand))
        row["away_lineup_wrc_vs_hand"] = wrc.get((canon_team(record["away_name"]), home_hand))
    if "park_factor" in have:
        row["park_factor"] = (ext.get("park") or {}).get(canon_team(record["home_name"]))


def row_from_record(hist: History, record: dict, book: dict, ext: dict, include_targets: bool) -> dict:
    context = game_context(hist, record)
    home = side_features(hist, record, "home")
    away = side_features(hist, record, "away")
    row = {
        "game_pk": record["game_pk"],
        "game_date": record["game_date"],
        "game_number": record["game_number"],
        "game_type": record["game_type"],
        "away": record["away_name"],
        "home": record["home_name"],
        "away_id": record["away_id"],
        "home_id": record["home_id"],
        "starter_away": (record.get("away_starter") or {}).get("name"),
        "starter_home": (record.get("home_starter") or {}).get("name"),
        "starter_away_id": (record.get("away_starter") or {}).get("id"),
        "starter_home_id": (record.get("home_starter") or {}).get("id"),
        "series_game_number": context["series_game_number"],
        "game1_played": context["game1_played"],
        "home_won_game1": context["home_won_game1"],
    }
    for key, value in home.items():
        row[f"home_{key}"] = value
    for key, value in away.items():
        row[f"away_{key}"] = value
    row["home_judge_out"] = absence_flag(book, "Aaron Judge", record["home_name"], record["game_date"])
    row["away_judge_out"] = absence_flag(book, "Aaron Judge", record["away_name"], record["game_date"])
    row["home_anthony_out"] = absence_flag(book, "Roman Anthony", record["home_name"], record["game_date"])
    row["away_anthony_out"] = absence_flag(book, "Roman Anthony", record["away_name"], record["game_date"])
    row["home_estrada_out"] = absence_flag(book, "Jeremiah Estrada", record["home_name"], record["game_date"])
    row["away_estrada_out"] = absence_flag(book, "Jeremiah Estrada", record["away_name"], record["game_date"])
    attach_pybaseball(row, record, ext)
    if include_targets:
        row["away_runs"] = int(record["away_runs"])
        row["home_runs"] = int(record["home_runs"])
        if record["home_runs"] == record["away_runs"]:
            row["home_win"] = np.nan
        else:
            row["home_win"] = int(record["home_runs"] > record["away_runs"])
    return row


def build_table(records: list[dict], book: dict, ext: dict) -> list[dict]:
    hist = History()
    rows = []
    records = sorted(records, key=lambda rec: (rec["game_date"], rec["game_number"], rec["game_pk"]))
    index = 0
    while index < len(records):
        nxt = index + 1
        while (
            nxt < len(records)
            and records[nxt]["game_date"] == records[index]["game_date"]
            and records[nxt]["game_number"] == records[index]["game_number"]
        ):
            nxt += 1
        batch = records[index:nxt]
        rows.extend(row_from_record(hist, rec, book, ext, True) for rec in batch)
        for rec in batch:
            update_history(hist, rec)
        index = nxt
    return rows, hist


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = [f"home_{name}" for name in SIDE_STATS] + [f"away_{name}" for name in SIDE_STATS] + list(GAME_STATS)
    for name in PYB_SIDE:
        for side in ("home", "away"):
            col = f"{side}_{name}"
            if col in frame.columns and frame[col].notna().any():
                cols.append(col)
    for name in PYB_GAME:
        if name in frame.columns and frame[name].notna().any():
            cols.append(name)
    return cols


def scale_fit(train: pd.DataFrame):
    med = train.median(numeric_only=True)
    filled = train.fillna(med).fillna(0.0)
    mu = filled.mean()
    sd = filled.std(ddof=0).replace(0, 1.0)
    return med, mu, sd


def scale_apply(frame: pd.DataFrame, med, mu, sd) -> pd.DataFrame:
    filled = frame.fillna(med).fillna(0.0)
    return (filled - mu) / sd


def lineup_source(ext: dict, raw_home, raw_away) -> str:
    source = "mlb_boxscore_vs_starter_hand"
    if (ext.get("team_wrc") or {}):
        source += "+fangraphs_wrc_2025_prior_on_2026"
    if ext.get("pitcher_prior"):
        source += "+savant_2025_prior_on_2026"
    if pd.isna(raw_home) or pd.isna(raw_away):
        source += "_partial_impute"
    return source


def write_notes(summary: dict) -> None:
    skipped = summary.get("skipped") or ["none"]
    added = summary.get("used") or ["none"]
    body = [
        "# Notes",
        "",
        "## Sources used",
        "",
        "FanGraphs was not called. The previous run returned HTTP 403.",
        "",
        "Sources used: " + ", ".join(added) + ".",
        "",
        "Schedule and final box scores remain the MLB Stats API. 2025 Savant and MLB split totals are joined only onto 2026 games.",
        "",
        "## Fields added",
        "",
    ]
    body.extend(f"- {item}" for item in added)
    body.extend(["", "## Fields still skipped", ""])
    body.extend(f"- {item}" for item in skipped)
    body.extend(
        [
            "",
            "## Data",
            "",
            "The training table is league-wide completed games from 2025-03-01. Pre-2025 rows were dropped. The two-game file is only a slice of the full table.",
            "",
            f"Minimum date: {summary['data_min']}",
            f"Maximum date: {summary['data_max']}",
            f"Row count: {summary['rows']}",
            "",
            f"Features: {summary['feature_count']}. September accuracy: {summary['accuracy']:.4f}.",
            "",
            "## Bugs fixed",
            "",
        ]
    )
    body.extend(f"- {item}" for item in BUGS_FIXED)
    body.extend(["", "this model does not price sportsbook props and is not a bet.", ""])
    NOTES_PATH.write_text("\n".join(body), encoding="utf-8")


def delete_old_cache() -> None:
    for relative in ("blended_stats.csv", "legacy/stats.csv", "legacy/years_completed.json"):
        path = ROOT / relative
        if path.exists():
            path.unlink()
            note(f"deleted pre-2025 cache {relative}")


def align_series(meta: dict, records: list[dict]) -> None:
    if int(meta.get("series_game_number") or 1) <= 1:
        return
    previous = [
        rec
        for rec in records
        if rec["home_id"] == meta["home_id"]
        and rec["away_id"] == meta["away_id"]
        and rec["game_date"] < meta["game_date"]
        and rec["game_date"].year == meta["game_date"].year
        and int(rec["series_game_number"]) == 1
    ]
    if not previous:
        return
    previous.sort(key=lambda rec: (rec["game_date"], rec["game_number"], rec["game_pk"]))
    description = previous[-1]["series_description"]
    if description and description != meta["series_description"]:
        note(
            f"series description aligned {meta['away_name']} at {meta['home_name']} "
            f"from {meta['series_description']!r} to {description!r}"
        )
        meta["series_description"] = description


def unfinished_on(schedule: list[dict], day: date) -> list[dict]:
    rows = []
    for game in schedule:
        if game["game_date"] != day or game["game_type"] not in GAME_TYPES:
            continue
        if game["abstract"] == "Final":
            continue
        if game["detailed"] in {"Postponed", "Cancelled", "Suspended"}:
            continue
        rows.append(game)
    rows.sort(key=lambda game: (game["game_date"], game["game_number"], game["game_pk"]))
    return rows


def require_all_teams(names: set[str]) -> None:
    missing = [team for team in EXPECTED_TEAMS if team not in names]
    extra = sorted(name for name in names if name not in EXPECTED_TEAMS)
    if missing or extra:
        for team in missing:
            note(f"missing team {team}")
        for team in extra:
            note(f"unexpected team {team}")
        raise SystemExit(2)


def wrc_plus_2026(records: list[dict]) -> dict[str, int]:
    """Team wRC+ from 2026 box-score batting lines in this table.

    FanGraphs team batting is not available (HTTP 403). Baseball Reference
    publishes OPS+, which is not wRC+. Run values are the least-squares fit of
    runs on singles, doubles, triples, home runs, walks, hit batsmen, and outs
    in these 2026 games. Weights are league-wide. Each club is then park-adjusted
    with its 2026 home run environment, and the index is centered so 100 is the
    plate-appearance-weighted mean.
    """
    singles_l = []
    doubles_l = []
    triples_l = []
    hr_l = []
    bb_l = []
    hbp_l = []
    outs_l = []
    runs_l = []
    teams = []
    venues = []
    skipped = 0
    for record in records:
        if record["game_date"].year != 2026:
            continue
        for staff_key, team_key, runs_key in (
            ("away_staff", "home_name", "home_runs"),
            ("home_staff", "away_name", "away_runs"),
        ):
            staff = record.get(staff_key) or {}
            hits = int(staff.get("h") or 0)
            doubles = int(staff.get("doubles") or 0)
            triples = int(staff.get("triples") or 0)
            hr = int(staff.get("hr") or 0)
            singles = hits - doubles - triples - hr
            bb = int(staff.get("bb") or 0)
            hbp = int(staff.get("hbp") or 0)
            sf = int(staff.get("sf") or 0)
            ab = int(staff.get("ab") or 0)
            bf = int(staff.get("bf") or 0)
            outs = bf - hits - bb - hbp
            if singles < 0 or outs < 0 or ab + bb + hbp + sf <= 0:
                skipped += 1
                continue
            singles_l.append(singles)
            doubles_l.append(doubles)
            triples_l.append(triples)
            hr_l.append(hr)
            bb_l.append(bb)
            hbp_l.append(hbp)
            outs_l.append(outs)
            runs_l.append(int(record[runs_key]))
            teams.append(canon_team(record[team_key]))
            venues.append(record["venue_id"])
    if skipped:
        note(f"wRC+ skipped {skipped} team-games with incomplete batting lines")
    n = len(runs_l)
    if n < 100:
        note(f"wRC+ has {n} team-games; stopping")
        raise SystemExit(2)
    singles_a = np.asarray(singles_l, dtype=float)
    doubles_a = np.asarray(doubles_l, dtype=float)
    triples_a = np.asarray(triples_l, dtype=float)
    hr_a = np.asarray(hr_l, dtype=float)
    bb_a = np.asarray(bb_l, dtype=float)
    hbp_a = np.asarray(hbp_l, dtype=float)
    outs_a = np.asarray(outs_l, dtype=float)
    runs_a = np.asarray(runs_l, dtype=float)
    # No intercept: outs are nearly one per inning, so an intercept steals their coefficient.
    design = np.column_stack([singles_a, doubles_a, triples_a, hr_a, bb_a, hbp_a, outs_a])
    coef, _, _, _ = np.linalg.lstsq(design, runs_a, rcond=None)
    note(
        "wRC+ run values "
        f"1B={coef[0]:.3f} 2B={coef[1]:.3f} 3B={coef[2]:.3f} HR={coef[3]:.3f} "
        f"BB={coef[4]:.3f} HBP={coef[5]:.3f} out={coef[6]:.3f}"
    )
    ordered = coef[0] < coef[1] < coef[2] < coef[3]
    if not (0.2 <= coef[0] <= 0.9 and 0.9 <= coef[3] <= 2.5 and coef[6] < 0 and ordered and 0 < coef[4] < coef[0]):
        note("wRC+ run values are outside a baseball range; stopping")
        raise SystemExit(2)
    predicted = design @ coef
    home_games: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    venue_runs: dict = defaultdict(int)
    venue_games: dict = defaultdict(int)
    for record in records:
        if record["game_date"].year != 2026:
            continue
        venue_runs[record["venue_id"]] += int(record["home_runs"]) + int(record["away_runs"])
        venue_games[record["venue_id"]] += 1
        home_games[canon_team(record["home_name"])][record["venue_id"]] += 1
    league_games = sum(venue_games.values())
    league_runs = sum(venue_runs.values())
    if league_games < 100 or league_runs <= 0:
        note("wRC+ park sample is too small; stopping")
        raise SystemExit(2)
    league_rpg = league_runs / league_games
    park_of = {}
    for team, counts in home_games.items():
        venue = max(counts, key=counts.get)
        played = venue_games[venue]
        park_of[team] = 100.0 if played < 20 else 100.0 * (venue_runs[venue] / played) / league_rpg
    buckets: dict[str, dict] = defaultdict(lambda: {"pred": 0.0, "pa": 0.0, "runs": 0.0})
    for index, team in enumerate(teams):
        pa = (
            singles_a[index]
            + doubles_a[index]
            + triples_a[index]
            + hr_a[index]
            + bb_a[index]
            + hbp_a[index]
            + outs_a[index]
        )
        slot = buckets[team]
        slot["pred"] += float(predicted[index])
        slot["pa"] += float(pa)
        slot["runs"] += float(runs_a[index])
    total_pa = sum(slot["pa"] for slot in buckets.values())
    total_pred = sum(slot["pred"] for slot in buckets.values())
    total_runs = sum(slot["runs"] for slot in buckets.values())
    if total_pa <= 0 or total_runs <= 0:
        note("wRC+ league plate appearances are zero; stopping")
        raise SystemExit(2)
    lg_rpa = total_runs / total_pa
    raw = {}
    for team, slot in buckets.items():
        wraa = slot["pred"] - total_pred / total_pa * slot["pa"]
        pf = 1.0 + 0.5 * (park_of.get(team, 100.0) / 100.0 - 1.0)
        wrc_pa = (wraa - (pf - 1.0) * lg_rpa * slot["pa"]) / slot["pa"] + lg_rpa
        raw[team] = wrc_pa / lg_rpa * 100.0
    weighted = sum(raw[team] * buckets[team]["pa"] for team in raw) / total_pa
    note(f"wRC+ plate-appearance-weighted mean {weighted:.3f}")
    if not 95.0 <= weighted <= 105.0:
        note("wRC+ league mean is outside 95-105; stopping")
        raise SystemExit(2)
    return {team: int(round(value - weighted + 100.0)) for team, value in raw.items()}


def write_all_teams(records: list[dict], data_min: date, data_max: date, snapshot: date, ext: dict | None = None) -> pd.DataFrame:
    names = {canon_team(record["home_name"]) for record in records} | {
        canon_team(record["away_name"]) for record in records
    }
    require_all_teams(names)
    window_start = snapshot - timedelta(days=14)
    stats = {
        team: {
            "last_14_runs_scored": 0,
            "last_14_runs_allowed": 0,
            "pen_outs": 0,
            "pen_er": 0,
        }
        for team in EXPECTED_TEAMS
    }
    for record in records:
        if record["game_date"].year != 2026:
            continue
        if not (window_start <= record["game_date"] < snapshot):
            continue
        for side, opp in (("home", "away"), ("away", "home")):
            team = canon_team(record[f"{side}_name"])
            slot = stats[team]
            slot["last_14_runs_scored"] += int(record[f"{side}_runs"])
            slot["last_14_runs_allowed"] += int(record[f"{opp}_runs"])
            pen = record.get(f"{side}_bullpen") or {}
            slot["pen_outs"] += int(pen.get("outs") or 0)
            slot["pen_er"] += int(pen.get("er") or 0)
    rows = []
    for team in EXPECTED_TEAMS:
        slot = stats[team]
        if slot["pen_outs"] <= 0:
            note(f"skipped bullpen ERA for {team}: no bullpen innings in the last 14 days")
            bullpen = ""
        else:
            bullpen = f"{(slot['pen_er'] * 27.0 / slot['pen_outs']):.3f}"
        ops = (ext or {}).get("lineup_ops") or {}
        park = (ext or {}).get("park") or {}
        row = {
            "team": team,
            "last_14_runs_scored": slot["last_14_runs_scored"],
            "last_14_runs_allowed": slot["last_14_runs_allowed"],
            "bullpen_era_14d": bullpen,
            "data_min": data_min.isoformat(),
            "data_cutoff": data_max.isoformat(),
        }
        if "OPS_vs_hand" in ((ext or {}).get("have") or set()):
            row["ops_vs_lhp"] = ops.get((team, "L"))
            row["ops_vs_rhp"] = ops.get((team, "R"))
        if "park_factor" in ((ext or {}).get("have") or set()):
            row["park_factor"] = park.get(team)
        rows.append(row)
    columns = ["team", "last_14_runs_scored", "last_14_runs_allowed", "bullpen_era_14d"]
    if rows and "ops_vs_lhp" in rows[0]:
        columns.extend(["ops_vs_lhp", "ops_vs_rhp"])
    if rows and "park_factor" in rows[0]:
        columns.append("park_factor")
    columns.extend(["data_min", "data_cutoff"])
    frame = pd.DataFrame(rows, columns=columns)
    if len(frame) != 30 or frame["team"].nunique() != 30:
        note(f"all-teams file has {len(frame)} rows; stopping")
        raise SystemExit(2)
    ALL_TEAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ALL_TEAMS_PATH, index=False)
    note(f"wrote {ALL_TEAMS_PATH} teams={len(frame)}")
    return frame


def season_fip_const(records: list[dict], year: int) -> float:
    bucket = {"outs": 0, "er": 0, "hr": 0, "bb": 0, "hbp": 0, "k": 0}
    for record in records:
        if record["game_date"].year != year:
            continue
        for side in ("home_staff", "away_staff"):
            add_counts(bucket, record.get(side))
    outs = bucket["outs"]
    if outs < 900:
        return 3.10
    ip = outs_to_ip(outs)
    raw = (13 * bucket["hr"] + 3 * (bucket["bb"] + bucket["hbp"]) - 2 * bucket["k"]) / ip
    return bucket["er"] * 9.0 / ip - raw


def score_prediction_rows(
    frame: pd.DataFrame,
    pred_rows: list[dict],
    ext: dict,
    data_min: date | None = None,
    data_max: date | None = None,
) -> tuple[list[dict], float, list[str]]:
    features = feature_columns(frame)
    model_frame = frame.dropna(subset=["home_win"]).copy()
    model_frame["game_date"] = pd.to_datetime(model_frame["game_date"])
    if "game_number" not in model_frame.columns:
        model_frame["game_number"] = 1
    train = model_frame[model_frame["game_date"] < pd.Timestamp(TRAIN_END)].sort_values(
        ["game_date", "game_number", "game_pk"]
    )
    test = model_frame[model_frame["game_date"] >= pd.Timestamp(TRAIN_END)].sort_values(
        ["game_date", "game_number", "game_pk"]
    )
    if train.empty or test.empty:
        note(f"time split empty train={len(train)} test={len(test)}")
        raise SystemExit(2)
    med, mu, sd = scale_fit(train[features])
    x_train = scale_apply(train[features], med, mu, sd)
    x_test = scale_apply(test[features], med, mu, sd)
    y_train = train["home_win"].astype(int).to_numpy()
    y_test = test["home_win"].astype(int).to_numpy()
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(x_train, y_train)
    accuracy = float((ridge.predict(x_test) == y_test).mean())
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(ridge.decision_function(x_train).reshape(-1, 1), y_train)
    home_runs_model = LinearRegression()
    away_runs_model = LinearRegression()
    home_runs_model.fit(x_train, train["home_runs"].to_numpy())
    away_runs_model.fit(x_train, train["away_runs"].to_numpy())
    note(f"rows {len(frame)}")
    note(f"feature_count {len(features)}")
    note(f"september_accuracy {accuracy:.4f}")
    note(f"train_rows {len(train)} test_rows {len(test)}")
    if not pred_rows:
        return [], accuracy, features
    pred = pd.DataFrame(pred_rows)
    x_pred = scale_apply(pred[features], med, mu, sd)
    probs = calibrator.predict_proba(ridge.decision_function(x_pred).reshape(-1, 1))[:, 1]
    pred_home_runs = np.clip(home_runs_model.predict(x_pred), 0, None)
    pred_away_runs = np.clip(away_runs_model.predict(x_pred), 0, None)
    cutoff = None if data_max is None else data_max.isoformat()
    out_rows = []
    for pos, (_, row) in enumerate(pred.iterrows()):
        prob = float(probs[pos])
        side = row["home"] if prob >= 0.5 else row["away"]
        out_rows.append(
            {
                "date": pd.to_datetime(row["game_date"]).date().isoformat(),
                "away": row["away"],
                "home": row["home"],
                "starter_away": row["starter_away"],
                "starter_home": row["starter_home"],
                "home_win_prob": f"{prob:.4f}",
                "predicted_away_runs": f"{float(pred_away_runs[pos]):.2f}",
                "predicted_home_runs": f"{float(pred_home_runs[pos]):.2f}",
                "model_side": side,
                "data_cutoff": cutoff,
                "unfinished": bool(row.get("unfinished")),
            }
        )
    return out_rows, accuracy, features


def write_remaining(out_rows: list[dict]) -> None:
    REMAINING_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows, columns=list(GAME_COLUMNS)).to_csv(REMAINING_PATH, index=False)
    note(f"wrote {REMAINING_PATH} games={len(out_rows)}")
    for item in out_rows:
        note(
            f"remaining {item['away']} at {item['home']} {item['starter_away']} vs {item['starter_home']} "
            f"p_home={item['home_win_prob']} runs={item['predicted_away_runs']}-{item['predicted_home_runs']} "
            f"side={item['model_side']}"
        )


def max_final(games: list[dict]) -> date | None:
    dates = [game["game_date"] for game in games]
    return max(dates) if dates else None


def confirm_dates(records: list[dict], finals: list[dict]) -> tuple[date, date]:
    if not records:
        note("MIN_DATE missing")
        note("MAX_DATE missing")
        raise SystemExit(2)
    data_min = min(record["game_date"] for record in records)
    data_max = max(record["game_date"] for record in records)
    sched_max = max_final(finals)
    note(f"MIN_DATE {data_min.isoformat()}")
    note(f"MAX_DATE {data_max.isoformat()}")
    note(f"SCHEDULE_MAX_FINAL {sched_max.isoformat() if sched_max else 'missing'}")
    if data_min.year != 2025 or data_max != sched_max or data_max.year != 2026 or data_max.month != 9:
        note("date check failed; stopping before training")
        raise SystemExit(2)
    pre = [record for record in records if record["game_date"] < START]
    if pre:
        note(f"pre-2025 rows still present: {len(pre)}")
        raise SystemExit(2)
    return data_min, data_max


def print_known_finals(records: list[dict], schedule: list[dict]) -> None:
    wanted = (
        (date(2026, 9, 29), "Boston Red Sox", "New York Yankees"),
        (date(2026, 9, 29), "Chicago Cubs", "San Diego Padres"),
        (date(2026, 9, 29), "Chicago White Sox", "Houston Astros"),
        (date(2026, 9, 30), "Chicago White Sox", "Houston Astros"),
        (date(2026, 9, 30), "Philadelphia Phillies", "Atlanta Braves"),
    )
    by_pk = {record["game_pk"]: record for record in records}
    for game_date, away, home in wanted:
        matches = [
            game
            for game in schedule
            if game["game_date"] == game_date and game["away_name"] == away and game["home_name"] == home
        ]
        if not matches:
            note(f"schedule missing {game_date} {away} at {home}")
            continue
        game = matches[0]
        if game["game_pk"] in by_pk:
            row = by_pk[game["game_pk"]]
            note(
                f"included final {game_date} {away} {row['away_runs']} at {home} {row['home_runs']} "
                f"status={game['detailed']}"
            )
        else:
            note(f"skipped not-final {game_date} {away} at {home} status={game['abstract']}/{game['detailed']}")


def train_and_predict(frame: pd.DataFrame, pred_rows: list[dict], ext: dict, data_min: date, data_max: date) -> None:
    out_rows, accuracy, features = score_prediction_rows(frame, pred_rows, ext, data_min, data_max)
    slice_rows = []
    for away_name, home_name in SLICE_MATCHUPS:
        slice_rows.extend(item for item in out_rows if item["away"] == away_name and item["home"] == home_name)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if len(slice_rows) != len(SLICE_MATCHUPS):
        note(f"two-game slice has {len(slice_rows)} rows; expected {len(SLICE_MATCHUPS)}")
        raise SystemExit(2)
    pd.DataFrame(slice_rows, columns=list(GAME_COLUMNS)).to_csv(PRED_PATH, index=False)
    note(f"wrote {PRED_PATH}")
    for item in slice_rows:
        note(
            f"prediction {item['away']} at {item['home']} {item['starter_away']} vs {item['starter_home']} "
            f"p_home={item['home_win_prob']} "
            f"runs={item['predicted_away_runs']}-{item['predicted_home_runs']} side={item['model_side']}"
        )
    write_remaining([item for item in out_rows if item.get("unfinished")])
    keep = [
        "game_pk",
        "game_date",
        "game_type",
        "away",
        "home",
        "away_runs",
        "home_runs",
        "home_win",
        "starter_away",
        "starter_home",
    ] + features
    saved = frame.copy()
    saved["game_date"] = pd.to_datetime(saved["game_date"]).dt.date.astype(str)
    saved[keep].to_csv(TABLE_PATH, index=False)
    note(f"wrote {TABLE_PATH}")
    note(f"wrote {PRED_PATH}")
    delete_old_cache()
    write_notes(
        {
            "data_min": data_min.isoformat(),
            "data_max": data_max.isoformat(),
            "rows": len(frame),
            "feature_count": len(features),
            "accuracy": accuracy,
            "used": ext.get("used") or [],
            "skipped": ext.get("skipped") or [],
        }
    )


def synthetic_game(meta: dict, hands: dict[int, str]) -> dict:
    def starter(side: str) -> dict | None:
        pid = meta.get(f"{side}_probable_id")
        name = meta.get(f"{side}_probable_name")
        if not pid or not name:
            return None
        return {"id": int(pid), "name": name, "hand": hands.get(int(pid)), "outs": 0, "pitches": 0, "bf": 0, "ab": 0, "h": 0, "bb": 0, "hbp": 0, "k": 0, "hr": 0, "sf": 0, "doubles": 0, "triples": 0}

    home_starter = starter("home")
    away_starter = starter("away")
    if not home_starter or not away_starter:
        raise RuntimeError(f"missing probable pitcher for {meta['away_name']} at {meta['home_name']}")
    return {
        "game_pk": meta["game_pk"],
        "game_date": meta["game_date"],
        "game_number": meta["game_number"],
        "game_type": meta["game_type"],
        "venue_id": meta["venue_id"] or meta["home_id"],
        "series_description": meta["series_description"],
        "series_game_number": meta["series_game_number"],
        "home_id": meta["home_id"],
        "away_id": meta["away_id"],
        "home_name": meta["home_name"],
        "away_name": meta["away_name"],
        "home_runs": 0,
        "away_runs": 0,
        "home_starter": home_starter,
        "away_starter": away_starter,
        "home_bullpen": {},
        "away_bullpen": {},
        "home_staff": {},
        "away_staff": {},
        "home_highlev_outs": 0,
        "away_highlev_outs": 0,
    }


def rebuild_history(records: list[dict]) -> History:
    hist = History()
    ordered = sorted(records, key=lambda rec: (rec["game_date"], rec["game_number"], rec["game_pk"]))
    index = 0
    while index < len(ordered):
        nxt = index + 1
        while (
            nxt < len(ordered)
            and ordered[nxt]["game_date"] == ordered[index]["game_date"]
            and ordered[nxt]["game_number"] == ordered[index]["game_number"]
        ):
            nxt += 1
        for rec in ordered[index:nxt]:
            update_history(hist, rec)
        index = nxt
    return hist


def main() -> None:
    self_check()
    started = denver_now()
    cutoff = started.date()
    note(f"script_start_denver {started.isoformat()}")
    note(f"window {START.isoformat()} through finals on or before {cutoff.isoformat()}")
    if cutoff.year != 2026 or cutoff.month != 9:
        note(f"Denver date {cutoff.isoformat()} is not September 2026; stopping")
        raise SystemExit(2)

    ext_holder: dict = {"ext": empty_ext()}
    tx_holder: dict = {"rows": []}

    def ext_worker() -> None:
        try:
            ext_holder["ext"] = pull_pybaseball((2025, 2026))
        except Exception:
            note("pybaseball pull failed:\n" + traceback.format_exc())
            failed = empty_ext()
            for field in ("FIP", "K%", "BB%", "lineup wRC+ versus starter hand", "park factor"):
                skip_field(failed, field, "pybaseball pull failed")
            ext_holder["ext"] = failed

    def tx_worker() -> None:
        try:
            tx_holder["rows"] = download_transactions(START, cutoff)
        except Exception as exc:
            note(f"MLB transactions failed: {exc}")

    ext_thread = threading.Thread(target=ext_worker, daemon=True)
    tx_thread = threading.Thread(target=tx_worker, daemon=True)
    ext_thread.start()
    tx_thread.start()

    records: list[dict] = []
    finals: list[dict] = []
    schedule: list[dict] = []
    for attempt in range(3):
        schedule = fetch_schedule(START, cutoff)
        finals = finals_only(schedule, cutoff)
        note(f"schedule games {len(schedule)} finals {len(finals)} attempt {attempt + 1}")
        records = load_records(finals)
        tx_thread.join(timeout=120)
        confirm = fetch_schedule(START, cutoff)
        confirm_finals = finals_only(confirm, cutoff)
        have = {record["game_pk"] for record in records}
        want = {game["game_pk"] for game in confirm_finals}
        data_max = max(record["game_date"] for record in records)
        sched_max = max_final(confirm_finals)
        note(f"confirm finals {len(confirm_finals)} table_max {data_max} schedule_max {sched_max}")
        if have == want and data_max == sched_max:
            schedule = confirm
            finals = confirm_finals
            break
        note(f"schedule changed during the pull missing={len(want - have)} extra={len(have - want)}")
    else:
        data_min = min((record["game_date"] for record in records), default=None)
        data_max = max((record["game_date"] for record in records), default=None)
        note(f"MIN_DATE {data_min}")
        note(f"MAX_DATE {data_max}")
        note("date check failed; stopping before training")
        raise SystemExit(2)

    ext_thread.join(timeout=480)
    if ext_thread.is_alive():
        note("pybaseball still running after 480s; continuing without those tables")
        ext = empty_ext()
        for field in ("FIP", "K%", "BB%", "lineup wRC+ versus starter hand", "park factor"):
            skip_field(ext, field, "pybaseball was still running after 480s")
    else:
        ext = ext_holder["ext"]
    if tx_thread.is_alive():
        note("transactions still running; absences left unset rather than guessed")
        transactions = []
    else:
        transactions = tx_holder["rows"]

    starter_ids = set()
    for record in records:
        for side in ("home_starter", "away_starter"):
            starter = record.get(side)
            if starter and starter.get("id"):
                starter_ids.add(int(starter["id"]))
    for game in schedule:
        for key in ("home_probable_id", "away_probable_id"):
            if game.get(key):
                starter_ids.add(int(game[key]))
    hands = load_hands(starter_ids)
    attach_hands(records, hands)

    team_names = {game["home_name"] for game in schedule} | {game["away_name"] for game in schedule}
    team_names.add("Oakland Athletics")
    roster = fetch_named_roster()
    book = parse_absences(transactions, team_names, roster)

    rows, _hist = build_table(records, book, ext)
    data_min, data_max = confirm_dates(records, finals)
    print_known_finals(records, schedule)
    write_all_teams(records, data_min, data_max, cutoff, ext)

    hist = rebuild_history(records)
    pred_rows = []
    unfinished = unfinished_on(schedule, cutoff)
    for meta in unfinished:
        align_series(meta, records)
        synthetic = synthetic_game(meta, hands)
        row = row_from_record(hist, synthetic, book, ext, False)
        row["unfinished"] = True
        pred_rows.append(row)
        note(
            f"probable {meta['away_name']} {meta['away_probable_name']} at {meta['home_name']} "
            f"{meta['home_probable_name']} status={meta['abstract']}/{meta['detailed']}"
        )
    covered = {(row["away"], row["home"]) for row in pred_rows}
    for away_name, home_name in SLICE_MATCHUPS:
        if (away_name, home_name) in covered:
            continue
        finals_match = [
            record
            for record in records
            if record["game_date"] == cutoff
            and record["away_name"] == away_name
            and record["home_name"] == home_name
        ]
        if not finals_match:
            note(f"slice missing {away_name} at {home_name} on {cutoff.isoformat()}")
            continue
        row = row_from_record(hist, finals_match[0], book, ext, False)
        row["unfinished"] = False
        pred_rows.append(row)
        note(
            f"slice final {away_name} {(finals_match[0].get('away_starter') or {}).get('name')} "
            f"at {home_name} {(finals_match[0].get('home_starter') or {}).get('name')}"
        )
    if not unfinished:
        note(f"no unfinished games on {cutoff.isoformat()}")

    frame = pd.DataFrame(rows)
    pre = frame[pd.to_datetime(frame["game_date"]) < pd.Timestamp(START)]
    if not pre.empty:
        note(f"pre-2025 rows in frame {len(pre)}; stopping")
        raise SystemExit(2)
    train_and_predict(frame, pred_rows, ext, data_min, data_max)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
