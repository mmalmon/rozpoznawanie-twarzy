"""Lokalny serwer Flask obsługujący panel GUI.

Serwer nasłuchuje WYŁĄCZNIE na 127.0.0.1 (loopback) i jest uruchamiany
wewnątrz okna desktopowego z gui/main.py. Przeglądarka ani internet nie mają
do niego dostępu z zewnątrz - to celowe, bo endpointy sterują kamerą,
zapisują zdjęcia twarzy i uruchamiają trening.

Endpointy:
  GET  /                          widok panelu
  GET  /video                     strumień MJPEG z kamery
  GET  /api/kamery                lista kamer (skanowanie w tle)
  POST /api/kamery/odswiez        ponowne skanowanie kamer
  GET  /api/status                stan kamery/trybu
  POST /api/start                 start podglądu (tryb, emocje, osoba, cel)
  POST /api/stop                  stop podglądu

  GET  /api/model                 stan silnika rozpoznawania
  POST /api/model/przygotuj       wczytanie modeli w tle
  POST /api/model/emocje          włącz/wyłącz emocje

  POST /api/zbieranie/start       start zbierania twarzy do data/raw
  POST /api/zbieranie/stop        stop zbierania

  GET  /api/trening/status        stan treningu (etap, %, log)
  POST /api/trening/start         uruchom 02 + 03
  POST /api/trening/stop          przerwij trening

  GET  /api/baza                  osoby, liczba zdjęć, informacje o modelu
  GET  /api/baza/<osoba>/zdjecie  przykładowe zdjęcie osoby
  POST /api/baza/usun             usuń osobę (opcjonalnie przetrenuj)

  GET  /api/ewaluacja/status      stan ewaluacji (etap, %, log)
  POST /api/ewaluacja/start       uruchom 07_ewaluacja.py
  POST /api/ewaluacja/stop        przerwij ewaluację
  GET  /api/ewaluacja/raport      ostatni raport JSON
  GET  /api/ewaluacja/wykres/<n>  wykres PNG z models/ewaluacja/wykresy
  GET  /api/ewaluacja/pomylka/<n> zdjęcie z galerii pomyłek
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

from common.dane_osob import (
    FOLDER_PROCESSED,
    FOLDER_RAW,
    NAZWA_KLASY_NIEZNAJOMY,
    PLIK_WZORCOW,
    lista_osob,
    poprawna_nazwa,
    usun_dane_osoby,
)
from gui.camera_stream import (
    TRYB_DETEKCJA,
    TRYB_ROZPOZNAWANIE,
    TRYB_ZBIERANIE,
    TRYBY,
    StrumienKamery,
)
from gui.ewaluacja import (
    FOLDER_POMYLEK,
    FOLDER_WYKRESOW,
    PLIK_RAPORTU,
    MenedzerEwaluacji,
)
from gui.rozpoznawanie import SilnikRozpoznawania
from gui.trening import MenedzerTreningu

ROZSZERZENIA_ZDJEC = {".jpg", ".jpeg", ".png"}
ROZSZERZENIA_GRAFIK = {".png", ".jpg", ".jpeg", ".webp"}


def _bledy_pliku_wzorcow() -> str | None:
    """Szybki test czytelności pliku wzorców (bez wczytywania do silnika)."""
    if not PLIK_WZORCOW.exists():
        return "Brak wytrenowanych wzorców (models/wzorce_osob.json)."
    try:
        json.loads(PLIK_WZORCOW.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as blad:
        return f"Plik wzorców jest uszkodzony: {blad}"
    return None


def _zdjecia_osoby(osoba: str) -> list[Path]:
    folder = FOLDER_RAW / osoba
    if not folder.is_dir():
        return []
    pliki = [
        plik
        for plik in folder.iterdir()
        if plik.is_file() and plik.suffix.lower() in ROZSZERZENIA_ZDJEC
    ]
    return sorted(pliki, key=lambda plik: plik.name)


def _plik_ewaluacji(folder: Path, nazwa: str) -> Path | None:
    """Bezpiecznie zamienia nazwę z URL na plik wewnątrz folderu ewaluacji."""
    if not nazwa:
        return None
    bezpieczna = Path(nazwa).name
    if Path(bezpieczna).suffix.lower() not in ROZSZERZENIA_GRAFIK:
        return None
    sciezka = folder / bezpieczna
    return sciezka if sciezka.is_file() else None


def _opis_bazy() -> dict:
    osoby = lista_osob()
    wynik = []
    for osoba in osoby:
        zdjecia = _zdjecia_osoby(osoba)
        wynik.append(
            {
                "nazwa": osoba,
                "zdjecia": len(zdjecia),
                "przyklady": [
                    f"/api/baza/{osoba}/zdjecie?plik={plik.name}"
                    for plik in list(reversed(zdjecia))[:3]
                ],
            }
        )

    model: dict = {"istnieje": False, "osoby": [], "wymiar": None}
    if PLIK_WZORCOW.exists():
        try:
            dane = json.loads(PLIK_WZORCOW.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            model["blad"] = _bledy_pliku_wzorcow()
        else:
            model = {
                "istnieje": True,
                "osoby": sorted(dane.get("osoby", {})),
                "wymiar": dane.get("wymiar_cech"),
                "wersja": dane.get("wersja_przygotowania_obrazu"),
            }

    return {"osoby": wynik, "model": model}


def utworz_aplikacje(
    strumien: StrumienKamery,
    silnik: SilnikRozpoznawania,
) -> Flask:
    """Tworzy aplikację Flask z endpointami panelu."""
    aplikacja = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    # Polskie znaki mają trafiać do JSON-a bez escapowania \uXXXX.
    aplikacja.json.ensure_ascii = False
    aplikacja.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    def _po_treningu(sukces: bool) -> None:
        """Po udanym treningu przeładowujemy wzorce bez restartu panelu."""
        if not sukces:
            return
        try:
            silnik.przeladuj_wzorce()
        except Exception as blad:  # noqa: BLE001 - komunikat trafia do logu
            menedzer.dopisz_log(f"[uwaga] Nie udało się odświeżyć wzorców: {blad}")

    menedzer = MenedzerTreningu(po_zakonczeniu=_po_treningu)
    # Main.py używa tego przy zamykaniu okna, żeby zabić proces treningu.
    aplikacja.menedzer_treningu = menedzer  # type: ignore[attr-defined]

    menedzer_ewaluacji = MenedzerEwaluacji()
    aplikacja.menedzer_ewaluacji = menedzer_ewaluacji  # type: ignore[attr-defined]

    @aplikacja.after_request
    def _naglowki(odpowiedz):
        odpowiedz.headers.setdefault("X-Content-Type-Options", "nosniff")
        if odpowiedz.mimetype == "application/json":
            odpowiedz.headers.setdefault("Cache-Control", "no-store")
        return odpowiedz

    @aplikacja.get("/")
    def index():
        return render_template("index.html")

    @aplikacja.get("/video")
    def video():
        return Response(
            strumien.strumien(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------- kamery
    @aplikacja.get("/api/kamery")
    def kamery():
        dane = strumien.kamery()
        if not dane["gotowe"]:
            # Pierwsze wejście na stronę uruchamia skanowanie w tle.
            strumien.odswiez_kamery_w_tle()
        return jsonify(dane)

    @aplikacja.post("/api/kamery/odswiez")
    def odswiez_kamery():
        if strumien.status()["dziala"]:
            return (
                jsonify(
                    {
                        "ok": False,
                        "komunikat": "Najpierw zatrzymaj podgląd, żeby odświeżyć listę kamer.",
                    }
                ),
                409,
            )
        strumien.odswiez_kamery_w_tle()
        return jsonify({"ok": True})

    @aplikacja.get("/api/status")
    def status():
        return jsonify(strumien.status())

    # ------------------------------------------------------------- podglad
    @aplikacja.post("/api/start")
    def start():
        dane = request.get_json(silent=True) or {}
        try:
            indeks = int(dane.get("kamera", 0))
            prog = float(dane.get("prog", 0.6))
            proporcja = float(dane.get("proporcja", 0.5))
            cel = int(dane.get("cel", 0))
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "ok": False,
                        "komunikat": "Parametry muszą być liczbami (kamera, próg, proporcja, cel).",
                    }
                ),
                400,
            )

        tryb = str(dane.get("tryb", TRYB_DETEKCJA))
        if tryb not in TRYBY:
            return (
                jsonify({"ok": False, "komunikat": f"Nieznany tryb: {tryb}."}),
                400,
            )
        osoba = str(dane.get("osoba", "")).strip()
        emocje = bool(dane.get("emocje", False))

        try:
            strumien.start(
                indeks,
                prog,
                proporcja,
                tryb=tryb,
                emocje=emocje,
                osoba=osoba,
                cel=cel,
            )
        except ValueError as blad:
            return jsonify({"ok": False, "komunikat": str(blad)}), 400
        except RuntimeError as blad:
            return jsonify({"ok": False, "komunikat": str(blad)}), 409

        return jsonify({"ok": True, "status": strumien.status()})

    @aplikacja.post("/api/stop")
    def stop():
        strumien.stop()
        return jsonify({"ok": True, "status": strumien.status()})

    # ------------------------------------------------------- model/emocje
    @aplikacja.get("/api/model")
    def model():
        return jsonify(silnik.status())

    @aplikacja.post("/api/model/przygotuj")
    def przygotuj_model():
        uruchomiono = silnik.przygotuj_w_tle()
        return jsonify({"ok": True, "uruchomiono": uruchomiono, "status": silnik.status()})

    @aplikacja.post("/api/model/emocje")
    def przelacz_emocje():
        dane = request.get_json(silent=True) or {}
        wlaczone = bool(dane.get("wlaczone", False))
        try:
            silnik.ustaw_emocje(wlaczone)
        except RuntimeError as blad:
            return jsonify({"ok": False, "komunikat": str(blad)}), 400
        strumien.ustaw_emocje(wlaczone)
        return jsonify({"ok": True, "status": silnik.status()})

    @aplikacja.post("/api/model/korekta")
    def korekta_progu():
        dane = request.get_json(silent=True) or {}
        try:
            korekta = float(dane.get("korekta", 0.0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "komunikat": "Korekta musi być liczbą."}), 400
        if not -0.5 <= korekta <= 0.5:
            return (
                jsonify(
                    {"ok": False, "komunikat": "Korekta musi mieścić się w zakresie -0,5 do 0,5."}
                ),
                400,
            )
        silnik.ustaw_korekte_progu(korekta)
        return jsonify({"ok": True, "status": silnik.status()})

    # ---------------------------------------------------------- zbieranie
    @aplikacja.post("/api/zbieranie/start")
    def zbieranie_start():
        dane = request.get_json(silent=True) or {}
        osoba = str(dane.get("osoba", "")).strip()
        if not osoba or not poprawna_nazwa(osoba):
            return (
                jsonify(
                    {
                        "ok": False,
                        "komunikat": (
                            "Podaj nazwę osoby: litery, cyfry, myślnik lub "
                            "podkreślenie (bez spacji i polskich znaków)."
                        ),
                    }
                ),
                400,
            )

        try:
            indeks = int(dane.get("kamera", 0))
            prog = float(dane.get("prog", 0.6))
            proporcja = float(dane.get("proporcja", 0.5))
            cel = int(dane.get("cel", 0))
        except (TypeError, ValueError):
            return (
                jsonify({"ok": False, "komunikat": "Parametry muszą być liczbami."}),
                400,
            )

        try:
            strumien.start(
                indeks,
                prog,
                proporcja,
                tryb=TRYB_ZBIERANIE,
                osoba=osoba,
                cel=cel,
            )
        except ValueError as blad:
            return jsonify({"ok": False, "komunikat": str(blad)}), 400
        except RuntimeError as blad:
            return jsonify({"ok": False, "komunikat": str(blad)}), 409

        return jsonify({"ok": True, "status": strumien.status()})

    @aplikacja.post("/api/zbieranie/stop")
    def zbieranie_stop():
        status = strumien.status()
        if status["tryb"] != TRYB_ZBIERANIE:
            return (
                jsonify({"ok": False, "komunikat": "Zbieranie nie jest uruchomione."}),
                409,
            )
        strumien.stop()
        return jsonify({"ok": True, "status": strumien.status()})

    # ------------------------------------------------------------ trening
    @aplikacja.get("/api/trening/status")
    def trening_status():
        return jsonify(menedzer.status())

    @aplikacja.post("/api/trening/start")
    def trening_start():
        try:
            menedzer.start()
        except RuntimeError as blad:
            return jsonify({"ok": False, "komunikat": str(blad)}), 409
        return jsonify({"ok": True, "status": menedzer.status()})

    @aplikacja.post("/api/trening/stop")
    def trening_stop():
        przerwano = menedzer.zatrzymaj()
        return jsonify(
            {"ok": True, "przerwano": przerwano, "status": menedzer.status()}
        )

    # --------------------------------------------------------- ewaluacja
    @aplikacja.get("/api/ewaluacja/status")
    def ewaluacja_status():
        return jsonify(menedzer_ewaluacji.status())

    @aplikacja.post("/api/ewaluacja/start")
    def ewaluacja_start():
        try:
            menedzer_ewaluacji.start()
        except RuntimeError as blad:
            return jsonify({"ok": False, "komunikat": str(blad)}), 409
        return jsonify({"ok": True, "status": menedzer_ewaluacji.status()})

    @aplikacja.post("/api/ewaluacja/stop")
    def ewaluacja_stop():
        przerwano = menedzer_ewaluacji.zatrzymaj()
        return jsonify(
            {
                "ok": True,
                "przerwano": przerwano,
                "status": menedzer_ewaluacji.status(),
            }
        )

    @aplikacja.get("/api/ewaluacja/raport")
    def ewaluacja_raport():
        if not PLIK_RAPORTU.is_file():
            return jsonify(
                {
                    "dostepny": False,
                    "komunikat": "Brak raportu. Uruchom ewaluację.",
                }
            )
        try:
            dane = json.loads(PLIK_RAPORTU.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as blad:
            return (
                jsonify(
                    {
                        "dostepny": False,
                        "komunikat": f"Raport jest uszkodzony: {blad}",
                    }
                ),
                500,
            )
        dane["dostepny"] = True
        try:
            dane["mtime"] = int(PLIK_RAPORTU.stat().st_mtime)
        except OSError:
            dane["mtime"] = 0
        return jsonify(dane)

    @aplikacja.get("/api/ewaluacja/wykres/<nazwa>")
    def ewaluacja_wykres(nazwa: str):
        sciezka = _plik_ewaluacji(FOLDER_WYKRESOW, nazwa)
        if sciezka is None:
            return jsonify({"ok": False, "komunikat": "Nie ma takiego wykresu."}), 404
        return send_file(sciezka, max_age=0)

    @aplikacja.get("/api/ewaluacja/pomylka/<nazwa>")
    def ewaluacja_pomylka(nazwa: str):
        sciezka = _plik_ewaluacji(FOLDER_POMYLEK, nazwa)
        if sciezka is None:
            return jsonify({"ok": False, "komunikat": "Nie ma takiego zdjęcia."}), 404
        return send_file(sciezka, max_age=0)

    # ---------------------------------------------------------------- baza
    @aplikacja.get("/api/baza")
    def baza():
        dane = _opis_bazy()
        dane["silnik"] = silnik.status()
        return jsonify(dane)

    @aplikacja.get("/api/baza/<osoba>/zdjecie")
    def zdjecie_osoby(osoba: str):
        if not poprawna_nazwa(osoba):
            return jsonify({"ok": False, "komunikat": "Niepoprawna nazwa osoby."}), 400

        zdjecia = _zdjecia_osoby(osoba)
        if not zdjecia:
            return jsonify({"ok": False, "komunikat": "Brak zdjęć tej osoby."}), 404

        zadany = request.args.get("plik", "")
        if zadany:
            # Path(...).name usuwa ewentualne "..\" - nie pozwalamy wyjść
            # poza folder osoby.
            nazwa = Path(zadany).name
            sciezka = FOLDER_RAW / osoba / nazwa
            if not sciezka.is_file():
                return jsonify({"ok": False, "komunikat": "Nie ma takiego zdjęcia."}), 404
        else:
            sciezka = zdjecia[-1]

        return send_file(sciezka, mimetype="image/jpeg", max_age=0)

    @aplikacja.post("/api/baza/usun")
    def usun_osobe():
        dane = request.get_json(silent=True) or {}
        osoba = str(dane.get("osoba", "")).strip()
        if not osoba or not poprawna_nazwa(osoba):
            return jsonify({"ok": False, "komunikat": "Niepoprawna nazwa osoby."}), 400
        if dane.get("potwierdzenie") != osoba:
            return (
                jsonify(
                    {
                        "ok": False,
                        "komunikat": "Brak potwierdzenia - wpisz nazwę osoby, aby usunąć.",
                    }
                ),
                400,
            )

        status = strumien.status()
        if status["dziala"]:
            return (
                jsonify(
                    {
                        "ok": False,
                        "komunikat": "Najpierw zatrzymaj podgląd kamery (przycisk Stop).",
                    }
                ),
                409,
            )

        try:
            usun_dane_osoby(osoba)
        except (OSError, ValueError) as blad:
            return jsonify({"ok": False, "komunikat": str(blad)}), 500

        przetrenowano = False
        if bool(dane.get("przetrenuj", True)):
            pozostale = [
                folder
                for folder in FOLDER_RAW.iterdir()
                if folder.is_dir() and folder.name != NAZWA_KLASY_NIEZNAJOMY
            ] if FOLDER_RAW.exists() else []
            if pozostale:
                try:
                    menedzer.start()
                    przetrenowano = True
                except RuntimeError:
                    przetrenowano = False
            else:
                # Nie została żadna osoba - czyścimy model i podział danych.
                PLIK_WZORCOW.unlink(missing_ok=True)
                shutil.rmtree(FOLDER_PROCESSED, ignore_errors=True)

        return jsonify(
            {
                "ok": True,
                "przetrenowano": przetrenowano,
                "baza": _opis_bazy(),
            }
        )

    @aplikacja.errorhandler(404)
    def _brak_strony(blad):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "komunikat": "Nie znaleziono endpointu."}), 404
        return "Nie znaleziono strony.", 404

    return aplikacja
