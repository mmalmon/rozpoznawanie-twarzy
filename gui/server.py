"""Lokalny serwer Flask obsługujący panel GUI.

Serwer nasłuchuje WYŁĄCZNIE na 127.0.0.1 (loopback) i jest uruchamiany
wewnątrz okna desktopowego z gui/main.py. Przeglądarka ani internet nie mają
do niego dostępu z zewnątrz - to celowe, bo endpointy sterują kamerą.
"""

from __future__ import annotations

from flask import Flask, Response, jsonify, render_template, request

from gui.camera_stream import StrumienKamery


def utworz_aplikacje(strumien: StrumienKamery) -> Flask:
    """Tworzy aplikację Flask z endpointami panelu."""
    aplikacja = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    # Polskie znaki mają trafiać do JSON-a bez escapowania \uXXXX.
    aplikacja.json.ensure_ascii = False
    aplikacja.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

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

    @aplikacja.post("/api/start")
    def start():
        dane = request.get_json(silent=True) or {}
        try:
            indeks = int(dane.get("kamera", 0))
            prog = float(dane.get("prog", 0.6))
            proporcja = float(dane.get("proporcja", 0.5))
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "ok": False,
                        "komunikat": "Parametry muszą być liczbami (kamera, próg, proporcja).",
                    }
                ),
                400,
            )

        try:
            strumien.start(indeks, prog, proporcja)
        except ValueError as blad:
            return jsonify({"ok": False, "komunikat": str(blad)}), 400
        except RuntimeError as blad:
            return jsonify({"ok": False, "komunikat": str(blad)}), 409

        return jsonify({"ok": True, "status": strumien.status()})

    @aplikacja.post("/api/stop")
    def stop():
        strumien.stop()
        return jsonify({"ok": True, "status": strumien.status()})

    @aplikacja.errorhandler(404)
    def _brak_strony(blad):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "komunikat": "Nie znaleziono endpointu."}), 404
        return "Nie znaleziono strony.", 404

    return aplikacja
