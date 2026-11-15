"""
Awaryjne GUI w Tkinterze - wybór kamery i podgląd detekcji twarzy.

To starsza wersja panelu, zachowana jako fallback na wypadek problemów z
WebView2. Główny interfejs to gui/main.py (okno desktopowe z wyglądem
AdminLTE/Bootstrap).

Użycie:
    python gui/tkinter_app.py

Uwaga: ten panel pokazuje DETEKCJĘ twarzy (gdzie jest twarz), a nie
rozpoznawanie osób. Pełne rozpoznawanie z wytrenowanymi wzorcami uruchamia
skrypt 04_rozpoznawanie_na_zywo.py.
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Uruchamiany jako "python gui/main.py" Python dodaje do sys.path folder gui/,
# a nie katalog projektu - dlatego sami dodajemy katalog główny, żeby import
# "common.*" działał niezależnie od miejsca uruchomienia.
KORZEN_PROJEKTU = Path(__file__).resolve().parent.parent
if str(KORZEN_PROJEKTU) not in sys.path:
    sys.path.insert(0, str(KORZEN_PROJEKTU))

from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

# Musi być przed importem cv2/Pillow - patrz common/uruchom_w_venv.py.
przelacz_na_venv_jesli_trzeba()

import cv2
from PIL import Image, ImageTk

from common.camera import wykryj_dostepne_kamery, otworz_kamere
from src.detection.detector import detect_face

SZEROKOSC_KAMERY = 1280
WYSOKOSC_KAMERY = 720
ROZMIAR_PODGLADU = (800, 450)


class MainApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Rozpoznawanie twarzy - detekcja na żywo")
        self.root.geometry("900x700")
        self.root.minsize(700, 560)

        # Mapa "etykieta z listy" -> indeks kamery w systemie.
        self.kamera_na_indeks: dict[str, int] = {}
        self._uruchomiona = False
        self._watek: threading.Thread | None = None
        self._kolejka: queue.Queue = queue.Queue(maxsize=1)
        self._zdjecie_podgladu: ImageTk.PhotoImage | None = None
        self._prog_detekcji = 0.6

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._exit)

        # Wykrywanie kamer otwiera urządzenia, więc robimy je dopiero po
        # pokazaniu okna - inaczej użytkownik widzi "zamrożoną" aplikację.
        self.root.after(100, self._odswiez_liste_kamer)
        self.root.after(30, self._odswiez_podglad)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        menu_frame = tk.Frame(self.root)
        menu_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        function_frame = tk.LabelFrame(menu_frame, text="Funkcja", padx=10, pady=10)
        function_frame.pack(fill=tk.X, pady=5)

        self.function_var = tk.StringVar(value="Detekcja na żywo")
        self.function_combo = ttk.Combobox(
            function_frame,
            textvariable=self.function_var,
            state="readonly",
            values=[
                "Detekcja na żywo",
                "Analiza zdjęć (w przygotowaniu)",
                "Przygotowanie danych (w przygotowaniu)",
                "Konfiguracja",
            ],
        )
        self.function_combo.pack(fill=tk.X, pady=5)

        camera_frame = tk.LabelFrame(menu_frame, text="Kamera", padx=10, pady=10)
        camera_frame.pack(fill=tk.X, pady=5)

        self.camera_var = tk.StringVar(value="Szukam kamer...")
        self.camera_combo = ttk.Combobox(
            camera_frame,
            textvariable=self.camera_var,
            state="readonly",
            values=["Szukam kamer..."],
        )
        self.camera_combo.pack(fill=tk.X, pady=5)

        detection_config_frame = tk.LabelFrame(
            menu_frame, text="Parametry detekcji", padx=10, pady=10
        )
        detection_config_frame.pack(fill=tk.X, pady=5)

        tk.Label(detection_config_frame, text="Próg pewności (0-1):").pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value="0.6")
        self.threshold_entry = tk.Entry(
            detection_config_frame, textvariable=self.threshold_var, width=8
        )
        self.threshold_entry.pack(side=tk.LEFT, padx=(5, 15))

        tk.Label(detection_config_frame, text="Min. proporcja szer./wys.:").pack(side=tk.LEFT)
        self.aspect_ratio_var = tk.StringVar(value="0.5")
        self.aspect_ratio_entry = tk.Entry(
            detection_config_frame, textvariable=self.aspect_ratio_var, width=8
        )
        self.aspect_ratio_entry.pack(side=tk.LEFT, padx=5)

        detection_frame = tk.Frame(menu_frame)
        detection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_frame = tk.Label(
            detection_frame,
            bg="black",
            text="Podgląd kamery",
            fg="white",
        )
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(detection_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.start_button = tk.Button(
            button_frame, text="Start", command=self._start_detection
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(
            button_frame, text="Stop", command=self._stop_detection, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Gotowe.")
        tk.Label(button_frame, textvariable=self.status_var, anchor="w").pack(
            side=tk.LEFT, padx=15
        )

        action_frame = tk.Frame(menu_frame)
        action_frame.pack(fill=tk.X, pady=10)

        self.analyze_button = tk.Button(
            action_frame, text="Analiza zdjęć", command=self._analyze_images
        )
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        self.prepare_button = tk.Button(
            action_frame, text="Przygotuj dane", command=self._prepare_data
        )
        self.prepare_button.pack(side=tk.LEFT, padx=5)

        self.config_button = tk.Button(
            action_frame, text="Konfiguracja", command=self._configure
        )
        self.config_button.pack(side=tk.LEFT, padx=5)

        self.exit_button = tk.Button(action_frame, text="Wyjście", command=self._exit)
        self.exit_button.pack(side=tk.LEFT, padx=5)

    # ------------------------------------------------------------ kamery
    def _odswiez_liste_kamer(self) -> None:
        self.status_var.set("Szukam dostępnych kamer...")
        self.root.update_idletasks()
        try:
            kamery = wykryj_dostepne_kamery()
        except Exception as blad:  # noqa: BLE001 - błąd sprzętu nie może wywalić GUI
            self.camera_var.set("Brak kamer")
            self.camera_combo["values"] = ["Brak kamer"]
            self.status_var.set(f"Nie udało się wykryć kamer: {blad}")
            return

        self.kamera_na_indeks = {}
        etykiety: list[str] = []
        for kamera in kamery:
            etykieta = f"Kamera {kamera.indeks} ({kamera.maks_szerokosc}x{kamera.maks_wysokosc})"
            if kamera.czy_prawdopodobnie_4k:
                etykieta += " - prawdopodobnie 4K"
            self.kamera_na_indeks[etykieta] = kamera.indeks
            etykiety.append(etykieta)

        if not etykiety:
            etykiety = ["Brak kamer"]
        self.camera_combo["values"] = etykiety
        self.camera_var.set(etykiety[0])
        self.status_var.set(
            f"Znaleziono kamer: {len(kamery)}." if kamery else "Nie znaleziono kamery."
        )

    # ------------------------------------------------------- detekcja
    def _start_detection(self) -> None:
        if self._uruchomiona:
            return

        indeks = self.kamera_na_indeks.get(self.camera_var.get())
        if indeks is None:
            messagebox.showwarning(
                "Wybierz kamerę", "Wybierz kamerę z listy przed uruchomieniem."
            )
            return

        try:
            prog = float(self.threshold_var.get())
            proporcja = float(self.aspect_ratio_var.get())
        except ValueError:
            messagebox.showerror(
                "Niepoprawne parametry",
                "Próg pewności i proporcja muszą być liczbami (np. 0.6 oraz 0.5).",
            )
            return
        if not 0 < prog < 1:
            messagebox.showerror(
                "Niepoprawny próg", "Próg pewności musi być w przedziale (0, 1)."
            )
            return
        if not 0 < proporcja <= 1:
            messagebox.showerror(
                "Niepoprawna proporcja",
                "Minimalna proporcja musi być w przedziale (0, 1].",
            )
            return

        self._prog_detekcji = prog
        self._proporcja = proporcja
        self._uruchomiona = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Uruchamianie kamery...")

        self._watek = threading.Thread(
            target=self._petla_detekcji,
            args=(indeks, prog, proporcja),
            daemon=True,
            name="detekcja-twarzy",
        )
        self._watek.start()

    def _stop_detection(self) -> None:
        if not self._uruchomiona:
            return
        self._uruchomiona = False
        self.status_var.set("Zatrzymywanie...")

    def _petla_detekcji(self, indeks: int, prog: float, proporcja: float) -> None:
        """Pętla robocza - NIE dotyka widgetów Tk bezpośrednio (Tk nie jest
        bezpieczny wątkowo). Klatki przekazuje przez kolejkę do wątku UI."""
        kamera = None
        try:
            kamera = otworz_kamere(
                indeks,
                szerokosc=SZEROKOSC_KAMERY,
                wysokosc=WYSOKOSC_KAMERY,
            )
            while self._uruchomiona:
                czy_odczytano, klatka = kamera.read()
                if not czy_odczytano or klatka is None:
                    raise RuntimeError("Nie udało się odczytać klatki z kamery.")

                klatka = cv2.flip(klatka, 1)
                for twarz in detect_face(
                    klatka, threshold=prog, aspect_ratio=proporcja
                ):
                    x, y, w, h = twarz["bbox"]
                    cv2.rectangle(klatka, (x, y), (x + w, y + h), (0, 200, 0), 2)
                self._wloz_do_kolejki(klatka)
        except Exception as blad:  # noqa: BLE001 - błąd pokazujemy w GUI
            self._wloz_do_kolejki(("blad", str(blad)))
        finally:
            if kamera is not None:
                kamera.release()
            self._wloz_do_kolejki(("koniec", None))

    def _wloz_do_kolejki(self, element) -> None:
        """Wstawia najnowszą klatkę, wyrzucając starszą - podgląd ma pokazywać
        bieżący obraz, a nie kolejkę opóźnionych klatek."""
        try:
            self._kolejka.put_nowait(element)
        except queue.Full:
            try:
                self._kolejka.get_nowait()
            except queue.Empty:
                pass
            try:
                self._kolejka.put_nowait(element)
            except queue.Full:
                pass

    def _odswiez_podglad(self) -> None:
        try:
            element = self._kolejka.get_nowait()
        except queue.Empty:
            element = None

        if isinstance(element, tuple):
            rodzaj, tresc = element
            self._uruchomiona = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            if rodzaj == "blad":
                self.status_var.set(f"Błąd: {tresc}")
                messagebox.showerror("Błąd kamery", tresc)
            else:
                self.status_var.set("Zatrzymano.")
        elif element is not None:
            self._pokaz_klatke(element)

        self.root.after(30, self._odswiez_podglad)

    def _pokaz_klatke(self, klatka) -> None:
        obraz = Image.fromarray(cv2.cvtColor(klatka, cv2.COLOR_BGR2RGB))
        obraz.thumbnail(ROZMIAR_PODGLADU, Image.Resampling.LANCZOS)
        self._zdjecie_podgladu = ImageTk.PhotoImage(obraz)
        self.video_frame.config(image=self._zdjecie_podgladu, text="")

    # ----------------------------------------------------- pozostałe
    def _analyze_images(self) -> None:
        sciezka = filedialog.askopenfilename(
            title="Wybierz zdjęcie",
            filetypes=(
                ("Obrazy JPEG", "*.jpg *.jpeg"),
                ("Obrazy PNG", "*.png"),
                ("Wszystkie pliki", "*.*"),
            ),
        )
        if sciezka:
            messagebox.showinfo(
                "Analiza zdjęć",
                "Ta funkcja nie jest jeszcze zaimplementowana.\n"
                "Na razie użyj skryptu 00_test_detekcji.py lub 04_rozpoznawanie_na_zywo.py.",
            )

    def _prepare_data(self) -> None:
        sciezka = filedialog.askdirectory(title="Wybierz folder ze zdjęciami")
        if sciezka:
            messagebox.showinfo(
                "Przygotowanie danych",
                "Ta funkcja nie jest jeszcze zaimplementowana.\n"
                "Użyj skryptów 01_zbieranie_danych.py i 02_podzial_danych.py.",
            )

    def _configure(self) -> None:
        messagebox.showinfo(
            "Konfiguracja",
            "Progi rozpoznawania osób są zapisane w models/wzorce_osob.json "
            "i kalibrowane skryptem 03_trenowanie_modelu.py.\n"
            "Do szybkiej korekty użyj --korekta-progu w 04_rozpoznawanie_na_zywo.py.",
        )

    def _exit(self) -> None:
        self._uruchomiona = False
        self.root.after(50, self.root.destroy)


def main() -> None:
    root = tk.Tk()
    MainApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
