# GUI - eksperymentalne okno podglądu

Proste okno Tkinter pokazujące obraz z kamery z ramkami wokół wykrytych
twarzy. To **etap eksperymentalny**: GUI nie prowadzi jeszcze rozpoznawania
osób ani nie steruje zbieraniem danych. Do pełnego pipeline'u używaj skryptów
`01`–`04` z konsoli (patrz główny [README](../README.md)).

## Wymagania

`tkinter` jest częścią standardowej biblioteki Pythona — **nie instaluj go
przez pip**. Na Windows jest dołączony do instalatora z python.org (opcja
„tcl/tk and IDLE”). Pozostałe zależności instalujesz z pliku
`requirements.txt` w katalogu głównym projektu.

## Użycie

```bash
python gui/main.py
```

## Funkcje

- podgląd obrazu z wybranej kamery,
- rysowanie ramek wokół wykrytych twarzy (YuNet, z zapasem w postaci kaskady
  Haara — patrz `common/face_detector.py`),
- wybór kamery z listy wykrytych urządzeń,
- start/stop podglądu.

Pozostałe pozycje menu (rozpoznawanie, analiza zdjęć, przygotowanie danych)
są oznaczone jako „wkrótce” i jeszcze nie działają.

## Konfiguracja

Konfiguracja aplikacji jest przechowywana w pliku `config.yaml`.
