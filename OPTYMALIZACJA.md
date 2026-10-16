# Optymalizacja projektu "Rozpoznawanie twarzy"

## 1. Architektura i organizacja plików

### 1.1 Problem
- Kod jest rozproszony po wielu plikach (00-04, listuj_kamery.py, pobierz_zdjecia_nieznajomych.py)
- Brak wyraźnego podziału na warstwy (detection, training, inference)
- Brak konfiguracji i środowisk

### 1.2 Rozwiązania
- **Przyjąć architekturę modułową** z wyraźnym podziałem na warstwy:
  ```
  src/
  ├── detection/          # wykrywanie twarzy (OpenCV, MediaPipe)
  ├── training/           # przygotowanie i trening modeli
  ├── inference/          # rozpoznawanie na żywo
  ├── data/               # surowe i przetworzone dane
  ├── common/             # wspólny kod (uruchom_w_venv.py, utils.py)
  ├── config/             # konfiguracje
  └── tests/              # testy
  ```
- **Przyjąć konfigurację** (config.yaml, config.json):
  - parametry detekcji (threshold, aspect ratio)
  - parametry treningu (rozmiary obrazu, augmentacje)
  - parametry rozpoznawania (threshold rozpoznania, imiona osób)
- **Przyjąć środowiska** (.env, .venv, Dockerfile):
  - .env: kluczowe zmienne środowiskowe (GITHUB_TOKEN, API keys)
  - .venv: wirtualne środowisko Pythona
  - Dockerfile: kontenerizacja środowiska

## 2. Wydajność detekcji

### 2.1 Problem
- MediaPipe wykrywa 100-200 klatk na sekundę
- Przy 30 FPS to 30-60 klatek na sekundę
- Przy 1080p to 10-20 MB/s przepływu danych

### 2.2 Rozwiązania
- **Cache obrazów** (np. z biblioteki `cachetools`):
  ```python
  @cached
  def load_face_features(face_id: str) -> np.ndarray:
      # wczytanie z dysku zamiast z bazy danych
  ```
- **AsyncIO** do odczytu z kamery i zapisu do bazy równolegle:
  ```python
  async def main():
      async with asyncio.timeout(10):
          while True:
              klatka = await camera.read()
              features = await face_detector.detect(klatka)
              await save_to_db(features)
  ```
- **Batch processing** - wykrywanie twarzy dla wielu klatek naraz:
  ```python
  def batch_detect(images: List[Image]) -> List[Face]:
      # wykrywanie dla 100 klatek naraz
  ```

## 3. Wydajność treningu

### 3.1 Problem
- Trening na CPU zajmuje godziny
- Brak wyraźnego podziału na etapy

### 3.2 Rozwiązania
- **GPU acceleration** (PyTorch):
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
- **Dzielenie na etapy**:
  1. **Preprocessing** - przygotowanie danych (10-30 min)
  2. **Feature extraction** - ekstrakcja cech (30-60 min)
  3. **Training** - trening modelu (1-2 godziny)
  4. **Evaluation** - weryfikacja (10-30 min)
- **Checkpointing** - zapisywanie modelu co 100 iteracji

## 4. Baza danych

### 4.1 Problem
- Brak bazy danych
- Zapisywanie na dysku jest wolne

### 4.2 Rozwiązania
- **SQLite** (szybkie, bez instalacji):
  ```python
  from sqlalchemy import create_engine
  engine = create_engine("sqlite:///faces.db")
  ```
- **PostgreSQL** (lepsza wydajność przy dużej liczbie osób):
  ```bash
  docker run -d -e POSTGRES_DB=faces -e POSTGRES_USER=faces -e POSTGRES_PASSWORD=secret \
    postgres:17
  ```
- **Redis** (szybki cache):
  ```python
  import redis
  r = redis.from_url("redis://localhost:6379/0")
  ```

## 5. Rozpoznawanie na żywo

### 5.1 Problem
- Brak wyraźnego API
- Brak obsługi błędów

### 5.2 Rozwiązania
- **API rozpoznawania**:
  ```python
  async def recognize_face(image_path: str) -> str | None:
      # rozpoznawanie twarzy
  ```
- **Error handling**:
  ```python
  try:
      result = await api.recognize(image)
  except APIError as e:
      print(f"Nie udało się rozpoznania: {e}")
  except TimeoutError:
      print("Oczekiwanie na odpowiedź...")
  ```

## 6. Monitoring i logowanie

### 6.1 Problem
- Brak logowania
- Brak monitorowania

### 6.2 Rozwiązania
- **Logging**:
  ```python
  import logging
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  )
  logger = logging.getLogger(__name__)
  ```
- **Monitoring**:
  - Grafy FPS, liczba osób wykrytych
  - Liczba błędów
  - Użycie CPU/RAM

## 7. Deployment

### 7.1 Problem
- Brak instrukcji deployment
- Brak kontenerizacji

### 7.2 Rozwiązania
- **Dockerfile**:
  ```dockerfile
  FROM python:3.12-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- **Instrukcja deployment**:
  ```bash
  docker build -t rozpoznawanie-twarz .
  docker run -d -p 8000:8000 rozpoznawanie-twarz
  ```

## 8. Testy

### 8.1 Problem
- Brak testów
- Brak CI/CD

### 8.2 Rozwiązania
- **Testy**:
  ```python
  # tests/test_detection.py
  def test_detect_face(image_path: str) -> None:
      faces = detect_face(image_path)
      assert len(faces) > 0
  ```
- **CI/CD**:
  ```yaml
  # .github/workflows/ci.yml
  name: CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
        - run: pytest
  ```

## 9. Dokumentacja

### 9.1 Problem
- Brak dokumentacji
- Brak instrukcji użycia

### 9.2 Rozwiązania
- **README.md**:
  - Instalacja
  - Użycie
  - Konfiguracja
  - Rozwiązywanie problemów
- **Dokumentacja API**:
  ```python
  # src/inference/api.py
  @app.post("/recognize")
  async def recognize(image: Image):
      ...
  ```
- **Dokumentacja Python**:
  ```python
  # src/detection/__init__.py
  __all__ = ["detect_face", "detect_faces"]
  ```

## 10. Bezpieczeństwo

### 10.1 Problem
- Brak zabezpieczeń
- Brak logowania błędów

### 10.2 Rozwiązania
- **Zabezpieczenie API**:
  ```python
  @app.post("/recognize")
  async def recognize(image: Image, token: str):
      if not verify_token(token):
          raise HTTPException(401, "Nieprawidłowe hasło")
      ...
  ```
- **Logowanie błędów**:
  ```python
  logger.error(f"Error: {e}", exc_info=True)
  ```
- **Monitoring**:
  - Liczba błędów
  - Czas odpowiedzi
  - Użycie zasobów

## 11. Wnioski

1. **Przyjąć architekturę modułową** - ułatwia to rozwój i utrzymanie
2. **Przyjąć konfigurację** - ułatwia to deployment i testowanie
3. **Przyjąć środowiska** - ułatwia to deployment i testowanie
4. **Przyjąć monitoring i logowanie** - ułatwia to debugowanie i utrzymanie
5. **Przyjąć testy** - ułatwia to utrzymanie i rozwój
6. **Przyjąć dokumentację** - ułatwia to użycie i utrzymanie
