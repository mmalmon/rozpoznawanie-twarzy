/* Logika panelu: widoki, API, strumień MJPEG i odświeżanie statystyk. */
(() => {
  'use strict';

  const el = {
    badgeStatus: document.getElementById('badgeStatus'),
    statRow: document.getElementById('statRow'),
    statStatus: document.getElementById('statStatus'),
    statFps: document.getElementById('statFps'),
    statTwarze: document.getElementById('statTwarze'),
    statKamera: document.getElementById('statKamera'),

    sekcjaKamery: document.getElementById('sekcjaKamery'),
    trybBadge: document.getElementById('trybBadge'),
    videoWrap: document.getElementById('videoWrap'),
    video: document.getElementById('video'),
    placeholder: document.getElementById('placeholder'),
    placeholderSpinner: document.getElementById('placeholderSpinner'),
    placeholderTitle: document.getElementById('placeholderTitle'),
    placeholderText: document.getElementById('placeholderText'),

    btnStart: document.getElementById('btnStart'),
    btnStop: document.getElementById('btnStop'),
    statusText: document.getElementById('statusText'),
    kameraSelect: document.getElementById('kameraSelect'),
    btnOdswiezKamery: document.getElementById('btnOdswiezKamery'),

    panelDetekcja: document.getElementById('panelDetekcja'),
    prog: document.getElementById('prog'),
    progWartosc: document.getElementById('progWartosc'),
    proporcja: document.getElementById('proporcja'),
    proporcjaWartosc: document.getElementById('proporcjaWartosc'),

    panelRozpoznawanie: document.getElementById('panelRozpoznawanie'),
    modelBadge: document.getElementById('modelBadge'),
    modelInfo: document.getElementById('modelInfo'),
    przelacznikEmocji: document.getElementById('przelacznikEmocji'),
    korektaProgu: document.getElementById('korektaProgu'),
    korektaProguWartosc: document.getElementById('korektaProguWartosc'),
    listaOsobModelu: document.getElementById('listaOsobModelu'),
    badgeWyniku: document.getElementById('badgeWyniku'),
    wynikiRozpoznawania: document.getElementById('wynikiRozpoznawania'),

    panelDodawanie: document.getElementById('panelDodawanie'),
    osobaNazwa: document.getElementById('osobaNazwa'),
    celZdjec: document.getElementById('celZdjec'),
    licznikZdjec: document.getElementById('licznikZdjec'),
    pasekZbierania: document.getElementById('pasekZbierania'),
    wskazowkaZbierania: document.getElementById('wskazowkaZbierania'),

    viewTrening: document.getElementById('viewTrening'),
    badgeTreningu: document.getElementById('badgeTreningu'),
    etapTreningu: document.getElementById('etapTreningu'),
    procentTreningu: document.getElementById('procentTreningu'),
    pasekTreningu: document.getElementById('pasekTreningu'),
    btnTreningStart: document.getElementById('btnTreningStart'),
    btnTreningStop: document.getElementById('btnTreningStop'),
    czasTreningu: document.getElementById('czasTreningu'),
    alertTreningu: document.getElementById('alertTreningu'),
    alertTreninguBlad: document.getElementById('alertTreninguBlad'),
    logTreningu: document.getElementById('logTreningu'),
    modelWTreningu: document.getElementById('modelWTreningu'),

    viewEwaluacja: document.getElementById('viewEwaluacja'),
    badgeEwaluacji: document.getElementById('badgeEwaluacji'),
    etapEwaluacji: document.getElementById('etapEwaluacji'),
    procentEwaluacji: document.getElementById('procentEwaluacji'),
    pasekEwaluacji: document.getElementById('pasekEwaluacji'),
    btnEwaluacjaStart: document.getElementById('btnEwaluacjaStart'),
    btnEwaluacjaStop: document.getElementById('btnEwaluacjaStop'),
    czasEwaluacji: document.getElementById('czasEwaluacji'),
    alertEwaluacji: document.getElementById('alertEwaluacji'),
    alertEwaluacjiBlad: document.getElementById('alertEwaluacjiBlad'),
    logEwaluacji: document.getElementById('logEwaluacji'),
    podsumowanieEwaluacji: document.getElementById('podsumowanieEwaluacji'),
    dataRaportu: document.getElementById('dataRaportu'),
    tabelaMetryk: document.getElementById('tabelaMetryk'),
    tabelaPorownania: document.getElementById('tabelaPorownania'),
    wykresyEwaluacji: document.getElementById('wykresyEwaluacji'),
    galeriaPomylek: document.getElementById('galeriaPomylek'),
    badgePomylek: document.getElementById('badgePomylek'),
    ostrzezeniaEwaluacji: document.getElementById('ostrzezeniaEwaluacji'),

    viewBaza: document.getElementById('viewBaza'),
    badgeBazy: document.getElementById('badgeBazy'),
    btnOdswiezBaze: document.getElementById('btnOdswiezBaze'),
    listaOsob: document.getElementById('listaOsob'),
    infoModeluBazy: document.getElementById('infoModeluBazy'),

    viewInfo: document.getElementById('viewInfo'),

    modalKamery: document.getElementById('modalKamery'),
    listaKamer: document.getElementById('listaKamer'),
    btnOdswiezKameryModal: document.getElementById('btnOdswiezKameryModal'),

    modalUsun: document.getElementById('modalUsun'),
    usunOsobaNazwa: document.getElementById('usunOsobaNazwa'),
    usunPrzetrenuj: document.getElementById('usunPrzetrenuj'),
    btnPotwierdzUsun: document.getElementById('btnPotwierdzUsun'),

    toastBlad: document.getElementById('toastBlad'),
    toastBladTresc: document.getElementById('toastBladTresc'),
    toastInfo: document.getElementById('toastInfo'),
    toastInfoTresc: document.getElementById('toastInfoTresc'),

    btnSidebar: document.getElementById('btnSidebar'),
    sidebarBackdrop: document.getElementById('sidebarBackdrop'),
    tytulWidoku: document.getElementById('tytulWidoku'),
    podtytulWidoku: document.getElementById('podtytulWidoku'),
  };

  const WIDOKI = {
    detekcja: {
      tytul: 'Detekcja na żywo',
      podtytul: 'Ramki wokół twarzy i statystyki z kamery',
      tryb: 'detekcja',
      kamera: true,
      panel: el.panelDetekcja,
    },
    rozpoznawanie: {
      tytul: 'Rozpoznawanie osób',
      podtytul: 'Sprawdź, kto jest przed kamerą',
      tryb: 'rozpoznawanie',
      kamera: true,
      panel: el.panelRozpoznawanie,
    },
    dodawanie: {
      tytul: 'Dodaj twarz do bazy',
      podtytul: 'Zbierz zdjęcia nowej osoby z kamery',
      tryb: 'zbieranie',
      kamera: true,
      panel: el.panelDodawanie,
    },
    trening: {
      tytul: 'Trening modelu',
      podtytul: 'Podział danych, embeddingi i kalibracja progów',
      kamera: false,
    },
    ewaluacja: {
      tytul: 'Ewaluacja modelu',
      podtytul: 'Accuracy, FAR/FRR, macierz pomyłek i porównanie metod',
      kamera: false,
    },
    baza: {
      tytul: 'Baza twarzy',
      podtytul: 'Podgląd i usuwanie zapisanych osób',
      kamera: false,
    },
    info: {
      tytul: 'O projekcie',
      podtytul: 'Jak korzystać z panelu i o czym pamiętać',
      kamera: false,
    },
  };

  const NAZWY_TRYBOW = {
    detekcja: 'Detekcja',
    rozpoznawanie: 'Rozpoznawanie',
    zbieranie: 'Zbieranie zdjęć',
  };

  const OPIS_STANU = {
    zatrzymany: 'Zatrzymano',
    uruchamianie: 'Uruchamianie…',
    dziala: 'Działa',
    blad: 'Błąd',
  };

  const KLASA_STANU = {
    zatrzymany: 'text-bg-secondary',
    uruchamianie: 'text-bg-warning',
    dziala: 'text-bg-success',
    blad: 'text-bg-danger',
  };

  const OPIS_TRENINGU = {
    bezczynny: ['Bezczynny', 'text-bg-secondary'],
    dziala: ['Trwa trening', 'text-bg-warning'],
    gotowe: ['Gotowe', 'text-bg-success'],
    blad: ['Błąd', 'text-bg-danger'],
    przerwany: ['Przerwano', 'text-bg-secondary'],
  };

  const OPIS_EWALUACJI = {
    bezczynny: ['Bezczynna', 'text-bg-secondary'],
    dziala: ['Trwa ewaluacja', 'text-bg-warning'],
    gotowe: ['Gotowa', 'text-bg-success'],
    blad: ['Błąd', 'text-bg-danger'],
    przerwany: ['Przerwano', 'text-bg-secondary'],
  };

  const OPIS_WYKRESOW = {
    macierz_pomylek: 'Macierz pomyłek',
    rozklad_podobienstw: 'Rozkład podobieństw: swoi vs obcy',
    krzywa_progu: 'Kompromis FAR / FRR',
    porownanie_metod: 'Porównanie metod',
    pca_embeddingi: 'Embeddingi w 2D (PCA)',
  };

  const toastBlad = bootstrap.Toast.getOrCreateInstance(el.toastBlad, { delay: 9000 });
  const toastInfo = bootstrap.Toast.getOrCreateInstance(el.toastInfo, { delay: 6000 });
  const modalUsun = bootstrap.Modal.getOrCreateInstance(el.modalUsun);

  const stan = {
    widok: 'detekcja',
    status: null,
    model: null,
    maKamery: false,
    zajete: false,
    osobaDoUsuniecia: null,
    poprzednieZbieranie: 0,
    ostatniBlad: '',
    odswiezanie: false,
    raportEwaluacji: null,
    czasRaportu: null,
    ostatnieWyniki: '',
  };

  // ------------------------------------------------------------- pomocnicze
  const formatuj = (wartosc, miejsca = 1) =>
    Number(wartosc || 0).toFixed(miejsca).replace('.', ',');

  const formatujProcent = (wartosc, miejsca = 1) =>
    wartosc === null || wartosc === undefined
      ? '—'
      : `${formatuj(wartosc * 100, miejsca)}%`;

  const ogranicz = (wartosc, min = 0, max = 100) =>
    Math.min(max, Math.max(min, Number(wartosc) || 0));

  const escapeHtml = (tekst) =>
    String(tekst).replace(/[&<>"']/g, (znak) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[znak]));

  async function sprawdzOdpowiedz(odpowiedz) {
    let dane = null;
    try {
      dane = await odpowiedz.json();
    } catch (_) {
      /* odpowiedź mogła nie mieć JSON-a */
    }
    if (!odpowiedz.ok) {
      throw new Error((dane && dane.komunikat) || `Błąd serwera (${odpowiedz.status}).`);
    }
    return dane;
  }

  const pobierz = (sciezka) =>
    fetch(sciezka, { cache: 'no-store' }).then(sprawdzOdpowiedz);

  const wyslij = (sciezka, dane) =>
    fetch(sciezka, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(dane || {}),
    }).then(sprawdzOdpowiedz);

  function pokazBlad(tresc) {
    if (!tresc || tresc === stan.ostatniBlad) {
      return;
    }
    stan.ostatniBlad = tresc;
    el.toastBladTresc.textContent = tresc;
    toastBlad.show();
    setTimeout(() => {
      if (stan.ostatniBlad === tresc) {
        stan.ostatniBlad = '';
      }
    }, 10000);
  }

  function pokazInfo(tresc) {
    el.toastInfoTresc.textContent = tresc;
    toastInfo.show();
  }

  // Polska odmiana: 1 zdjęcie, 2 zdjęcia, 5 zdjęć.
  function odmianaZdjęć(liczba) {
    const n = Math.abs(Number(liczba) || 0);
    const reszta10 = n % 10;
    const reszta100 = n % 100;
    if (n === 1) {
      return 'zdjęcie';
    }
    if (reszta10 >= 2 && reszta10 <= 4 && !(reszta100 >= 12 && reszta100 <= 14)) {
      return 'zdjęcia';
    }
    return 'zdjęć';
  }

  // ---------------------------------------------------------------- widoki
  function czyWidokKamery() {
    return Boolean(WIDOKI[stan.widok] && WIDOKI[stan.widok].kamera);
  }

  function opisZatrzymania() {
    if (stan.widok === 'rozpoznawanie') {
      if (stan.model && stan.model.blad) {
        return 'Nie udało się wczytać modelu — szczegóły w panelu obok.';
      }
      if (!stan.model || (!stan.model.gotowy && !stan.model.ladowanie)) {
        return 'Wczytywanie modelu… To może potrwać kilka sekund.';
      }
      if (stan.model.ladowanie) {
        return 'Wczytywanie modelu… To może potrwać kilka sekund.';
      }
      return 'Kliknij „Start”, aby rozpoznawać osoby.';
    }
    if (stan.widok === 'dodawanie') {
      return 'Wpisz nazwę osoby i kliknij „Start”.';
    }
    return 'Wybierz kamerę i kliknij „Start”.';
  }

  function pokazZatrzymany(tytul = 'Podgląd zatrzymany', tekst = null) {
    el.videoWrap.classList.remove('aktywne');
    el.video.removeAttribute('src');
    el.placeholderSpinner.classList.add('d-none');
    el.placeholderTitle.textContent = tytul;
    el.placeholderText.textContent = tekst === null ? opisZatrzymania() : tekst;
  }

  function pokazUruchamianie() {
    el.placeholderSpinner.classList.remove('d-none');
    el.placeholderTitle.textContent = 'Uruchamianie kamery…';
    el.placeholderText.textContent = 'To może potrwać kilka sekund.';
  }

  function pokazPodglad() {
    el.video.src = `/video?t=${Date.now()}`;
    el.videoWrap.classList.add('aktywne');
    el.placeholderSpinner.classList.add('d-none');
  }

  async function przelaczWidok(nazwa) {
    if (!WIDOKI[nazwa] || nazwa === stan.widok) {
      return;
    }

    // Zmiana widoku = inny tryb kamery albo praca bez kamery. Zwalniamy
    // urządzenie, żeby np. trening nie walczył o kamerę z podglądem.
    if (stan.status && stan.status.dziala) {
      await stop(false);
    }

    stan.widok = nazwa;
    const opis = WIDOKI[nazwa];

    document.querySelectorAll('.nav-link[data-view]').forEach((przycisk) => {
      const aktywny = przycisk.dataset.view === nazwa;
      przycisk.classList.toggle('active', aktywny);
      if (aktywny) {
        przycisk.setAttribute('aria-current', 'page');
      } else {
        przycisk.removeAttribute('aria-current');
      }
    });

    el.tytulWidoku.textContent = opis.tytul;
    el.podtytulWidoku.textContent = opis.podtytul;
    el.trybBadge.textContent = NAZWY_TRYBOW[opis.tryb] || 'Detekcja';

    el.sekcjaKamery.classList.toggle('d-none', !opis.kamera);
    el.statRow.classList.toggle('d-none', !opis.kamera);
    el.viewTrening.classList.toggle('d-none', nazwa !== 'trening');
    el.viewEwaluacja.classList.toggle('d-none', nazwa !== 'ewaluacja');
    el.viewBaza.classList.toggle('d-none', nazwa !== 'baza');
    el.viewInfo.classList.toggle('d-none', nazwa !== 'info');

    document.querySelectorAll('.panel-widoku').forEach((panel) => {
      panel.classList.add('d-none');
    });
    if (opis.panel) {
      opis.panel.classList.remove('d-none');
    }

    if (opis.kamera) {
      pokazZatrzymany();
      el.statusText.textContent = 'Gotowe.';
    }

    if (nazwa === 'rozpoznawanie') {
      await zapewnijModel();
    } else if (nazwa === 'trening') {
      await odswiezTrening();
      await odswiezInfoModelu();
    } else if (nazwa === 'ewaluacja') {
      await odswiezEwaluacje();
      await odswiezRaportEwaluacji();
    } else if (nazwa === 'baza') {
      await odswiezBaze();
    }

    zastosujStatus(stan.status || {});
  }

  // ------------------------------------------------------------- kamery
  function renderujListeKamer(dane) {
    if (!dane.gotowe) {
      el.listaKamer.textContent = 'Skanowanie kamer…';
      return;
    }
    if (dane.blad) {
      el.listaKamer.textContent = `Nie udało się wykryć kamer: ${dane.blad}`;
      return;
    }
    if (!dane.lista || dane.lista.length === 0) {
      el.listaKamer.textContent = 'Nie znaleziono żadnej kamery.';
      return;
    }
    el.listaKamer.innerHTML = dane.lista
      .map(
        (kamera) =>
          `<div class="d-flex justify-content-between border rounded p-2">` +
          `<span>Kamera ${kamera.indeks}</span>` +
          `<span>${kamera.szerokosc}×${kamera.wysokosc}${kamera.cztery_k ? ' · 4K' : ''}</span>` +
          `</div>`,
      )
      .join('');
  }

  async function zaladujKamery(proba = 0) {
    try {
      const dane = await pobierz('/api/kamery');
      renderujListeKamer(dane);

      if (!dane.gotowe) {
        el.kameraSelect.innerHTML = '<option value="">Szukam kamer…</option>';
        el.kameraSelect.disabled = true;
        if (proba < 60) {
          setTimeout(() => zaladujKamery(proba + 1), 500);
        }
        return;
      }

      if (dane.blad) {
        pokazBlad(`Nie udało się wykryć kamer: ${dane.blad}`);
      }

      const lista = dane.lista || [];
      stan.maKamery = lista.length > 0;

      if (!stan.maKamery) {
        el.kameraSelect.innerHTML = '<option value="">Brak kamer</option>';
        el.kameraSelect.disabled = true;
        return;
      }

      const wybrany = el.kameraSelect.value;
      el.kameraSelect.innerHTML = lista
        .map(
          (kamera) =>
            `<option value="${kamera.indeks}">` +
            `Kamera ${kamera.indeks} (${kamera.szerokosc}×${kamera.wysokosc})` +
            `${kamera.cztery_k ? ' — prawdopodobnie 4K' : ''}` +
            `</option>`,
        )
        .join('');
      if (wybrany && lista.some((kamera) => String(kamera.indeks) === wybrany)) {
        el.kameraSelect.value = wybrany;
      }
      el.kameraSelect.disabled = Boolean(stan.status && stan.status.dziala);
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  async function odswiezKamery() {
    el.kameraSelect.innerHTML = '<option value="">Szukam kamer…</option>';
    el.kameraSelect.disabled = true;
    stan.maKamery = false;
    try {
      await wyslij('/api/kamery/odswiez');
      await zaladujKamery();
    } catch (blad) {
      pokazBlad(blad.message);
      await zaladujKamery();
    }
  }

  // -------------------------------------------------------------- status
  function zastosujStatus(dane) {
    const stanKamery = dane.stan || 'zatrzymany';
    const opis = OPIS_STANU[stanKamery] || stanKamery;
    const dziala = Boolean(dane.dziala);

    el.badgeStatus.textContent = opis;
    el.badgeStatus.className = `badge ${KLASA_STANU[stanKamery] || 'text-bg-secondary'}`;
    el.statStatus.textContent = opis;
    el.statFps.textContent = formatuj(dane.fps);
    el.statTwarze.textContent = dane.twarze ?? 0;
    el.statKamera.textContent = dziala
      ? `Kamera ${dane.kamera} · ${NAZWY_TRYBOW[dane.tryb] || ''}`
      : '—';

    // Start jest możliwy tylko, gdy znamy kamerę i (w trybie rozpoznawania)
    // model jest wczytany.
    let moznaStartowac = stan.maKamery && !dziala && !stan.zajete;
    if (stan.widok === 'rozpoznawanie' && !(stan.model && stan.model.gotowy)) {
      moznaStartowac = false;
    }
    el.btnStart.disabled = !moznaStartowac;
    el.btnStop.disabled = stan.zajete || !dziala;

    const blokada = dziala || !stan.maKamery;
    el.kameraSelect.disabled = blokada;
    el.prog.disabled = blokada;
    el.proporcja.disabled = blokada;
    el.osobaNazwa.disabled = dziala && dane.tryb === 'zbieranie';
    el.celZdjec.disabled = dziala && dane.tryb === 'zbieranie';

    if (dziala && !el.video.hasAttribute('src')) {
      pokazPodglad();
    }

    if (dane.tryb === 'zbieranie') {
      aktualizujLicznikZbierania(dane);
    }

    if (dziala && dane.tryb === 'zbieranie' && dane.cel > 0 && dane.zapisane >= dane.cel) {
      el.statusText.textContent = `Zebrano ${dane.zapisane} zdjęć. Zatrzymywanie…`;
    }

    if (stanKamery === 'blad' && dane.blad) {
      el.statusText.textContent = 'Błąd kamery.';
      pokazBlad(dane.blad);
      pokazZatrzymany('Błąd kamery', 'Sprawdź, czy kamera nie jest używana przez inny program.');
    }

    renderWynikiRozpoznawania(dane);
  }

  // ------------------------------------------- rozpoznawanie: TOP-3
  const NIEZNANA_OSOBA = 'Nieznana osoba';

  function renderWynikiRozpoznawania(dane) {
    const aktywne = Boolean(dane && dane.dziala && dane.tryb === 'rozpoznawanie');
    const wyniki =
      aktywne && dane.wyniki_aktualne && Array.isArray(dane.wyniki) ? dane.wyniki : [];

    if (!wyniki.length) {
      if (stan.ostatnieWyniki === '') {
        return;
      }
      stan.ostatnieWyniki = '';
      el.badgeWyniku.textContent = aktywne ? 'Brak twarzy' : '—';
      el.badgeWyniku.className = 'badge text-bg-secondary ms-auto';
      el.wynikiRozpoznawania.innerHTML = aktywne
        ? '<p class="small text-muted mb-0">Nie widzę twarzy w kadrze. Ustaw twarz przodem do kamery.</p>'
        : '<p class="small text-muted mb-0">Uruchom podgląd w trybie Rozpoznawanie, aby zobaczyć TOP-3 kandydatów i porównanie z progiem.</p>';
      return;
    }

    const klucz = JSON.stringify(wyniki);
    if (klucz === stan.ostatnieWyniki) {
      return;
    }
    stan.ostatnieWyniki = klucz;

    const rozpoznano = wyniki.some(
      (wynik) => wynik.osoba && wynik.osoba !== NIEZNANA_OSOBA,
    );
    el.badgeWyniku.textContent = rozpoznano ? 'Rozpoznano' : 'Osoba nieznana';
    el.badgeWyniku.className = `badge ms-auto ${
      rozpoznano ? 'text-bg-success' : 'text-bg-danger'
    }`;

    el.wynikiRozpoznawania.innerHTML = wyniki
      .map((wynik, indeks) => {
        const rozpoznana = Boolean(wynik.osoba && wynik.osoba !== NIEZNANA_OSOBA);
        const kandydaci = Array.isArray(wynik.kandydaci)
          ? wynik.kandydaci.slice(0, 3)
          : [];

        const paski = kandydaci.length
          ? kandydaci
              .map((kandydat) => {
                const podobienstwo = Math.max(0, Number(kandydat.podobienstwo) || 0);
                const prog = Math.max(0, Number(kandydat.prog) || 0);
                const klasa = kandydat.akceptacja ? 'bg-success' : 'bg-secondary';
                return (
                  '<div class="kandydat">' +
                  '<div class="d-flex justify-content-between small mb-1 gap-2">' +
                  `<span class="${kandydat.akceptacja ? 'fw-semibold' : ''}">${escapeHtml(kandydat.osoba)}</span>` +
                  '<span class="text-mono text-nowrap">' +
                  `${formatujProcent(kandydat.podobienstwo)} ` +
                  `<span class="text-muted">/ próg ${formatujProcent(kandydat.prog)}</span>` +
                  '</span></div>' +
                  '<div class="podobienstwo-pasek">' +
                  `<div class="progress"><div class="progress-bar ${klasa}" style="width:${ogranicz(podobienstwo * 100)}%"></div></div>` +
                  `<span class="prog-znacznik" style="left:${ogranicz(prog * 100)}%" title="próg ${formatujProcent(kandydat.prog)}"></span>` +
                  '</div></div>'
                );
              })
              .join('')
          : '<div class="small text-muted">Brak listy kandydatów.</div>';

        const emocja = wynik.emocja
          ? `<div class="small text-muted mb-2">Emocja: ${escapeHtml(wynik.emocja)} ` +
            `(${formatujProcent(wynik.pewnosc_emocji)})</div>`
          : '';

        return (
          '<div class="mb-3">' +
          '<div class="d-flex align-items-center gap-2 mb-2">' +
          `<span class="fw-semibold">Twarz ${indeks + 1}</span>` +
          `<span class="badge ${rozpoznana ? 'text-bg-success' : 'text-bg-danger'}">` +
          `${rozpoznana ? escapeHtml(wynik.osoba) : 'Nieznana osoba'}</span>` +
          `<span class="small text-muted ms-auto text-mono">${formatujProcent(wynik.podobienstwo)}</span>` +
          '</div>' +
          emocja +
          paski +
          '</div>'
        );
      })
      .join('<hr class="my-2">');
  }

  async function odswiezStatus() {
    if (stan.odswiezanie) {
      return;
    }
    stan.odswiezanie = true;
    try {
      const dane = await pobierz('/api/status');
      const poprzedni = stan.status;
      stan.status = dane;
      zastosujStatus(dane);

      // Zakończone zbieranie: pokazujemy podsumowanie tylko raz.
      if (
        poprzedni &&
        poprzedni.dziala &&
        poprzedni.tryb === 'zbieranie' &&
        !dane.dziala &&
        dane.zapisane > stan.poprzednieZbieranie
      ) {
        stan.poprzednieZbieranie = dane.zapisane;
        if (dane.zapisane > 0) {
          pokazInfo(
            `Zebrano ${dane.zapisane} ${odmianaZdjęć(dane.zapisane)} osoby „${dane.osoba}”. ` +
              'Możesz uruchomić trening.',
          );
        }
      }
    } catch (_) {
      /* pojedyncze nieudane odświeżenie nie powinno zasypywać użytkownika */
    } finally {
      stan.odswiezanie = false;
    }
  }

  // ---------------------------------------------------------------- start
  function nazwaOsobyPoprawna(nazwa) {
    return /^[A-Za-z0-9_-]+$/.test(nazwa);
  }

  function aktualizujLicznikZbierania(dane) {
    const cel = dane.cel || 0;
    const zapisane = dane.zapisane || 0;
    el.licznikZdjec.textContent = cel ? `${zapisane} / ${cel}` : `${zapisane} / bez limitu`;
    const procent = cel > 0 ? Math.min(100, (zapisane / cel) * 100) : 0;
    el.pasekZbierania.style.width = `${procent}%`;
  }

  async function start() {
    const indeks = Number.parseInt(el.kameraSelect.value, 10);
    if (Number.isNaN(indeks)) {
      pokazBlad('Najpierw wybierz kamerę z listy.');
      return;
    }

    const widok = WIDOKI[stan.widok];
    if (!widok || !widok.kamera) {
      return;
    }

    let daneStartu = {
      kamera: indeks,
      prog: Number.parseFloat(el.prog.value),
      proporcja: Number.parseFloat(el.proporcja.value),
    };

    let adres = '/api/start';

    if (widok.tryb === 'rozpoznawanie') {
      if (!stan.model || !stan.model.gotowy) {
        pokazBlad('Model rozpoznawania jeszcze się wczytuje.');
        return;
      }
      daneStartu.tryb = 'rozpoznawanie';
      daneStartu.emocje = el.przelacznikEmocji.checked;
    } else if (widok.tryb === 'zbieranie') {
      const osoba = el.osobaNazwa.value.trim();
      if (!nazwaOsobyPoprawna(osoba)) {
        el.osobaNazwa.classList.add('is-invalid');
        pokazBlad('Podaj poprawną nazwę osoby (litery, cyfry, myślnik lub podkreślenie).');
        return;
      }
      el.osobaNazwa.classList.remove('is-invalid');

      const cel = Number.parseInt(el.celZdjec.value, 10) || 0;
      if (cel < 10) {
        el.celZdjec.classList.add('is-invalid');
        pokazBlad('Cel musi wynosić co najmniej 10 zdjęć.');
        return;
      }
      el.celZdjec.classList.remove('is-invalid');

      adres = '/api/zbieranie/start';
      daneStartu.osoba = osoba;
      daneStartu.cel = cel;
    } else {
      daneStartu.tryb = 'detekcja';
    }

    stan.zajete = true;
    el.btnStart.disabled = true;
    el.btnStop.disabled = true;
    el.statusText.textContent = 'Uruchamianie kamery…';
    pokazUruchamianie();

    try {
      await wyslij(adres, daneStartu);
      pokazPodglad();
      el.statusText.textContent =
        widok.tryb === 'zbieranie' ? 'Zbieranie zdjęć…' : 'Podgląd działa.';
      stan.poprzednieZbieranie = 0;
    } catch (blad) {
      pokazBlad(blad.message);
      el.statusText.textContent = 'Nie udało się uruchomić kamery.';
      pokazZatrzymany();
    } finally {
      stan.zajete = false;
      await odswiezStatus();
    }
  }

  async function stop(pokazKomunikat = true) {
    stan.zajete = true;
    el.btnStop.disabled = true;
    el.statusText.textContent = 'Zatrzymywanie…';
    try {
      await wyslij('/api/stop');
      if (pokazKomunikat) {
        el.statusText.textContent = 'Zatrzymano.';
      }
    } catch (blad) {
      pokazBlad(blad.message);
    } finally {
      stan.zajete = false;
      pokazZatrzymany();
      await odswiezStatus();
      if (czyWidokKamery() && !blokadaStartu()) {
        el.statusText.textContent = 'Gotowe.';
      }
    }
  }

  function blokadaStartu() {
    return stan.widok === 'rozpoznawanie' && !(stan.model && stan.model.gotowy);
  }

  // ---------------------------------------------------------------- model
  function renderModel() {
    const model = stan.model;
    if (!model) {
      el.modelBadge.textContent = '—';
      el.modelBadge.className = 'badge text-bg-secondary';
      return;
    }

    if (model.ladowanie) {
      el.modelBadge.textContent = 'Wczytywanie…';
      el.modelBadge.className = 'badge text-bg-warning';
      el.modelInfo.textContent = 'Wczytywanie sieci i wzorców. To może potrwać kilka sekund.';
    } else if (model.gotowy) {
      el.modelBadge.textContent = 'Gotowy';
      el.modelBadge.className = 'badge text-bg-success';
      el.modelInfo.textContent =
        `Wzorce dla ${model.liczba_osob} ` +
        `${model.liczba_osob === 1 ? 'osoby' : 'osób'} · urządzenie: ${model.urzadzenie || 'cpu'}.`;
    } else if (model.blad) {
      el.modelBadge.textContent = 'Brak modelu';
      el.modelBadge.className = 'badge text-bg-danger';
      el.modelInfo.textContent = model.blad;
    } else {
      el.modelBadge.textContent = 'Nieaktywny';
      el.modelBadge.className = 'badge text-bg-secondary';
      el.modelInfo.textContent = 'Model nie został jeszcze wczytany.';
    }

    const osoby = model.osoby || [];
    el.listaOsobModelu.innerHTML = osoby.length
      ? osoby
          .map((osoba) => `<span class="badge text-bg-light border">${escapeHtml(osoba)}</span>`)
          .join('')
      : '<span class="text-muted">Brak wytrenowanych osób — uruchom trening.</span>';

    const emocjeDostepne = Boolean(model.emocje_dostepne);
    el.przelacznikEmocji.disabled = !emocjeDostepne;
    if (!emocjeDostepne) {
      el.przelacznikEmocji.checked = false;
      el.modelInfo.textContent += ' Model emocji nie jest dostępny.';
    } else {
      el.przelacznikEmocji.checked = Boolean(model.emocje_wlaczone);
    }

    el.korektaProgu.value = String(model.korekta_progu || 0);
    el.korektaProguWartosc.textContent = formatuj(el.korektaProgu.value, 2);

    // Gdy model właśnie się wczytał (albo zmienił stan), odświeżamy
    // podpowiedź na zatrzymanym podglądzie - inaczej zostałby tekst
    // "Wczytywanie modelu..." mimo gotowego silnika.
    if (
      stan.widok === 'rozpoznawanie' &&
      !(stan.status && stan.status.dziala) &&
      !el.video.hasAttribute('src')
    ) {
      el.placeholderText.textContent = opisZatrzymania();
    }
  }

  async function odswiezModel() {
    try {
      stan.model = await pobierz('/api/model');
      renderModel();
      zastosujStatus(stan.status || {});
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  async function zapewnijModel() {
    await odswiezModel();
    if (stan.model && !stan.model.gotowy && !stan.model.ladowanie) {
      try {
        await wyslij('/api/model/przygotuj');
      } catch (blad) {
        pokazBlad(blad.message);
      }
      await odswiezModel();
    }
  }

  async function przelaczEmocje() {
    const wlaczone = el.przelacznikEmocji.checked;
    try {
      const dane = await wyslij('/api/model/emocje', { wlaczone });
      stan.model = dane.status;
      renderModel();
      pokazInfo(wlaczone ? 'Włączono wykrywanie emocji.' : 'Wyłączono wykrywanie emocji.');
    } catch (blad) {
      el.przelacznikEmocji.checked = !wlaczone;
      pokazBlad(blad.message);
    }
  }

  async function ustawKorekte() {
    el.korektaProguWartosc.textContent = formatuj(el.korektaProgu.value, 2);
    try {
      const dane = await wyslij('/api/model/korekta', {
        korekta: Number.parseFloat(el.korektaProgu.value),
      });
      stan.model = dane.status;
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  // ------------------------------------------------------------- trening
  function opisCzasu(sekundy) {
    const s = Math.max(0, Math.round(Number(sekundy) || 0));
    const minuty = Math.floor(s / 60);
    const reszta = s % 60;
    return `${minuty}:${String(reszta).padStart(2, '0')}`;
  }

  function renderTrening(dane) {
    const [opis, klasa] = OPIS_TRENINGU[dane.stan] || OPIS_TRENINGU.bezczynny;
    el.badgeTreningu.textContent = opis;
    el.badgeTreningu.className = `badge ${klasa}`;
    el.etapTreningu.textContent = dane.etap || 'Gotowy do uruchomienia';
    el.procentTreningu.textContent = `${Math.round(dane.postep || 0)}%`;
    el.pasekTreningu.style.width = `${Math.min(100, Math.max(0, dane.postep || 0))}%`;

    const trwa = dane.stan === 'dziala';
    el.btnTreningStart.disabled = trwa;
    el.btnTreningStop.disabled = !trwa;
    el.pasekTreningu.classList.toggle('progress-bar-animated', trwa);
    el.pasekTreningu.classList.toggle('progress-bar-striped', trwa);

    el.czasTreningu.textContent = dane.czas_s ? `Czas: ${opisCzasu(dane.czas_s)}` : '—';

    el.alertTreningu.classList.toggle('d-none', dane.stan !== 'gotowe');
    if (dane.stan === 'gotowe') {
      el.alertTreningu.textContent =
        'Trening zakończony. Wzorce zostały przeładowane — możesz przejść do Rozpoznawania.';
    }

    const bladWidoczny = dane.stan === 'blad' || dane.stan === 'przerwany';
    el.alertTreninguBlad.classList.toggle('d-none', !bladWidoczny);
    if (bladWidoczny) {
      el.alertTreninguBlad.textContent = dane.blad || 'Trening nie został ukończony.';
    }

    const log = (dane.log || []).join('\n');
    if (log && el.logTreningu.textContent !== log) {
      el.logTreningu.textContent = log;
      el.logTreningu.scrollTop = el.logTreningu.scrollHeight;
    }
  }

  async function odswiezTrening() {
    try {
      const dane = await pobierz('/api/trening/status');
      renderTrening(dane);
      if (dane.stan === 'gotowe') {
        await odswiezModel();
      }
      return dane;
    } catch (blad) {
      pokazBlad(blad.message);
      return null;
    }
  }

  async function startTreningu() {
    el.btnTreningStart.disabled = true;
    try {
      await wyslij('/api/trening/start');
      el.logTreningu.textContent = 'Uruchamianie…';
      pokazInfo('Trening uruchomiony.');
      await odswiezTrening();
    } catch (blad) {
      pokazBlad(blad.message);
      el.btnTreningStart.disabled = false;
    }
  }

  async function stopTreningu() {
    el.btnTreningStop.disabled = true;
    try {
      await wyslij('/api/trening/stop');
      await odswiezTrening();
      pokazInfo('Trening przerwany.');
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  // ----------------------------------------------------------- ewaluacja
  function renderEwaluacja(dane) {
    const [opis, klasa] = OPIS_EWALUACJI[dane.stan] || OPIS_EWALUACJI.bezczynny;
    el.badgeEwaluacji.textContent = opis;
    el.badgeEwaluacji.className = `badge ms-auto ${klasa}`;
    el.etapEwaluacji.textContent = dane.etap || 'Gotowa do uruchomienia';
    el.procentEwaluacji.textContent = `${Math.round(dane.postep || 0)}%`;
    el.pasekEwaluacji.style.width = `${ogranicz(dane.postep)}%`;

    const trwa = dane.stan === 'dziala';
    el.btnEwaluacjaStart.disabled = trwa;
    el.btnEwaluacjaStop.disabled = !trwa;
    el.pasekEwaluacji.classList.toggle('progress-bar-animated', trwa);
    el.pasekEwaluacji.classList.toggle('progress-bar-striped', trwa);

    el.czasEwaluacji.textContent = dane.czas_s ? `Czas: ${opisCzasu(dane.czas_s)}` : '—';

    el.alertEwaluacji.classList.toggle('d-none', dane.stan !== 'gotowe');
    if (dane.stan === 'gotowe') {
      el.alertEwaluacji.textContent =
        'Ewaluacja zakończona. Raport, wykresy i galeria pomyłek są gotowe.';
    }

    const bladWidoczny = dane.stan === 'blad' || dane.stan === 'przerwany';
    el.alertEwaluacjiBlad.classList.toggle('d-none', !bladWidoczny);
    if (bladWidoczny) {
      el.alertEwaluacjiBlad.textContent =
        dane.blad || 'Ewaluacja nie została ukończona.';
    }

    const log = (dane.log || []).join('\n');
    if (log && el.logEwaluacji.textContent !== log) {
      el.logEwaluacji.textContent = log;
      el.logEwaluacji.scrollTop = el.logEwaluacji.scrollHeight;
    }
  }

  async function odswiezEwaluacje() {
    try {
      const dane = await pobierz('/api/ewaluacja/status');
      renderEwaluacja(dane);
      if (dane.czas_raportu && dane.czas_raportu !== stan.czasRaportu) {
        stan.czasRaportu = dane.czas_raportu;
        await odswiezRaportEwaluacji();
      }
      return dane;
    } catch (blad) {
      pokazBlad(blad.message);
      return null;
    }
  }

  async function startEwaluacji() {
    el.btnEwaluacjaStart.disabled = true;
    try {
      await wyslij('/api/ewaluacja/start');
      el.logEwaluacji.textContent = 'Uruchamianie ewaluacji…';
      pokazInfo('Ewaluacja uruchomiona.');
      await odswiezEwaluacje();
    } catch (blad) {
      pokazBlad(blad.message);
      el.btnEwaluacjaStart.disabled = false;
    }
  }

  async function stopEwaluacji() {
    el.btnEwaluacjaStop.disabled = true;
    try {
      await wyslij('/api/ewaluacja/stop');
      await odswiezEwaluacje();
      pokazInfo('Ewaluacja przerwana.');
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  function renderRaportEwaluacji(dane) {
    if (!dane || !dane.dostepny) {
      el.dataRaportu.textContent = '—';
      el.podsumowanieEwaluacji.innerHTML =
        `<span class="text-muted">${escapeHtml(
          (dane && dane.komunikat) || 'Brak raportu — uruchom ewaluację.',
        )}</span>`;
      el.tabelaMetryk.innerHTML =
        '<tr><td colspan="5" class="text-muted small">Brak danych.</td></tr>';
      el.tabelaPorownania.innerHTML =
        '<tr><td colspan="6" class="text-muted small">Brak danych.</td></tr>';
      el.wykresyEwaluacji.innerHTML =
        '<div class="col-12 text-muted small">Wykresy pojawią się po ewaluacji.</div>';
      el.galeriaPomylek.innerHTML =
        '<div class="col-12 text-muted small">Brak zdjęć z błędami.</div>';
      el.badgePomylek.textContent = '0';
      return;
    }

    stan.raportEwaluacji = dane;
    const wersja = dane.mtime || Date.now();
    const prototypy = dane.prototypy || {};
    const metryki = prototypy.metryki || {};
    const informacje = dane.dane || {};

    el.dataRaportu.textContent = dane.wygenerowano
      ? dane.wygenerowano.replace('T', ' ')
      : '—';

    const karty = [
      ['Accuracy', formatujProcent(metryki.accuracy), `${metryki.poprawne ?? '—'}/${metryki.liczba_znanych ?? '—'}`],
      ['FAR', formatujProcent(metryki.far), `${metryki.zaakceptowani_obcy ?? '—'} obcych`],
      ['FRR', formatujProcent(metryki.frr), `${metryki.odrzucone_znane ?? '—'} odrzuconych`],
      ['EER', formatujProcent(prototypy.eer), prototypy.prog_eer != null ? `próg ${formatuj(prototypy.prog_eer, 2)}` : '—'],
      ['AUC', prototypy.auc != null ? formatuj(prototypy.auc, 3) : '—', '1,0 = idealnie'],
    ];

    el.podsumowanieEwaluacji.innerHTML =
      '<div class="row g-2">' +
      karty
        .map(
          ([nazwa, wartosc, dopisek]) =>
            '<div class="col-6"><div class="border rounded p-2 h-100">' +
            `<div class="text-muted text-uppercase" style="font-size:.65rem;">${nazwa}</div>` +
            `<div class="h5 mb-0 text-mono">${wartosc}</div>` +
            `<div class="text-muted" style="font-size:.7rem;">${dopisek}</div>` +
            '</div></div>',
        )
        .join('') +
      '</div>' +
      `<p class="small text-muted mt-2 mb-0">Osoby: ${escapeHtml((informacje.osoby || []).join(', ') || '—')}` +
      ` · trening: ${informacje.trening ? informacje.trening.zdjecia : '—'} zdjęć` +
      ` · obcy: ${informacje.obcy ? informacje.obcy.zdjecia : '—'} zdjęć</p>`;

    const naOsobe = Object.entries(prototypy.na_osobe || {});
    el.tabelaMetryk.innerHTML = naOsobe.length
      ? naOsobe
          .map(
            ([osoba, metryka]) =>
              '<tr>' +
              `<td class="fw-semibold">${escapeHtml(osoba)}</td>` +
              `<td class="text-end text-mono">${metryka.zdjecia}</td>` +
              `<td class="text-end text-mono">${formatujProcent(metryka.precision)}</td>` +
              `<td class="text-end text-mono">${formatujProcent(metryka.recall)}</td>` +
              `<td class="text-end text-mono">${formatujProcent(metryka.f1)}</td>` +
              '</tr>',
          )
          .join('')
      : '<tr><td colspan="5" class="text-muted small">Brak danych.</td></tr>';

    const porownanie = dane.porownanie || [];
    el.tabelaPorownania.innerHTML = porownanie.length
      ? porownanie
          .map(
            (wpis) =>
              '<tr>' +
              `<td class="fw-semibold">${escapeHtml(wpis.metoda)}</td>` +
              `<td class="text-end text-mono">${formatujProcent(wpis.accuracy_zamknieta)}</td>` +
              `<td class="text-end text-mono">${formatujProcent(wpis.far)}</td>` +
              `<td class="text-end text-mono">${formatujProcent(wpis.frr)}</td>` +
              `<td class="text-end text-mono">${formatujProcent(wpis.eer)}</td>` +
              `<td class="text-end text-mono">${wpis.auc != null ? formatuj(wpis.auc, 3) : '—'}</td>` +
              '</tr>',
          )
          .join('')
      : '<tr><td colspan="6" class="text-muted small">Brak danych.</td></tr>';

    const wykresy = Object.entries(dane.wykresy || {});
    el.wykresyEwaluacji.innerHTML = wykresy.length
      ? wykresy
          .map(([klucz, plik]) => {
            const opis = OPIS_WYKRESOW[klucz] || klucz;
            const adres = `/api/ewaluacja/wykres/${encodeURIComponent(plik)}?v=${wersja}`;
            return (
              '<div class="col-xl-6"><div class="card h-100 border">' +
              '<div class="card-body p-2">' +
              `<img class="wykres-ewaluacji" src="${adres}" alt="${escapeHtml(opis)}" loading="lazy">` +
              `<div class="small text-muted mt-2 text-center">${escapeHtml(opis)}</div>` +
              '</div></div></div>'
            );
          })
          .join('')
      : '<div class="col-12 text-muted small">Brak wykresów.</div>';

    const pomylki = dane.pomylki || [];
    el.badgePomylek.textContent = String(pomylki.length);
    el.galeriaPomylek.innerHTML = pomylki.length
      ? pomylki
          .map((wpis) => {
            if (!wpis.plik) {
              return '';
            }
            const opis = `${wpis.prawdziwa} → ${wpis.przewidziana}`;
            const typKlasy =
              wpis.typ === 'obcy zaakceptowany' ? 'text-bg-danger' : 'text-bg-warning';
            const adres = `/api/ewaluacja/pomylka/${encodeURIComponent(wpis.plik)}?v=${wersja}`;
            return (
              '<div class="col-6 col-md-4 col-xxl-3"><div class="card h-100 border">' +
              `<img class="pomylka-thumb" src="${adres}" alt="${escapeHtml(opis)}" loading="lazy">` +
              '<div class="card-body p-2 small">' +
              `<div class="fw-semibold text-truncate" title="${escapeHtml(opis)}">` +
              `${escapeHtml(wpis.prawdziwa)} → ${escapeHtml(wpis.przewidziana)}</div>` +
              '<div class="d-flex justify-content-between align-items-center mt-1 gap-1">' +
              `<span class="badge ${typKlasy}">${escapeHtml(wpis.typ)}</span>` +
              `<span class="text-mono text-muted">${formatujProcent(wpis.podobienstwo)}</span>` +
              '</div></div></div></div>'
            );
          })
          .join('')
      : '<div class="col-12 text-muted small">Model nie popełnił żadnych błędów (albo raport ich nie zawiera).</div>';

    const ostrzezenia = informacje.ostrzezenia || [];
    el.ostrzezeniaEwaluacji.innerHTML = ostrzezenia.length
      ? ostrzezenia
          .map(
            (tekst, indeks) =>
              `<p class="${indeks === ostrzezenia.length - 1 ? 'mb-0' : 'mb-1'}">${escapeHtml(tekst)}</p>`,
          )
          .join('')
      : 'Ewaluacja korzysta ze zbioru walidacyjnego — wyniki są optymistyczne.';
  }

  async function odswiezRaportEwaluacji() {
    try {
      const dane = await pobierz('/api/ewaluacja/raport');
      renderRaportEwaluacji(dane);
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  // ----------------------------------------------------------------- baza
  function renderBaza(dane) {
    const osoby = dane.osoby || [];
    el.badgeBazy.textContent = String(osoby.length);

    if (!osoby.length) {
      el.listaOsob.innerHTML =
        '<div class="col-12 text-center text-muted py-4">' +
        '<p class="mb-2">Baza jest pusta.</p>' +
        '<button class="btn btn-sm btn-primary" type="button" data-przejdz="dodawanie">Dodaj pierwszą twarz</button>' +
        '</div>';
    } else {
      el.listaOsob.innerHTML = osoby
        .map((osoba) => {
          const zdjecie = osoba.przyklady && osoba.przyklady[0];
          const miniatura = zdjecie
            ? `<img class="person-thumb" src="${zdjecie}" alt="Zdjęcie: ${escapeHtml(osoba.nazwa)}" loading="lazy">`
            : '<div class="person-thumb person-thumb-brak">brak</div>';
          return (
            '<div class="col-md-6 col-xxl-4">' +
            '<div class="card app-card h-100"><div class="row g-0">' +
            `<div class="col-4">${miniatura}</div>` +
            '<div class="col-8"><div class="card-body p-3 d-flex flex-column h-100">' +
            `<div class="fw-semibold text-truncate" title="${escapeHtml(osoba.nazwa)}">${escapeHtml(osoba.nazwa)}</div>` +
            `<div class="small text-muted mb-2">${osoba.zdjecia} ${odmianaZdjęć(osoba.zdjecia)}</div>` +
            '<button class="btn btn-sm btn-outline-danger mt-auto align-self-start" type="button" ' +
            `data-usun="${escapeHtml(osoba.nazwa)}">Usuń</button>` +
            '</div></div></div></div></div>'
          );
        })
        .join('');
    }

    const model = dane.model || {};
    const silnik = dane.silnik || {};
    const osobyModelu = model.osoby || [];
    if (model.blad) {
      el.infoModeluBazy.innerHTML = `<span class="text-danger">${escapeHtml(model.blad)}</span>`;
    } else if (!model.istnieje) {
      el.infoModeluBazy.innerHTML =
        'Brak wytrenowanego modelu. Dodaj twarze i uruchom <strong>Trening</strong>.';
    } else {
      el.infoModeluBazy.innerHTML =
        `<p class="mb-1"><strong>Osoby w modelu:</strong> ${
          osobyModelu.length ? osobyModelu.map(escapeHtml).join(', ') : '—'
        }</p>` +
        `<p class="mb-1"><strong>Wymiar cech:</strong> ${model.wymiar ?? '—'}</p>` +
        `<p class="mb-0"><strong>Silnik:</strong> ${
          silnik.gotowy ? `gotowy (${escapeHtml(silnik.urzadzenie || 'cpu')})` : 'nieaktywny'
        }</p>`;
    }
  }

  async function odswiezBaze() {
    try {
      const dane = await pobierz('/api/baza');
      renderBaza(dane);
      renderInfoModelu(dane.model || {}, dane.silnik || {});
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  // Informacje o modelu są potrzebne w dwóch widokach (trening i baza).
  function renderInfoModelu(model, silnik) {
    const osoby = model.osoby || [];
    if (model.blad) {
      el.modelWTreningu.innerHTML = `<span class="text-danger">${escapeHtml(model.blad)}</span>`;
      return;
    }
    el.modelWTreningu.innerHTML = model.istnieje
      ? `<p class="mb-1"><strong>Osób:</strong> ${osoby.length}</p>` +
        `<p class="mb-1"><strong>Osoby:</strong> ${
          osoby.length ? osoby.map(escapeHtml).join(', ') : '—'
        }</p>` +
        `<p class="mb-1"><strong>Wymiar cech:</strong> ${model.wymiar ?? '—'}</p>` +
        `<p class="mb-0"><strong>Silnik:</strong> ${
          silnik.gotowy ? 'wczytany' : 'nieaktywny'
        }</p>`
      : '<span class="text-muted">Brak wytrenowanego modelu.</span>';
  }

  async function odswiezInfoModelu() {
    try {
      const dane = await pobierz('/api/baza');
      renderInfoModelu(dane.model || {}, dane.silnik || {});
    } catch (_) {
      /* widok treningu pokaże własny komunikat błędu */
    }
  }

  function otworzModalUsun(osoba) {
    stan.osobaDoUsuniecia = osoba;
    el.usunOsobaNazwa.textContent = osoba;
    el.usunPrzetrenuj.checked = true;
    modalUsun.show();
  }

  async function potwierdzUsuniecie() {
    const osoba = stan.osobaDoUsuniecia;
    if (!osoba) {
      return;
    }
    el.btnPotwierdzUsun.disabled = true;
    try {
      const dane = await wyslij('/api/baza/usun', {
        osoba,
        potwierdzenie: osoba,
        przetrenuj: el.usunPrzetrenuj.checked,
      });
      modalUsun.hide();
      pokazInfo(
        dane.przetrenowano
          ? `Usunięto „${osoba}”. Uruchomiono ponowny trening.`
          : `Usunięto „${osoba}”.`,
      );
      await odswiezBaze();
      if (dane.przetrenowano) {
        await odswiezTrening();
      }
    } catch (blad) {
      pokazBlad(blad.message);
    } finally {
      el.btnPotwierdzUsun.disabled = false;
      stan.osobaDoUsuniecia = null;
    }
  }

  // -------------------------------------------------------------- sidebar
  function przelaczSidebar() {
    if (window.innerWidth < 992) {
      document.body.classList.toggle('sidebar-otwarty');
    } else {
      document.body.classList.toggle('sidebar-zwiniety');
    }
  }

  function synchronizujSuwaki() {
    el.progWartosc.textContent = formatuj(el.prog.value, 2);
    el.proporcjaWartosc.textContent = formatuj(el.proporcja.value, 2);
    el.korektaProguWartosc.textContent = formatuj(el.korektaProgu.value, 2);
  }

  function aktualizujWskazowkeZbierania() {
    if (stan.status && stan.status.dziala && stan.status.tryb === 'zbieranie') {
      return;
    }
    const cel = Number.parseInt(el.celZdjec.value, 10) || 0;
    el.licznikZdjec.textContent = `0 / ${cel || 'bez limitu'}`;
    el.pasekZbierania.style.width = '0%';
  }

  // ----------------------------------------------------------------- init
  function init() {
    el.btnStart.addEventListener('click', start);
    el.btnStop.addEventListener('click', () => stop());

    el.prog.addEventListener('input', synchronizujSuwaki);
    el.proporcja.addEventListener('input', synchronizujSuwaki);
    el.korektaProgu.addEventListener('change', ustawKorekte);
    el.przelacznikEmocji.addEventListener('change', przelaczEmocje);

    el.btnOdswiezKamery.addEventListener('click', odswiezKamery);
    el.btnOdswiezKameryModal.addEventListener('click', odswiezKamery);

    el.btnTreningStart.addEventListener('click', startTreningu);
    el.btnTreningStop.addEventListener('click', stopTreningu);
    el.btnEwaluacjaStart.addEventListener('click', startEwaluacji);
    el.btnEwaluacjaStop.addEventListener('click', stopEwaluacji);
    el.btnOdswiezBaze.addEventListener('click', odswiezBaze);
    el.btnPotwierdzUsun.addEventListener('click', potwierdzUsuniecie);

    document.querySelectorAll('.nav-link[data-view]').forEach((przycisk) => {
      przycisk.addEventListener('click', () => {
        przelaczWidok(przycisk.dataset.view);
        document.body.classList.remove('sidebar-otwarty');
      });
    });

    // Delegacja zdarzeń dla dynamicznie generowanej listy osób.
    el.listaOsob.addEventListener('click', (wydarzenie) => {
      const przyciskUsun = wydarzenie.target.closest('[data-usun]');
      if (przyciskUsun) {
        otworzModalUsun(przyciskUsun.dataset.usun);
        return;
      }
      const przyciskPrzejdz = wydarzenie.target.closest('[data-przejdz]');
      if (przyciskPrzejdz) {
        przelaczWidok(przyciskPrzejdz.dataset.przejdz);
      }
    });

    el.osobaNazwa.addEventListener('input', () => {
      const poprawna = nazwaOsobyPoprawna(el.osobaNazwa.value.trim());
      el.osobaNazwa.classList.toggle('is-invalid', Boolean(el.osobaNazwa.value) && !poprawna);
    });
    el.celZdjec.addEventListener('input', aktualizujWskazowkeZbierania);

    el.btnSidebar.addEventListener('click', przelaczSidebar);
    el.sidebarBackdrop.addEventListener('click', () => {
      document.body.classList.remove('sidebar-otwarty');
    });

    document.addEventListener('keydown', (wydarzenie) => {
      if (wydarzenie.key === 'Escape') {
        document.body.classList.remove('sidebar-otwarty');
      }
    });

    window.addEventListener('resize', () => {
      if (window.innerWidth >= 992) {
        document.body.classList.remove('sidebar-otwarty');
      }
    });

    el.modalKamery.addEventListener('show.bs.modal', () => {
      pobierz('/api/kamery').then(renderujListeKamer).catch(() => {});
    });

    synchronizujSuwaki();
    pokazZatrzymany();
    zaladujKamery();
    odswiezStatus();

    // Status kamery: co sekundę (tanie zapytanie).
    setInterval(odswiezStatus, 1000);

    // Model: dociągamy, gdy jesteśmy w Rozpoznawaniu i jeszcze się ładuje.
    setInterval(() => {
      if (
        stan.widok === 'rozpoznawanie' &&
        (!stan.model || stan.model.ladowanie)
      ) {
        odswiezModel();
      }
    }, 2000);

    // Trening: odświeżamy tylko na jego widoku (log bywa długi).
    setInterval(() => {
      if (stan.widok === 'trening') {
        odswiezTrening();
      }
    }, 1000);

    // Ewaluacja: log i wykresy pobieramy tylko, gdy użytkownik patrzy na widok.
    setInterval(() => {
      if (stan.widok === 'ewaluacja') {
        odswiezEwaluacje();
      }
    }, 1000);
  }

  init();
})();
