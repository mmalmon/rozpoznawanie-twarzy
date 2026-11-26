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
    xaiBadge: document.getElementById('xaiBadge'),
    xaiTwarz: document.getElementById('xaiTwarz'),
    btnWyjasnij: document.getElementById('btnWyjasnij'),
    xaiWynik: document.getElementById('xaiWynik'),

    panelDodawanie: document.getElementById('panelDodawanie'),
    osobaNazwa: document.getElementById('osobaNazwa'),
    celZdjec: document.getElementById('celZdjec'),
    testTrudnyZbieranie: document.getElementById('testTrudnyZbieranie'),
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

    viewEksperymenty: document.getElementById('viewEksperymenty'),
    badgeEksperymentow: document.getElementById('badgeEksperymentow'),
    etapEksperymentow: document.getElementById('etapEksperymentow'),
    procentEksperymentow: document.getElementById('procentEksperymentow'),
    pasekEksperymentow: document.getElementById('pasekEksperymentow'),
    btnEksperymentyStart: document.getElementById('btnEksperymentyStart'),
    btnEksperymentyStop: document.getElementById('btnEksperymentyStop'),
    czasEksperymentow: document.getElementById('czasEksperymentow'),
    alertEksperymentow: document.getElementById('alertEksperymentow'),
    alertEksperymentowBlad: document.getElementById('alertEksperymentowBlad'),
    logEksperymentow: document.getElementById('logEksperymentow'),
    podsumowanieEksperymentow: document.getElementById('podsumowanieEksperymentow'),
    dataRaportuEksp: document.getElementById('dataRaportuEksp'),
    wnioskiEksperymentow: document.getElementById('wnioskiEksperymentow'),
    tabelaEksperymentow: document.getElementById('tabelaEksperymentow'),
    wykresyEksperymentow: document.getElementById('wykresyEksperymentow'),
    ostrzezeniaEksperymentow: document.getElementById('ostrzezeniaEksperymentow'),

    viewTestTrudny: document.getElementById('viewTestTrudny'),
    badgeTestTrudny: document.getElementById('badgeTestTrudny'),
    etapTestTrudny: document.getElementById('etapTestTrudny'),
    procentTestTrudny: document.getElementById('procentTestTrudny'),
    pasekTestTrudny: document.getElementById('pasekTestTrudny'),
    btnTestTrudnyStart: document.getElementById('btnTestTrudnyStart'),
    btnTestTrudnyStop: document.getElementById('btnTestTrudnyStop'),
    czasTestTrudny: document.getElementById('czasTestTrudny'),
    alertTestTrudny: document.getElementById('alertTestTrudny'),
    alertTestTrudnyBlad: document.getElementById('alertTestTrudnyBlad'),
    logTestTrudny: document.getElementById('logTestTrudny'),
    podsumowanieTestTrudny: document.getElementById('podsumowanieTestTrudny'),
    dataRaportuTestTrudny: document.getElementById('dataRaportuTestTrudny'),
    tabelaPorownaniaTrudny: document.getElementById('tabelaPorownaniaTrudny'),
    tabelaMetrykTrudny: document.getElementById('tabelaMetrykTrudny'),
    wykresyTestTrudny: document.getElementById('wykresyTestTrudny'),
    ostrzezeniaTestTrudny: document.getElementById('ostrzezeniaTestTrudny'),

    viewBaza: document.getElementById('viewBaza'),
    badgeBazy: document.getElementById('badgeBazy'),
    btnOdswiezBaze: document.getElementById('btnOdswiezBaze'),
    listaOsob: document.getElementById('listaOsob'),
    infoModeluBazy: document.getElementById('infoModeluBazy'),

    viewFrekwencja: document.getElementById('viewFrekwencja'),
    badgeFrekwencji: document.getElementById('badgeFrekwencji'),
    infoRejestru: document.getElementById('infoRejestru'),
    dataFrekwencji: document.getElementById('dataFrekwencji'),
    btnOdswiezFrekwencje: document.getElementById('btnOdswiezFrekwencje'),
    btnEksportFrekwencji: document.getElementById('btnEksportFrekwencji'),
    przelacznikRejestru: document.getElementById('przelacznikRejestru'),
    ostrzezenieRejestru: document.getElementById('ostrzezenieRejestru'),
    statObecni: document.getElementById('statObecni'),
    statNieobecni: document.getElementById('statNieobecni'),
    statRozpoznan: document.getElementById('statRozpoznan'),
    statWpisow: document.getElementById('statWpisow'),
    tabelaObecnych: document.getElementById('tabelaObecnych'),
    listaNieobecnych: document.getElementById('listaNieobecnych'),
    osobaDopisz: document.getElementById('osobaDopisz'),
    btnDopiszObecnosc: document.getElementById('btnDopiszObecnosc'),
    tabelaOstatnich: document.getElementById('tabelaOstatnich'),
    poleUsunDzien: document.getElementById('poleUsunDzien'),
    btnUsunDzien: document.getElementById('btnUsunDzien'),
    poleWyczysc: document.getElementById('poleWyczysc'),
    btnWyczyscRejestr: document.getElementById('btnWyczyscRejestr'),
    retencjaInfo: document.getElementById('retencjaInfo'),

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
    eksperymenty: {
      tytul: 'Eksperymenty',
      podtytul: 'Wpływ liczby zdjęć, jasności, k w k-NN i progu decyzyjnego',
      kamera: false,
    },
    'test-trudny': {
      tytul: 'Test trudny',
      podtytul: 'Ten sam model na zdjęciach z innych warunków — czy wynik się utrzyma?',
      kamera: false,
    },
    baza: {
      tytul: 'Baza twarzy',
      podtytul: 'Podgląd i usuwanie zapisanych osób',
      kamera: false,
    },
    frekwencja: {
      tytul: 'Frekwencja',
      podtytul: 'Rejestr rozpoznań, obecność dnia i eksport CSV',
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

  const OPIS_EKSPERYMENTOW = {
    bezczynny: ['Bezczynne', 'text-bg-secondary'],
    dziala: ['Trwają eksperymenty', 'text-bg-warning'],
    gotowe: ['Gotowe', 'text-bg-success'],
    blad: ['Błąd', 'text-bg-danger'],
    przerwany: ['Przerwano', 'text-bg-secondary'],
  };

  const OPIS_TESTU_TRUDNEGO = {
    bezczynny: ['Bezczynny', 'text-bg-secondary'],
    dziala: ['Trwa test trudny', 'text-bg-warning'],
    gotowe: ['Gotowe', 'text-bg-success'],
    blad: ['Błąd', 'text-bg-danger'],
    przerwany: ['Przerwano', 'text-bg-secondary'],
  };

  const OPIS_WYKRESOW_EKSPERYMENTOW = {
    wplyw_liczby_zdjec: 'Wpływ liczby zdjęć treningowych',
    wplyw_jasnosci: 'Wpływ augmentacji jasności',
    wplyw_k: 'Wpływ liczby sąsiadów k w k-NN',
    prog_decyzyjny: 'Kompromis FAR / FRR przy różnych progach',
  };

  const OPIS_WYKRESOW = {
    macierz_pomylek: 'Macierz pomyłek',
    rozklad_podobienstw: 'Rozkład podobieństw: swoi vs obcy',
    krzywa_progu: 'Kompromis FAR / FRR',
    porownanie_metod: 'Porównanie metod',
    pca_embeddingi: 'Embeddingi w 2D (PCA)',
    przedzialy_ufnosci: 'Przedziały ufności (95%)',
  };

  const OPIS_WYKRESOW_TESTU_TRUDNEGO = {
    podsumowanie: 'Wynik testu trudnego z 95% przedziałem ufności',
    porownanie: 'Łatwy test vs trudny test',
    rozklad_podobienstw: 'Rozkład podobieństw: swoi vs obcy',
    macierz_pomylek: 'Macierz pomyłek',
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
    raportEksperymentow: null,
    czasRaportuEksp: null,
    raportTestuTrudnego: null,
    czasRaportuTestTrudny: null,
    ostatnieWyniki: '',
    rejestr: null,
    xaiLiczy: false,
    xaiSygnatura: '',
  };

  // ------------------------------------------------------------- pomocnicze
  const formatuj = (wartosc, miejsca = 1) =>
    Number(wartosc || 0).toFixed(miejsca).replace('.', ',');

  const formatujProcent = (wartosc, miejsca = 1) =>
    wartosc === null || wartosc === undefined
      ? '—'
      : `${formatuj(wartosc * 100, miejsca)}%`;

  // Przedział ufności w krótkiej formie, np. "CI 97,1–100,0%".
  const formatujCI = (przedzial, procent = true) => {
    if (!przedzial || przedzial.length !== 2) {
      return 'brak CI';
    }
    const [dolny, gorny] = przedzial;
    return procent
      ? `CI ${formatuj(dolny * 100)}–${formatuj(gorny * 100)}%`
      : `CI ${formatuj(dolny, 3)}–${formatuj(gorny, 3)}`;
  };

  // Przedział ufności z raportu: obiekty mają go w różnych miejscach.
  const ciZRaportu = (zrodlo, klucz) => {
    if (!zrodlo) {
      return null;
    }
    const przedzialy = zrodlo.przedzialy_ufnosci || {};
    return (przedzialy.metryki || {})[klucz] || null;
  };

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
    el.viewEksperymenty.classList.toggle('d-none', nazwa !== 'eksperymenty');
    el.viewTestTrudny.classList.toggle('d-none', nazwa !== 'test-trudny');
    el.viewBaza.classList.toggle('d-none', nazwa !== 'baza');
    el.viewFrekwencja.classList.toggle('d-none', nazwa !== 'frekwencja');
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
    } else if (nazwa === 'eksperymenty') {
      await odswiezEksperymenty();
      await odswiezRaportEksperymentow();
    } else if (nazwa === 'test-trudny') {
      await odswiezTestTrudny();
      await odswiezRaportTestuTrudnego();
    } else if (nazwa === 'baza') {
      await odswiezBaze();
    } else if (nazwa === 'frekwencja') {
      await odswiezFrekwencje();
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
    el.testTrudnyZbieranie.disabled = dziala && dane.tryb === 'zbieranie';

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

    aktualizujWyborXai(wyniki, aktywne);

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

  // ------------------------------------------------------------------ XAI
  function aktualizujWyborXai(wyniki, aktywne) {
    const opcje = aktywne && Array.isArray(wyniki) ? wyniki : [];
    // Sygnatura zmienia się tylko wtedy, gdy zmienił się skład twarzy.
    // Dzięki temu lista nie jest przebudowywana co sekundę i nie gubi wyboru.
    const sygnatura = JSON.stringify(
      opcje.map((wynik) => wynik.osoba || NIEZNANA_OSOBA),
    );
    if (sygnatura === stan.xaiSygnatura) {
      return;
    }
    stan.xaiSygnatura = sygnatura;

    const poprzedni = el.xaiTwarz.value;
    el.xaiTwarz.innerHTML = opcje.length
      ? opcje
          .map((wynik, indeks) => {
            const osoba = wynik.osoba || NIEZNANA_OSOBA;
            const opis = `Twarz ${indeks + 1}: ${osoba}`;
            return `<option value="${indeks}">${escapeHtml(opis)}</option>`;
          })
          .join('')
      : '<option value="">Brak wyników</option>';

    if (poprzedni !== '' && Number(poprzedni) < opcje.length) {
      el.xaiTwarz.value = poprzedni;
    }

    const dostepne = opcje.length > 0;
    el.xaiTwarz.disabled = !dostepne || stan.xaiLiczy;
    el.btnWyjasnij.disabled = !dostepne || stan.xaiLiczy;
  }

  async function wyjasnijDecyzje() {
    if (stan.xaiLiczy) {
      return;
    }
    const indeks = Number(el.xaiTwarz.value);
    if (!Number.isInteger(indeks) || indeks < 0) {
      pokazBlad('Najpierw wybierz twarz do wyjaśnienia.');
      return;
    }

    stan.xaiLiczy = true;
    el.xaiTwarz.disabled = true;
    el.btnWyjasnij.disabled = true;
    el.btnWyjasnij.innerHTML =
      '<span class="spinner-border spinner-border-sm me-1" aria-hidden="true"></span>Liczę…';
    el.xaiBadge.textContent = 'liczę…';
    el.xaiBadge.className = 'badge text-bg-warning ms-auto';

    try {
      const dane = await wyslij('/api/xai/wyjasnij', { indeks });
      renderXai(dane);
    } catch (blad) {
      pokazBlad(blad.message);
      el.xaiBadge.textContent = 'błąd';
      el.xaiBadge.className = 'badge text-bg-danger ms-auto';
      el.xaiWynik.innerHTML =
        `<div class="alert alert-danger mb-0 py-2">${escapeHtml(blad.message)}</div>`;
    } finally {
      stan.xaiLiczy = false;
      el.btnWyjasnij.textContent = 'Wyjaśnij decyzję';
      const dostepne = el.xaiTwarz.value !== '';
      el.xaiTwarz.disabled = !dostepne;
      el.btnWyjasnij.disabled = !dostepne;
    }
  }

  function renderXai(dane) {
    const rozpoznana = Boolean(dane.rozpoznana);
    el.xaiBadge.textContent = rozpoznana ? 'rozpoznana' : 'poniżej progu';
    el.xaiBadge.className = `badge ms-auto ${
      rozpoznana ? 'text-bg-success' : 'text-bg-warning'
    }`;

    const kandydaci = Array.isArray(dane.kandydaci) ? dane.kandydaci.slice(0, 3) : [];
    const ranking = kandydaci.length
      ? kandydaci
          .map(
            (kandydat) =>
              `<span class="badge ${
                kandydat.akceptacja ? 'text-bg-success' : 'text-bg-light border'
              }">${escapeHtml(kandydat.osoba)} ${formatujProcent(kandydat.podobienstwo)}</span>`,
          )
          .join(' ')
      : '<span class="text-muted">brak rankingu</span>';

    const wzorce = Array.isArray(dane.wzorce) ? dane.wzorce : [];
    const miniatury = wzorce.length
      ? `<div class="xai-wzorce">${wzorce
          .map((wzorzec) => {
            const obraz = wzorzec.url
              ? `<img class="xai-wzorzec" src="${wzorzec.url}" alt="Wzorzec: ${escapeHtml(
                  wzorzec.osoba,
                )}" loading="lazy">`
              : '<div class="xai-wzorzec-brak">bez zdjęcia</div>';
            const jasnosc = Number(wzorzec.jasnosc);
            return (
              '<div class="xai-wzorzec-karta">' +
              obraz +
              `<div class="xai-wzorzec-opis">${escapeHtml(wzorzec.osoba)}</div>` +
              `<div class="xai-wzorzec-meta">${formatujProcent(wzorzec.podobienstwo)}` +
              (Number.isFinite(jasnosc) ? ` · jasność ${formatuj(jasnosc, 2)}` : '') +
              '</div></div>'
            );
          })
          .join('')}</div>`
      : '<p class="text-muted mb-0">Ten model nie ma zapisanych źródeł wzorców. Uruchom ponownie trening (krok 03), aby zobaczyć galerię zdjęć treningowych.</p>';

    const mapa = dane.obraz_url
      ? `<img class="xai-mapa" src="${dane.obraz_url}" alt="Mapa okluzji: oryginał i obszary ważne dla decyzji">`
      : '<p class="text-muted mb-0">Nie udało się zapisać wizualizacji mapy.</p>';

    const prog =
      dane.prog === null || dane.prog === undefined
        ? ''
        : ` / próg ${formatujProcent(dane.prog)}`;

    el.xaiWynik.innerHTML =
      '<div class="d-flex flex-wrap align-items-center gap-2 mb-2">' +
      `<span class="fw-semibold">${escapeHtml(dane.osoba)}</span>` +
      `<span class="text-mono">${formatujProcent(dane.podobienstwo)}${prog}</span>` +
      `<span class="text-muted ms-auto">${dane.liczba_wzorcow} prototypów · ${dane.czas_ms} ms</span>` +
      '</div>' +
      `<div class="mb-3">${ranking}</div>` +
      `<p class="mb-2">${escapeHtml(dane.opis)}. Największy spadek podobieństwa: ` +
      `<strong>${formatujProcent(dane.maks_spadek)}</strong>.</p>` +
      '<h6 class="mt-3">Najbliższe wzorce treningowe</h6>' +
      miniatury +
      '<h6 class="mt-3">Mapa okluzji</h6>' +
      mapa +
      '<p class="text-muted mt-2 mb-0">Wyjaśnienie dotyczy jednej, zapisanej ' +
      'klatki — nie aktualizuje się przy każdej zmianie kadru.</p>';
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
      daneStartu.test_trudny = el.testTrudnyZbieranie.checked;
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
        '<tr><td colspan="7" class="text-muted small">Brak danych.</td></tr>';
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
    const ci = (prototypy.przedzialy_ufnosci || {}).metryki || {};
    const informacje = dane.dane || {};

    el.dataRaportu.textContent = dane.wygenerowano
      ? dane.wygenerowano.replace('T', ' ')
      : '—';

    const karty = [
      ['Accuracy', formatujProcent(metryki.accuracy), `${metryki.poprawne ?? '—'}/${metryki.liczba_znanych ?? '—'} · ${formatujCI(ci.accuracy)}`],
      ['FAR', formatujProcent(metryki.far), `${metryki.zaakceptowani_obcy ?? '—'} obcych · ${formatujCI(ci.far)}`],
      ['FRR', formatujProcent(metryki.frr), `${metryki.odrzucone_znane ?? '—'} odrzuconych · ${formatujCI(ci.frr)}`],
      ['EER', formatujProcent(prototypy.eer), `${prototypy.prog_eer != null ? `próg ${formatuj(prototypy.prog_eer, 2)}` : '—'} · ${formatujCI(ci.eer)}`],
      ['AUC', prototypy.auc != null ? formatuj(prototypy.auc, 3) : '—', formatujCI(ci.auc, false)],
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
      ` · test: ${informacje.test ? informacje.test.zdjecia_znanych : '—'} zdjęć` +
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
              `<td class="text-end text-mono">${formatujCI(ciZRaportu(wpis, 'accuracy'))}</td>` +
              '</tr>',
          )
          .join('')
      : '<tr><td colspan="7" class="text-muted small">Brak danych.</td></tr>';

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
      : 'Ewaluacja raportuje na zbiorze testowym — prog kalibrowano na walidacji.';
  }

  async function odswiezRaportEwaluacji() {
    try {
      const dane = await pobierz('/api/ewaluacja/raport');
      renderRaportEwaluacji(dane);
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  // -------------------------------------------------------- eksperymenty
  function renderEksperymenty(dane) {
    const [opis, klasa] = OPIS_EKSPERYMENTOW[dane.stan] || OPIS_EKSPERYMENTOW.bezczynny;
    el.badgeEksperymentow.textContent = opis;
    el.badgeEksperymentow.className = `badge ms-auto ${klasa}`;
    el.etapEksperymentow.textContent = dane.etap || 'Gotowe do uruchomienia';
    el.procentEksperymentow.textContent = `${Math.round(dane.postep || 0)}%`;
    el.pasekEksperymentow.style.width = `${ogranicz(dane.postep)}%`;

    const trwa = dane.stan === 'dziala';
    el.btnEksperymentyStart.disabled = trwa;
    el.btnEksperymentyStop.disabled = !trwa;
    el.pasekEksperymentow.classList.toggle('progress-bar-animated', trwa);
    el.pasekEksperymentow.classList.toggle('progress-bar-striped', trwa);

    el.czasEksperymentow.textContent = dane.czas_s
      ? `Czas: ${opisCzasu(dane.czas_s)}`
      : '—';

    el.alertEksperymentow.classList.toggle('d-none', dane.stan !== 'gotowe');
    if (dane.stan === 'gotowe') {
      el.alertEksperymentow.textContent =
        'Eksperymenty zakończone. Raport, CSV i wykresy są gotowe.';
    }

    const bladWidoczny = dane.stan === 'blad' || dane.stan === 'przerwany';
    el.alertEksperymentowBlad.classList.toggle('d-none', !bladWidoczny);
    if (bladWidoczny) {
      el.alertEksperymentowBlad.textContent =
        dane.blad || 'Eksperymenty nie zostały ukończone.';
    }

    const log = (dane.log || []).join('\n');
    if (log && el.logEksperymentow.textContent !== log) {
      el.logEksperymentow.textContent = log;
      el.logEksperymentow.scrollTop = el.logEksperymentow.scrollHeight;
    }
  }

  async function odswiezEksperymenty() {
    try {
      const dane = await pobierz('/api/eksperymenty/status');
      renderEksperymenty(dane);
      if (dane.czas_raportu && dane.czas_raportu !== stan.czasRaportuEksp) {
        stan.czasRaportuEksp = dane.czas_raportu;
        await odswiezRaportEksperymentow();
      }
      return dane;
    } catch (blad) {
      pokazBlad(blad.message);
      return null;
    }
  }

  async function startEksperymentow() {
    el.btnEksperymentyStart.disabled = true;
    try {
      await wyslij('/api/eksperymenty/start');
      el.logEksperymentow.textContent = 'Uruchamianie eksperymentów…';
      pokazInfo('Eksperymenty uruchomione.');
      await odswiezEksperymenty();
    } catch (blad) {
      pokazBlad(blad.message);
      el.btnEksperymentyStart.disabled = false;
    }
  }

  async function stopEksperymentow() {
    el.btnEksperymentyStop.disabled = true;
    try {
      await wyslij('/api/eksperymenty/stop');
      await odswiezEksperymenty();
      pokazInfo('Eksperymenty przerwane.');
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  function renderRaportEksperymentow(dane) {
    if (!dane || !dane.dostepny) {
      el.dataRaportuEksp.textContent = '—';
      el.podsumowanieEksperymentow.innerHTML =
        `<span class="text-muted">${escapeHtml(
          (dane && dane.komunikat) || 'Brak raportu — uruchom eksperymenty.',
        )}</span>`;
      el.wnioskiEksperymentow.innerHTML =
        '<span class="text-muted">Wnioski pojawią się po eksperymentach.</span>';
      el.tabelaEksperymentow.innerHTML =
        '<tr><td colspan="10" class="text-muted small">Brak danych.</td></tr>';
      el.wykresyEksperymentow.innerHTML =
        '<div class="col-12 text-muted small">Wykresy pojawią się po eksperymentach.</div>';
      return;
    }

    stan.raportEksperymentow = dane;
    const wersja = dane.mtime || Date.now();
    const eksperymenty = dane.eksperymenty || {};
    const informacje = dane.dane || {};
    const rozmiary = eksperymenty.rozmiar_zbioru || [];
    const wzorzec =
      rozmiary.find((wpis) => wpis.etykieta === 'wszystkie') ||
      rozmiary[rozmiary.length - 1];
    const metryki = (wzorzec && wzorzec.metryki) || {};
    const ci = (metryki.przedzialy_ufnosci || {}).metryki || {};

    el.dataRaportuEksp.textContent = dane.wygenerowano
      ? dane.wygenerowano.replace('T', ' ')
      : '—';

    const karty = [
      ['Accuracy (zamk.)', formatujProcent(metryki.accuracy_zamknieta), `kogo-kim · ${formatujCI(ci.accuracy_zamknieta)}`],
      ['Accuracy (otw.)', formatujProcent(metryki.accuracy_otwarta), `z odrzucaniem obcych · ${formatujCI(ci.accuracy)}`],
      [
        'EER',
        formatujProcent(metryki.eer),
        `${metryki.prog_eer != null ? `próg ${formatuj(metryki.prog_eer, 2)}` : '—'} · ${formatujCI(ci.eer)}`,
      ],
      ['FAR', formatujProcent(metryki.far), `${metryki.zaakceptowani_obcy ?? '—'} obcych · ${formatujCI(ci.far)}`],
      ['FRR', formatujProcent(metryki.frr), `${metryki.odrzucone_znane ?? '—'} odrzuconych · ${formatujCI(ci.frr)}`],
      ['AUC', metryki.auc != null ? formatuj(metryki.auc, 3) : '—', formatujCI(ci.auc, false)],
    ];

    el.podsumowanieEksperymentow.innerHTML =
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
      `<p class="small text-muted mt-2 mb-0">Osoby: ${escapeHtml(
        (informacje.osoby || []).join(', ') || '—',
      )}` +
      ` · trening: ${informacje.zdjecia_treningowe_razem ?? '—'} zdjęć` +
      ` · test: ${informacje.test ? informacje.test.zdjecia_znanych : '—'} zdjęć` +
      ` · obcy: ${informacje.obcy ? informacje.obcy.zdjecia : '—'} zdjęć` +
      ` · czas: ${formatuj(dane.czas_s, 0)} s</p>`;

    const sekcje = [
      ['Liczba zdjęć', rozmiary, (wpis) => wpis.etykieta],
      ['Jasność', eksperymenty.jasnosc || [], (wpis) => wpis.etykieta],
      ['k-NN', eksperymenty.knn || [], (wpis) => `k = ${wpis.k}`],
      [
        'Próg decyzyjny',
        eksperymenty.prog_decyzyjny || [],
        (wpis) => (wpis.prog != null ? formatuj(wpis.prog, 2) : '—'),
      ],
    ];

    const wiersze = [];
    sekcje.forEach(([nazwa, lista, opisWariantu]) => {
      lista.forEach((wpis) => {
        const m = wpis.metryki || wpis;
        wiersze.push(
          '<tr>' +
            `<td class="fw-semibold">${escapeHtml(nazwa)}</td>` +
            `<td>${escapeHtml(opisWariantu(wpis))}</td>` +
            `<td class="text-end text-mono">${wpis.liczba_zdjec_treningowych ?? '—'}</td>` +
            `<td class="text-end text-mono">${formatujProcent(m.accuracy_zamknieta)}</td>` +
            `<td class="text-end text-mono">${formatujProcent(m.accuracy_otwarta ?? m.accuracy)}</td>` +
            `<td class="text-end text-mono">${formatujProcent(m.far)}</td>` +
            `<td class="text-end text-mono">${formatujProcent(m.frr)}</td>` +
            `<td class="text-end text-mono">${formatujProcent(m.eer)}</td>` +
            `<td class="text-end text-mono">${m.auc != null ? formatuj(m.auc, 3) : '—'}</td>` +
            `<td class="text-end text-mono">${formatujCI(ciZRaportu(m, 'accuracy'))}</td>` +
            '</tr>',
        );
      });
    });

    el.tabelaEksperymentow.innerHTML = wiersze.length
      ? wiersze.join('')
      : '<tr><td colspan="10" class="text-muted small">Brak danych.</td></tr>';

    const wnioski = dane.wnioski || [];
    el.wnioskiEksperymentow.innerHTML = wnioski.length
      ? `<ol class="mb-0 ps-3">${wnioski
          .map((tekst) => `<li class="mb-2">${escapeHtml(tekst)}</li>`)
          .join('')}</ol>`
      : '<span class="text-muted">Raport nie zawiera wniosków.</span>';

    const wykresy = Object.entries(dane.wykresy || {});
    el.wykresyEksperymentow.innerHTML = wykresy.length
      ? wykresy
          .map(([klucz, plik]) => {
            const opis = OPIS_WYKRESOW_EKSPERYMENTOW[klucz] || klucz;
            const adres = `/api/eksperymenty/wykres/${encodeURIComponent(plik)}?v=${wersja}`;
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

    const ostrzezenia = informacje.ostrzezenia || [];
    el.ostrzezeniaEksperymentow.innerHTML = ostrzezenia.length
      ? ostrzezenia
          .map(
            (tekst, indeks) =>
              `<p class="${indeks === ostrzezenia.length - 1 ? 'mb-0' : 'mb-1'}">${escapeHtml(tekst)}</p>`,
          )
          .join('')
      : 'Eksperymenty raportują na zbiorze testowym — próg wybrano na walidacji.';
  }

  async function odswiezRaportEksperymentow() {
    try {
      const dane = await pobierz('/api/eksperymenty/raport');
      renderRaportEksperymentow(dane);
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  // -------------------------------------------------------- test trudny
  function renderTestTrudny(dane) {
    const [opis, klasa] = OPIS_TESTU_TRUDNEGO[dane.stan] || OPIS_TESTU_TRUDNEGO.bezczynny;
    el.badgeTestTrudny.textContent = opis;
    el.badgeTestTrudny.className = `badge ms-auto ${klasa}`;
    el.etapTestTrudny.textContent = dane.etap || 'Gotowy do uruchomienia';
    el.procentTestTrudny.textContent = `${Math.round(dane.postep || 0)}%`;
    el.pasekTestTrudny.style.width = `${ogranicz(dane.postep)}%`;

    const trwa = dane.stan === 'dziala';
    el.btnTestTrudnyStart.disabled = trwa;
    el.btnTestTrudnyStop.disabled = !trwa;
    el.pasekTestTrudny.classList.toggle('progress-bar-animated', trwa);
    el.pasekTestTrudny.classList.toggle('progress-bar-striped', trwa);

    el.czasTestTrudny.textContent = dane.czas_s ? `Czas: ${opisCzasu(dane.czas_s)}` : '—';

    el.alertTestTrudny.classList.toggle('d-none', dane.stan !== 'gotowe');
    if (dane.stan === 'gotowe') {
      el.alertTestTrudny.textContent =
        'Test trudny zakończony. Raport i wykresy są gotowe.';
    }

    const bladWidoczny = dane.stan === 'blad' || dane.stan === 'przerwany';
    el.alertTestTrudnyBlad.classList.toggle('d-none', !bladWidoczny);
    if (bladWidoczny) {
      el.alertTestTrudnyBlad.textContent =
        dane.blad || 'Test trudny nie został ukończony.';
    }

    const log = (dane.log || []).join('\n');
    if (log && el.logTestTrudny.textContent !== log) {
      el.logTestTrudny.textContent = log;
      el.logTestTrudny.scrollTop = el.logTestTrudny.scrollHeight;
    }
  }

  async function odswiezTestTrudny() {
    try {
      const dane = await pobierz('/api/test-trudny/status');
      renderTestTrudny(dane);
      if (dane.czas_raportu && dane.czas_raportu !== stan.czasRaportuTestTrudny) {
        stan.czasRaportuTestTrudny = dane.czas_raportu;
        await odswiezRaportTestuTrudnego();
      }
      return dane;
    } catch (blad) {
      pokazBlad(blad.message);
      return null;
    }
  }

  async function startTestuTrudnego() {
    el.btnTestTrudnyStart.disabled = true;
    try {
      await wyslij('/api/test-trudny/start');
      el.logTestTrudny.textContent = 'Uruchamianie testu trudnego…';
      pokazInfo('Test trudny uruchomiony.');
      await odswiezTestTrudny();
    } catch (blad) {
      pokazBlad(blad.message);
      el.btnTestTrudnyStart.disabled = false;
    }
  }

  async function stopTestuTrudnego() {
    el.btnTestTrudnyStop.disabled = true;
    try {
      await wyslij('/api/test-trudny/stop');
      await odswiezTestTrudny();
      pokazInfo('Test trudny przerwany.');
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  function renderRaportTestuTrudnego(dane) {
    if (!dane || !dane.dostepny) {
      el.dataRaportuTestTrudny.textContent = '—';
      el.podsumowanieTestTrudny.innerHTML =
        `<span class="text-muted">${escapeHtml(
          (dane && dane.komunikat) || 'Brak raportu — uruchom test trudny.',
        )}</span>`;
      el.tabelaPorownaniaTrudny.innerHTML =
        '<tr><td colspan="4" class="text-muted small">Brak danych.</td></tr>';
      el.tabelaMetrykTrudny.innerHTML =
        '<tr><td colspan="4" class="text-muted small">Brak danych.</td></tr>';
      el.wykresyTestTrudny.innerHTML =
        '<div class="col-12 text-muted small">Wykresy pojawią się po teście trudnym.</div>';
      return;
    }

    stan.raportTestuTrudnego = dane;
    const wersja = dane.mtime || Date.now();
    const prototypy = dane.prototypy || {};
    const metryki = prototypy.metryki || {};
    const ci = (prototypy.przedzialy_ufnosci || {}).metryki || {};
    const informacje = dane.dane || {};
    const porownanie = dane.porownanie_latwy_trudny || {};
    const latwy = porownanie.latwy || null;

    el.dataRaportuTestTrudny.textContent = dane.wygenerowano
      ? dane.wygenerowano.replace('T', ' ')
      : '—';

    const karty = [
      ['Accuracy', formatujProcent(metryki.accuracy), `${metryki.poprawne ?? '—'}/${metryki.liczba_znanych ?? '—'} · ${formatujCI(ci.accuracy)}`],
      ['FAR', formatujProcent(metryki.far), `${metryki.zaakceptowani_obcy ?? '—'} obcych · ${formatujCI(ci.far)}`],
      ['FRR', formatujProcent(metryki.frr), `${metryki.odrzucone_znane ?? '—'} odrzuconych · ${formatujCI(ci.frr)}`],
      ['EER', formatujProcent(prototypy.eer), `${prototypy.prog_eer != null ? `próg ${formatuj(prototypy.prog_eer, 2)}` : '—'} · ${formatujCI(ci.eer)}`],
      ['AUC', prototypy.auc != null ? formatuj(prototypy.auc, 3) : '—', formatujCI(ci.auc, false)],
    ];

    el.podsumowanieTestTrudny.innerHTML =
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
      `<p class="small text-muted mt-2 mb-0">Folder: ${escapeHtml(informacje.folder || '—')}` +
      ` · znani: ${informacje.zdjecia_znanych ?? '—'} zdjęć` +
      ` · obcy: ${informacje.zdjecia_obcych ?? '—'} zdjęć` +
      ` · czas: ${formatuj(dane.czas_s, 0)} s</p>` +
      ((informacje.brakujace_osoby || []).length
        ? `<p class="small text-warning-emphasis mb-0">Brak zdjęć osób: ${escapeHtml(
            informacje.brakujace_osoby.join(', '),
          )}</p>`
        : '');

    const pobierzWartosc = (zrodlo, klucz) => {
      if (!zrodlo) {
        return null;
      }
      if (klucz === 'eer' || klucz === 'auc') {
        return zrodlo[klucz] ?? null;
      }
      return (zrodlo.metryki || {})[klucz] ?? null;
    };

    const wiersze = [];
    const metrykiPorownania = [
      ['accuracy', 'Accuracy', true],
      ['far', 'FAR', true],
      ['frr', 'FRR', true],
      ['eer', 'EER', true],
      ['auc', 'AUC', false],
    ];
    metrykiPorownania.forEach(([klucz, nazwa, procent]) => {
      const stara = pobierzWartosc(latwy, klucz);
      const nowa = pobierzWartosc(prototypy, klucz);
      let roznica = '—';
      let klasa = '';
      let roznicaWartosc = null;
      if (stara != null && nowa != null) {
        roznicaWartosc = nowa - stara;
        // Dla FAR i EER mniej = lepiej, więc dodatnia różnica jest zła.
        const kierunek = klucz === 'far' || klucz === 'eer' ? -1 : 1;
        if (Math.abs(roznicaWartosc) > 1e-9) {
          klasa = roznicaWartosc * kierunek > 0 ? 'text-success' : 'text-danger';
        }
        roznica = procent
          ? `${roznicaWartosc >= 0 ? '+' : ''}${formatuj(roznicaWartosc * 100)} pkt proc.`
          : `${roznicaWartosc >= 0 ? '+' : ''}${formatuj(roznicaWartosc, 3)}`;
      }
      const sformatuj = (wartosc) =>
        wartosc == null ? '—' : procent ? formatujProcent(wartosc) : formatuj(wartosc, 3);
      wiersze.push(
        '<tr>' +
          `<td class="fw-semibold">${nazwa}</td>` +
          `<td class="text-end text-mono">${sformatuj(stara)}</td>` +
          `<td class="text-end text-mono">${sformatuj(nowa)}</td>` +
          `<td class="text-end text-mono ${klasa}">${escapeHtml(roznica)}</td>` +
          '</tr>',
      );
    });

    el.tabelaPorownaniaTrudny.innerHTML = wiersze.length
      ? wiersze.join('')
      : '<tr><td colspan="4" class="text-muted small">Brak danych z kroku 7.</td></tr>';

    const naOsobe = Object.entries(prototypy.na_osobe || {});
    el.tabelaMetrykTrudny.innerHTML = naOsobe.length
      ? naOsobe
          .map(
            ([osoba, metryka]) =>
              '<tr>' +
              `<td class="fw-semibold">${escapeHtml(osoba)}</td>` +
              `<td class="text-end text-mono">${metryka.zdjecia}</td>` +
              `<td class="text-end text-mono">${metryka.poprawne}</td>` +
              `<td class="text-end text-mono">${formatujProcent(metryka.recall)}</td>` +
              '</tr>',
          )
          .join('')
      : '<tr><td colspan="4" class="text-muted small">Brak danych.</td></tr>';

    const wykresy = Object.entries(dane.wykresy || {});
    el.wykresyTestTrudny.innerHTML = wykresy.length
      ? wykresy
          .map(([klucz, plik]) => {
            const opis = OPIS_WYKRESOW_TESTU_TRUDNEGO[klucz] || klucz;
            const adres = `/api/test-trudny/wykres/${encodeURIComponent(plik)}?v=${wersja}`;
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

    const ostrzezenia = informacje.ostrzezenia || [];
    el.ostrzezeniaTestTrudny.innerHTML = ostrzezenia.length
      ? ostrzezenia
          .map(
            (tekst, indeks) =>
              `<p class="${indeks === ostrzezenia.length - 1 ? 'mb-0' : 'mb-1'}">${escapeHtml(tekst)}</p>`,
          )
          .join('')
      : 'Trudny test mierzy model w innych warunkach niż trening.';
  }

  async function odswiezRaportTestuTrudnego() {
    try {
      const dane = await pobierz('/api/test-trudny/raport');
      renderRaportTestuTrudnego(dane);
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  // ----------------------------------------------------------- frekwencja
  const dzisiejszaData = () => {
    const teraz = new Date();
    const miesiac = String(teraz.getMonth() + 1).padStart(2, '0');
    const dzien = String(teraz.getDate()).padStart(2, '0');
    return `${teraz.getFullYear()}-${miesiac}-${dzien}`;
  };

  // Czas w bazie jest zapisany jako ISO, np. "2026-11-25T08:15:00".
  const godzina = (czas) =>
    typeof czas === 'string' && czas.length >= 19 ? czas.slice(11, 19) : '—';

  function renderRejestrStatus(dane) {
    stan.rejestr = dane;
    el.przelacznikRejestru.checked = Boolean(dane.wlaczony);
    el.statWpisow.textContent = String(dane.liczba_wpisow ?? 0);
    el.retencjaInfo.textContent = String(dane.retencja_dni ?? 90);

    const czesci = [];
    if (dane.plik) {
      czesci.push(`Baza: ${dane.plik}`);
    }
    if (dane.odstep_s) {
      czesci.push(`Zapis najwyżej co ${Math.round(dane.odstep_s / 60)} min`);
    }
    el.infoRejestru.textContent = czesci.join(' · ');

    if (dane.blad) {
      el.ostrzezenieRejestru.textContent = dane.blad;
      el.ostrzezenieRejestru.classList.remove('d-none');
    } else {
      el.ostrzezenieRejestru.classList.add('d-none');
    }
  }

  function renderDzienFrekwencji(dane) {
    const obecni = dane.obecni || [];
    const nieobecni = dane.nieobecni || [];
    const osoby = dane.osoby || [];

    el.statObecni.textContent = String(obecni.length);
    el.statNieobecni.textContent = String(nieobecni.length);
    el.statRozpoznan.textContent = String(dane.liczba_rozpoznan ?? 0);
    el.badgeFrekwencji.textContent = `${obecni.length}/${osoby.length}`;
    el.dataFrekwencji.value = dane.data || el.dataFrekwencji.value || dzisiejszaData();

    el.tabelaObecnych.innerHTML = obecni.length
      ? obecni
          .map(
            (wpis) =>
              '<tr>' +
              `<td class="ps-3">${escapeHtml(wpis.osoba)}</td>` +
              `<td>${godzina(wpis.pierwsze)}</td>` +
              `<td>${godzina(wpis.ostatnie)}</td>` +
              `<td class="text-end pe-3">${wpis.liczba}</td>` +
              '</tr>'
          )
          .join('')
      : '<tr><td colspan="4" class="text-muted small ps-3">Brak rozpoznań tego dnia.</td></tr>';

    el.listaNieobecnych.innerHTML = nieobecni.length
      ? nieobecni
          .map(
            (osoba) =>
              `<span class="badge text-bg-light border me-1 mb-1">${escapeHtml(osoba)}</span>`
          )
          .join('')
      : '<span class="text-muted">Brak listy osób — dodaj twarze i wytrenuj model.</span>';

    // Zachowujemy wybór użytkownika, bo lista jest przebudowywana przy
    // każdym odświeżeniu (co kilka sekund).
    const wybrana = el.osobaDopisz.value;
    el.osobaDopisz.innerHTML =
      '<option value="">Wybierz osobę…</option>' +
      osoby
        .map(
          (osoba) =>
            `<option value="${escapeHtml(osoba)}">${escapeHtml(osoba)}</option>`
        )
        .join('');
    if (osoby.includes(wybrana)) {
      el.osobaDopisz.value = wybrana;
    }
  }

  function renderOstatnieRejestru(wpisy) {
    el.tabelaOstatnich.innerHTML = wpisy.length
      ? wpisy
          .map((wpis) => {
            const podobienstwo =
              wpis.podobienstwo === null || wpis.podobienstwo === undefined
                ? '—'
                : `${formatuj(wpis.podobienstwo * 100)}%`;
            const zrodlo = wpis.zrodlo === 'recznie' ? 'ręcznie' : 'kamera';
            return (
              '<tr>' +
              `<td class="ps-3 text-nowrap">${escapeHtml(godzina(wpis.czas))}</td>` +
              `<td>${escapeHtml(wpis.osoba)}</td>` +
              `<td class="small text-muted">${zrodlo}</td>` +
              `<td class="text-end pe-3">${podobienstwo}</td>` +
              '</tr>'
            );
          })
          .join('')
      : '<tr><td colspan="4" class="text-muted small ps-3">Brak wpisów.</td></tr>';
  }

  async function odswiezFrekwencje() {
    const data = el.dataFrekwencji.value || dzisiejszaData();
    try {
      const [status, dzien, ostatnie] = await Promise.all([
        pobierz('/api/rejestr/status'),
        pobierz(`/api/rejestr/dzien?data=${encodeURIComponent(data)}`),
        pobierz('/api/rejestr/ostatnie?limit=20'),
      ]);
      renderRejestrStatus(status);
      renderDzienFrekwencji(dzien);
      renderOstatnieRejestru(ostatnie.wpisy || []);
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  async function przelaczRejestr() {
    const wlaczone = el.przelacznikRejestru.checked;
    try {
      const dane = await wyslij('/api/rejestr/wlacz', { wlaczone });
      renderRejestrStatus(dane.status || {});
      pokazInfo(
        wlaczone
          ? 'Rejestr włączony — rozpoznania będą zapisywane.'
          : 'Rejestr wyłączony — nowe rozpoznania nie będą zapisywane.'
      );
    } catch (blad) {
      el.przelacznikRejestru.checked = !wlaczone;
      pokazBlad(blad.message);
    }
  }

  async function dopiszObecnosc() {
    const osoba = el.osobaDopisz.value;
    const data = el.dataFrekwencji.value || dzisiejszaData();
    if (!osoba) {
      pokazBlad('Wybierz osobę do dopisania.');
      return;
    }
    el.btnDopiszObecnosc.disabled = true;
    try {
      await wyslij('/api/rejestr/dopisz', { osoba, data });
      pokazInfo(`Dopisano obecność: ${osoba}.`);
      await odswiezFrekwencje();
    } catch (blad) {
      pokazBlad(blad.message);
    } finally {
      el.btnDopiszObecnosc.disabled = false;
    }
  }

  async function eksportujFrekwencje() {
    const data = el.dataFrekwencji.value || dzisiejszaData();
    try {
      const odpowiedz = await fetch(
        `/api/rejestr/eksport?data=${encodeURIComponent(data)}`,
        { cache: 'no-store' }
      );
      if (!odpowiedz.ok) {
        let komunikat = 'Nie udało się wyeksportować CSV.';
        try {
          const dane = await odpowiedz.json();
          komunikat = dane.komunikat || komunikat;
        } catch (_) {
          /* odpowiedź mogła nie mieć JSON-a */
        }
        throw new Error(komunikat);
      }
      // Pobieramy plik jako blob, żeby pywebview nie opuszczał panelu.
      const blob = await odpowiedz.blob();
      const adres = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = adres;
      link.download = `frekwencja_${data}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(adres);
      pokazInfo(`Eksport CSV: frekwencja_${data}.csv`);
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  async function usunDzienRejestru() {
    const data = el.poleUsunDzien.value.trim();
    if (!data) {
      pokazBlad('Wpisz datę, aby potwierdzić usunięcie.');
      return;
    }
    el.btnUsunDzien.disabled = true;
    try {
      const dane = await wyslij('/api/rejestr/usun-dzien', {
        data,
        potwierdzenie: data,
      });
      pokazInfo(`Usunięto wpisów z dnia ${data}: ${dane.usunieto}.`);
      el.poleUsunDzien.value = '';
      await odswiezFrekwencje();
    } catch (blad) {
      pokazBlad(blad.message);
    } finally {
      el.btnUsunDzien.disabled = false;
    }
  }

  async function wyczyscRejestr() {
    const potwierdzenie = el.poleWyczysc.value.trim();
    if (potwierdzenie !== 'WYCZYSC') {
      pokazBlad('Wpisz WYCZYSC, aby potwierdzić wyczyszczenie rejestru.');
      return;
    }
    el.btnWyczyscRejestr.disabled = true;
    try {
      const dane = await wyslij('/api/rejestr/wyczysc', { potwierdzenie });
      pokazInfo(`Wyczyszczono rejestr. Usunięto wpisów: ${dane.usunieto}.`);
      el.poleWyczysc.value = '';
      await odswiezFrekwencje();
    } catch (blad) {
      pokazBlad(blad.message);
    } finally {
      el.btnWyczyscRejestr.disabled = false;
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
    el.wskazowkaZbierania.innerHTML = el.testTrudnyZbieranie.checked
      ? 'Tryb <strong>test trudny</strong>: zdjęcia trafią do <code>data/raw_test</code>. ' +
        'Zbierz je w innych warunkach niż trening (inny dzień, światło, kąt), ' +
        'żeby pomiar był uczciwy.'
      : 'Ustaw twarz w kadrze i kliknij <strong>Start</strong>. Zdjęcia zapisują ' +
        'się automatycznie co ok. 0,3 s. Zbierz różne kąty głowy, oświetlenie ' +
        'i wyraz twarzy (min. 150–300 zdjęć).';
  }

  // ----------------------------------------------------------------- init
  function init() {
    el.btnStart.addEventListener('click', start);
    el.btnStop.addEventListener('click', () => stop());

    el.prog.addEventListener('input', synchronizujSuwaki);
    el.proporcja.addEventListener('input', synchronizujSuwaki);
    el.korektaProgu.addEventListener('change', ustawKorekte);
    el.przelacznikEmocji.addEventListener('change', przelaczEmocje);
    el.btnWyjasnij.addEventListener('click', wyjasnijDecyzje);

    el.btnOdswiezKamery.addEventListener('click', odswiezKamery);
    el.btnOdswiezKameryModal.addEventListener('click', odswiezKamery);

    el.btnTreningStart.addEventListener('click', startTreningu);
    el.btnTreningStop.addEventListener('click', stopTreningu);
    el.btnEwaluacjaStart.addEventListener('click', startEwaluacji);
    el.btnEwaluacjaStop.addEventListener('click', stopEwaluacji);
    el.btnEksperymentyStart.addEventListener('click', startEksperymentow);
    el.btnEksperymentyStop.addEventListener('click', stopEksperymentow);
    el.btnTestTrudnyStart.addEventListener('click', startTestuTrudnego);
    el.btnTestTrudnyStop.addEventListener('click', stopTestuTrudnego);
    el.btnOdswiezBaze.addEventListener('click', odswiezBaze);
    el.btnPotwierdzUsun.addEventListener('click', potwierdzUsuniecie);

    el.dataFrekwencji.value = dzisiejszaData();
    el.btnOdswiezFrekwencje.addEventListener('click', odswiezFrekwencje);
    el.dataFrekwencji.addEventListener('change', odswiezFrekwencje);
    el.btnEksportFrekwencji.addEventListener('click', eksportujFrekwencje);
    el.przelacznikRejestru.addEventListener('change', przelaczRejestr);
    el.btnDopiszObecnosc.addEventListener('click', dopiszObecnosc);
    el.btnUsunDzien.addEventListener('click', usunDzienRejestru);
    el.btnWyczyscRejestr.addEventListener('click', wyczyscRejestr);

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
    el.testTrudnyZbieranie.addEventListener('change', aktualizujWskazowkeZbierania);

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

    // Eksperymenty: tak samo — poller działa tylko na aktywnym widoku.
    setInterval(() => {
      if (stan.widok === 'eksperymenty') {
        odswiezEksperymenty();
      }
    }, 1000);

    // Test trudny: poller tylko na jego widoku.
    setInterval(() => {
      if (stan.widok === 'test-trudny') {
        odswiezTestTrudny();
      }
    }, 1000);

    // Frekwencja: odświeżamy rzadziej — wpisy zmieniają się co najwyżej
    // raz na kilka minut (cooldown), a każde odświeżenie czyta bazę.
    setInterval(() => {
      if (stan.widok === 'frekwencja') {
        odswiezFrekwencje();
      }
    }, 5000);
  }

  init();
})();
