/* Logika panelu: sterowanie API, strumień MJPEG i odświeżanie statystyk. */
(() => {
  'use strict';

  const el = {
    badgeStatus: document.getElementById('badgeStatus'),
    statStatus: document.getElementById('statStatus'),
    statFps: document.getElementById('statFps'),
    statTwarze: document.getElementById('statTwarze'),
    statKamera: document.getElementById('statKamera'),
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
    btnOdswiezKameryModal: document.getElementById('btnOdswiezKameryModal'),
    listaKamer: document.getElementById('listaKamer'),
    prog: document.getElementById('prog'),
    progWartosc: document.getElementById('progWartosc'),
    proporcja: document.getElementById('proporcja'),
    proporcjaWartosc: document.getElementById('proporcjaWartosc'),
    toast: document.getElementById('toastBlad'),
    toastTresc: document.getElementById('toastBladTresc'),
    btnSidebar: document.getElementById('btnSidebar'),
    sidebarBackdrop: document.getElementById('sidebarBackdrop'),
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

  const toast = bootstrap.Toast.getOrCreateInstance(el.toast, { delay: 7000 });
  let ostatniBlad = '';
  let maKamery = false;
  let zajete = false;

  const formatuj = (wartosc, miejsca = 1) =>
    Number(wartosc || 0).toFixed(miejsca).replace('.', ',');

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
    if (!tresc || tresc === ostatniBlad) {
      return;
    }
    ostatniBlad = tresc;
    el.toastTresc.textContent = tresc;
    toast.show();
    setTimeout(() => {
      if (ostatniBlad === tresc) {
        ostatniBlad = '';
      }
    }, 8000);
  }

  function pokazZatrzymany() {
    el.videoWrap.classList.remove('aktywne');
    el.video.removeAttribute('src');
    el.placeholderSpinner.classList.add('d-none');
    el.placeholderTitle.textContent = 'Podgląd zatrzymany';
    el.placeholderText.textContent = 'Wybierz kamerę i kliknij „Start”.';
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

  function zastosujStatus(dane) {
    const stan = dane.stan || 'zatrzymany';
    const opis = OPIS_STANU[stan] || stan;

    el.badgeStatus.textContent = opis;
    el.badgeStatus.className = `badge ${KLASA_STANU[stan] || 'text-bg-secondary'}`;
    el.statStatus.textContent = opis;
    el.statFps.textContent = formatuj(dane.fps);
    el.statTwarze.textContent = dane.twarze ?? 0;
    el.statKamera.textContent = dane.dziala ? `Kamera ${dane.kamera}` : '—';

    el.btnStart.disabled = zajete || Boolean(dane.dziala);
    el.btnStop.disabled = zajete || !dane.dziala;
    el.kameraSelect.disabled = Boolean(dane.dziala) || !maKamery;
    el.prog.disabled = Boolean(dane.dziala);
    el.proporcja.disabled = Boolean(dane.dziala);

    if (dane.dziala && !el.video.hasAttribute('src')) {
      pokazPodglad();
    }

    if (stan === 'blad' && dane.blad) {
      el.statusText.textContent = 'Błąd kamery.';
      pokazBlad(dane.blad);
      pokazZatrzymany();
    }
  }

  async function odswiezStatus() {
    try {
      zastosujStatus(await pobierz('/api/status'));
    } catch (_) {
      /* pojedyncze nieudane odświeżenie nie powinno zasypywać użytkownika */
    }
  }

  async function start() {
    const indeks = Number.parseInt(el.kameraSelect.value, 10);
    if (Number.isNaN(indeks)) {
      pokazBlad('Najpierw wybierz kamerę z listy.');
      return;
    }

    el.btnStart.disabled = true;
    el.btnStop.disabled = true;
    zajete = true;
    el.statusText.textContent = 'Uruchamianie kamery…';
    pokazUruchamianie();

    try {
      await wyslij('/api/start', {
        kamera: indeks,
        prog: Number.parseFloat(el.prog.value),
        proporcja: Number.parseFloat(el.proporcja.value),
      });
      pokazPodglad();
      el.statusText.textContent = 'Podgląd działa.';
    } catch (blad) {
      pokazBlad(blad.message);
      el.statusText.textContent = 'Nie udało się uruchomić kamery.';
      pokazZatrzymany();
    } finally {
      zajete = false;
      await odswiezStatus();
    }
  }

  async function stop() {
    el.btnStop.disabled = true;
    zajete = true;
    el.statusText.textContent = 'Zatrzymywanie…';
    try {
      await wyslij('/api/stop');
      el.statusText.textContent = 'Zatrzymano.';
    } catch (blad) {
      pokazBlad(blad.message);
    } finally {
      zajete = false;
      pokazZatrzymany();
      await odswiezStatus();
    }
  }

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
      maKamery = lista.length > 0;

      if (!maKamery) {
        el.kameraSelect.innerHTML = '<option value="">Brak kamer</option>';
        el.kameraSelect.disabled = true;
        el.btnStart.disabled = true;
        return;
      }

      el.kameraSelect.innerHTML = lista
        .map(
          (kamera) =>
            `<option value="${kamera.indeks}">` +
            `Kamera ${kamera.indeks} (${kamera.szerokosc}×${kamera.wysokosc})` +
            `${kamera.cztery_k ? ' — prawdopodobnie 4K' : ''}` +
            `</option>`,
        )
        .join('');
      el.kameraSelect.disabled = false;
      el.btnStart.disabled = false;
    } catch (blad) {
      pokazBlad(blad.message);
    }
  }

  async function odswiezKamery() {
    el.kameraSelect.innerHTML = '<option value="">Szukam kamer…</option>';
    el.kameraSelect.disabled = true;
    maKamery = false;
    try {
      await wyslij('/api/kamery/odswiez');
      await zaladujKamery();
    } catch (blad) {
      pokazBlad(blad.message);
      await zaladujKamery();
    }
  }

  function synchronizujSuwaki() {
    el.progWartosc.textContent = formatuj(el.prog.value, 2);
    el.proporcjaWartosc.textContent = formatuj(el.proporcja.value, 2);
  }

  function przelaczSidebar() {
    if (window.innerWidth < 992) {
      document.body.classList.toggle('sidebar-otwarty');
    } else {
      document.body.classList.toggle('sidebar-zwiniety');
    }
  }

  function init() {
    el.btnStart.addEventListener('click', start);
    el.btnStop.addEventListener('click', stop);
    el.prog.addEventListener('input', synchronizujSuwaki);
    el.proporcja.addEventListener('input', synchronizujSuwaki);

    el.btnOdswiezKamery.addEventListener('click', odswiezKamery);
    el.btnOdswiezKameryModal.addEventListener('click', odswiezKamery);

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

    // Modal "Wykryte kamery" zawsze pokazuje świeże dane.
    document
      .getElementById('modalKamery')
      .addEventListener('show.bs.modal', () => {
        pobierz('/api/kamery').then(renderujListeKamer).catch(() => {});
      });

    synchronizujSuwaki();
    zaladujKamery();
    odswiezStatus();
    setInterval(odswiezStatus, 1000);
  }

  init();
})();
