# footintel: Football Tactical & Betting Intelligence Platform

Premier League, La Liga, Süper Lig, Serie A ve Ligue 1 maçlarını gerçek ve ücretsiz
veri kaynaklarıyla inceleyen, taktiksel eşleşmeleri değerlendiren, bahis piyasaları için
olasılıksal tahminler üreten ve kendi tahmin performansını ölçen bir analiz platformu.

> Durum: **Faz 2 / Adım 1.** football-data.co.uk veri hattı (indirme, temizleme, DuckDB).

## Kurulum

```bash
# uv: hızlı Python paket ve ortam yöneticisi (https://docs.astral.sh/uv/)
uv sync                                     # sanal ortam + bağımlılıklar
uv run pytest                               # testler
uv run python pipelines/build_football_data.py   # veriyi indir, temizle, yükle
```

## Veriyi almak

football-data.co.uk **Türkiye'den erişime kapalı**. Bu yüzden veri hattı GitHub
Actions'ta (yurt dışındaki sunucularda) çalışır: Pazartesi ve Perşembe sabahı otomatik,
ya da **Actions → data → Run workflow** ile elle. Kapsama raporu her çalıştırmanın
özet sayfasında görünür. Temiz veriyi bilgisayarına indirmek için
([GitHub CLI](https://cli.github.com) gerekir, bir kez `gh auth login`):

```bash
gh run download --repo Erdincuyar/football-intel --name footintel-data --dir data
```

Bu komut `data/footintel.duckdb` ve `data/clean/*.parquet` dosyalarını yerine koyar.

## Tahmin modelleri (Faz 3)

- `src/footintel/models/strength.py`: Dixon-Coles gol modeli ve Negatif Binom/Poisson
  korner modeli (zaman ağırlıklı, analitik gradyanlı). Skor matrisinden 1X2, KG,
  Ü/A 1.5-2.5-3.5, doğru skor ve takım golleri; korner dağılımından korner Ü/A.
- `src/footintel/models/backtest.py`: Walk-forward backtest. Her hafta yalnızca geçmiş
  maçlarla yeniden fit edip piyasa ve basit baseline'larla karşılaştırır.
- `pipelines/build_models.py`: 2024/25'te ayar, 2025/26'da test. Rapor CI özetinde.

## Tahmin defteri ve otomasyon (Faz 4)

CI her sabah (06:00 UTC) şunları yapar:
1. football-data.co.uk fikstür listesini indirir.
2. Önümüzdeki 10 gündeki, henüz tahmin edilmemiş maçlar için tüm pazarların olasılıklarını
   hesaplar (`pipelines/predict_upcoming.py`, model sürümü `src/footintel/models/registry.py`).
3. Bunları `data/predictions/` altına **yeni** bir Parquet dosyası olarak yazar ve git'e
   commit'ler. Defter yalnızca büyür: bir maç aynı model sürümüyle bir kez tahmin edilir,
   var olan dosyanın üzerine yazılmaz. Commit zamanı maç saatinden önce olduğu için
   tahminlerin sonradan değiştirilmediği herkes tarafından doğrulanabilir.
4. Oynanan maçları değerlendirir (`pipelines/evaluate_ledger.py` → `prediction_eval`).

### Arayüz

```bash
gh run download --repo Erdincuyar/football-intel --name footintel-data --dir data
uv sync --group app
uv run streamlit run app/Home.py
```

Ana sayfa yaklaşan maçların olasılıklarını, **Model Performansı** sayfası canlı defterin ve
walk-forward backtest'in sonuçlarını (piyasayla kıyas, kalibrasyon) gösterir.

## Taktik metrikler (Faz 6)

Olay verisi (her pasın, presin, şutun sahadaki konumu) ücretli bir ürün. Bu yüzden iki
katmanlı bir yol izliyoruz:

1. **StatsBomb açık verisi, 2015/16** (PL, La Liga, Serie A, Ligue 1; 1517 maç):
   PPDA, savunma yüksekliği, ileri pres oranı, field tilt, xG/xT payı, uzun pas ve orta
   oranı (`src/footintel/features/tactics.py`). xT (expected threat) ızgarası aynı veriden
   fit edilir.
2. **Güncel sezon stil profili** (5 lig, Süper Lig dahil): football-data'nın şut, isabetli
   şut, korner ve faul sayılarından. Zengin metriklerin bir kısmı bu sayılardan doğrusal bir
   modelle tahmin edilir. Model StatsBomb verisinde, her seferinde bir lig dışarıda
   bırakılarak test edilir ve yalnızca dış ligde R² ≥ 0.6 olan metrikler kullanılır.

2015/16 sonuçları (`pipelines/build_tactics.py`):

| Soru | Bulgu |
| --- | --- |
| Hangi metrik kalıcı bir stil? | Pas isabeti, uzun pas oranı, pas payı (sezon güvenilirliği ≥ 0.95). Korner payı en gürültülü (0.60). |
| İlk yarı, ikinci yarı puanını ne kadar açıklar? (R²) | xG payı 0.45 > gol farkı 0.42 > şut payı 0.38 ≈ isabetli şut payı 0.37 > korner payı 0.28 |
| Basit sayılardan tahmin edilebilir mi? (şut, isabetli şut ve korner payından; dış lig R²) | xG payı 0.88, xT payı 0.87, field tilt 0.77 kullanılıyor. Pas payı 0.58 ile eşiğin altında. **PPDA 0.10: pres yoğunluğu bu sayılardan çıkmıyor.** |
| İki kaynak aynı şeyi mi sayıyor? (2015/16, maç başı) | Şut, isabetli şut ve korner %1-3 farkla aynı. Faulde StatsBomb %8-17 fazla sayıyor, bu yüzden faul proxy'ye girmiyor. |

Arayüzde **Taktik Profil** sayfası. Veri: [StatsBomb Open Data](https://github.com/statsbomb/open-data)
(kaynak gösterilerek, ticari olmayan kullanım).

## Taktiksel eşleşme (Faz 7)

Kurallar veriye bakılmadan önce yazıldı (`src/footintel/analysis/rules.py`). Her kural,
bir takımın gücünü rakibin ilgili zayıflığıyla eşleştiriyor. Örnek: "çok korner kazanan
takım, korner veren rakibe karşı modelin beklediğinden fazla korner kazanır."

- **Profil:** Her takımın o maçtan önceki son 10 maçı (football-data). "Yüksek" ve "düşük"
  eşikleri aynı ligin önceki sezonlarından gelir, böylece sızıntı olmaz.
- **Test:** Kural, modelin artığına karşı test edilir (gerçekleşen eksi Dixon-Coles veya
  korner modeli). Modelin zaten bildiğini kural diye saymamak için bu gerekli.
- **Kapı:** Bir kural iki şartla tutulur: 2022-24 keşif döneminde Benjamini-Hochberg
  düzeltmesinden sonra anlamlı olmalı, 2024-26 doğrulama döneminde de aynı yönde
  tekrarlamalı.
- **Stil grupları:** Takım-sezon profillerine K-Means uygulanır. Gruplar StatsBomb
  2015/16 metrikleriyle yorumlanır.

Arayüzde **Eşleşme** sayfası. Sonuçlar `pipelines/build_matchup.py` çıktısında ve CI
özetinde.

## Yayın: Neon + Streamlit Community Cloud (Faz 8)

Temiz veri CI'ın içinde üretilir ve repoya yazılmaz (her çalışmada parquet commit'lemek repoyu
şişirir). Bu yüzden CI, uygulamanın okuduğu tabloları ücretsiz bir Neon PostgreSQL
veritabanına (`app` şeması) yazar; Streamlit uygulaması oradan okur. Bağlantı adresi
tanımlı değilse uygulama `data/clean/` altındaki dosyalara düşer, yani yerel geliştirme
aynen çalışır (`src/footintel/store.py`).

| Tablo (`app.` şeması) | İçerik |
|---|---|
| `matches_football_data` | 5 lig, maç sonuçları, istatistikler, oranlar |
| `backtest_predictions` | walk-forward tahminleri (Model Performansı) |
| `predictions` | tahmin defteri (repodaki `data/predictions/` ile aynı) |
| `style_profile`, `style_now` | güncel stil profilleri (Taktik Profil, Eşleşme) |
| `tactics_team_match_1516` | StatsBomb 2015/16 takım-maç metrikleri |
| `json_docs` | küçük JSON çıktılar (kurallar, eşikler, gruplar, xT ızgarası) |
| `meta` | son yayın zamanı |

Kurulum (bir kez, ikisi de ücretsiz):

1. **Neon**: neon.com'da hesap aç, bir proje oluştur, *Connection string*'i kopyala
   (`postgresql://...?sslmode=require`).
2. **GitHub**: repo > Settings > Secrets and variables > Actions > *New repository secret*:
   ad `FOOTINTEL_DATABASE_URL`, değer bağlantı adresi. Bir sonraki main çalışması veriyi yazar
   (Actions'tan *Run workflow* ile hemen de tetiklenebilir).
3. **Streamlit Community Cloud**: share.streamlit.io'da GitHub ile giriş yap, *Create app*:
   repo `Erdincuyar/football-intel`, branch `main`, dosya `app/Home.py`. *Advanced settings >
   Secrets* kısmına:
   ```toml
   FOOTINTEL_DATABASE_URL = "postgresql://..."
   ```

Streamlit Cloud bağımlılıkları `app/requirements.txt`'ten kurar. Bu dosya `uv.lock`'tan
üretilir ve CI aynı olup olmadığını kontrol eder:

```bash
uv export -q --no-hashes --no-dev --group app --no-emit-project \
  --format requirements-txt > app/requirements.txt
```

## Keşifsel analiz (EDA)

`notebooks/01_eda.ipynb` gol, korner, ev avantajı ve bahis piyasası analizini içerir.
CI her çalıştırmada notebook'u gerçek veriyle çalıştırır ve HTML olarak saklar:

```bash
gh run download --repo Erdincuyar/football-intel --name eda-report --dir reports
open reports/01_eda.html
```

Yerelde çalıştırmak için önce veriyi indir (yukarıda), sonra:

```bash
uv sync --group eda
uv run jupyter lab notebooks/01_eda.ipynb
```
Erişimin olan bir ağdaysan hattı yerelde de çalıştırabilirsin:
`uv run python pipelines/build_football_data.py`.

## Veri kaynakları

| Kaynak | Kullanım | Not |
| --- | --- | --- |
| [football-data.co.uk](https://www.football-data.co.uk/data.php) | Sonuç, şut, korner, kart, bahis oranları | "Yalnızca maç tahmini amaçlı" kullanım koşulu |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | 2015/16 olay verisi (taktik metrikler) | Kaynak gösterilmeli; Süper Lig yok |

Ham veri repoya eklenmez. Pipeline ile herkes kendi kopyasını indirir.

## Proje yapısı

```
src/footintel/   analiz paketi (ingest, clean, db, config)
pipelines/       uçtan uca çalıştırılabilir betikler
sql/             veritabanı şeması
tests/           pytest testleri (fixtures/ = sentetik örnekler)
data/            raw/ ve clean/ git'e girmez; predictions/ girer
```

## Sık karşılaşılan sorunlar

**`CERTIFICATE_VERIFY_FAILED: self-signed certificate`**: Bilgisayarındaki bir
antivirüs (HTTPS tarama), kurum VPN'i veya ağ filtresi bağlantıya kendi sertifikasını
koyuyor. Kod, `truststore` ile işletim sisteminin sertifika deposunu kullanır; o
sertifika Keychain'de güvenilir olarak kayıtlıysa sorun çözülür. Çözülmezse
sertifikayı veren yazılımı bul:

```bash
openssl s_client -connect www.football-data.co.uk:443 -servername www.football-data.co.uk </dev/null 2>/dev/null | openssl x509 -noout -issuer
```

Çıkan `issuer` gerçek bir sertifika otoritesi değilse (örneğin bir antivirüs adı),
o yazılımda HTTPS/web taramasını bu site için kapat. **SSL doğrulamasını kodda
kapatma** (`verify=False`): bu, araya giren herkesin veriyi değiştirmesine izin verir.

**`VIRTUAL_ENV ... does not match the project environment`**: Başka bir sanal ortam
(örneğin pyenv) aktif. Zararsız bir uyarı; uv projenin kendi `.venv` ortamını kullanır.
Susturmak için önce `pyenv deactivate` (veya `deactivate`) çalıştır.
