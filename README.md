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
