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
