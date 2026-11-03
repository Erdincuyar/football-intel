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
