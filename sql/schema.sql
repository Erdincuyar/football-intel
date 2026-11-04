-- Faz 2 / Adim 1: cekirdek tablolar.
-- DuckDB ve PostgreSQL ile uyumlu SQL yazmaya calisiyoruz; yayina gecerken
-- (Faz 8) ayni sema Neon PostgreSQL'de de calissin.

CREATE TABLE IF NOT EXISTS dim_league (
    league_id   VARCHAR PRIMARY KEY,   -- E0, SP1, T1, I1, F1
    name        VARCHAR NOT NULL,
    country     VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_team (
    team_id         VARCHAR PRIMARY KEY,   -- slug: 'man-united', 'fenerbahce'
    canonical_name  VARCHAR NOT NULL       -- football-data.co.uk yazimi
);

-- Kaynaklar arasi isim eslemesi. Adim 2'de Understat isimleri buraya eklenecek.
CREATE TABLE IF NOT EXISTS map_team_source (
    source            VARCHAR NOT NULL,    -- 'football_data', 'understat', ...
    source_team_name  VARCHAR NOT NULL,
    team_id           VARCHAR NOT NULL REFERENCES dim_team (team_id),
    PRIMARY KEY (source, source_team_name)
);

CREATE TABLE IF NOT EXISTS fact_match (
    match_id              VARCHAR PRIMARY KEY,
    league_id             VARCHAR NOT NULL REFERENCES dim_league (league_id),
    season                VARCHAR NOT NULL,   -- '2526'
    season_label          VARCHAR NOT NULL,   -- '2025/26'
    match_date            DATE NOT NULL,
    kickoff_time          VARCHAR,            -- kaynakta Birlesik Krallik saati
    home_team_id          VARCHAR NOT NULL REFERENCES dim_team (team_id),
    away_team_id          VARCHAR NOT NULL REFERENCES dim_team (team_id),
    result                VARCHAR,            -- H / D / A
    referee               VARCHAR,
    quality_flags         VARCHAR,            -- temizlikte NA yapilan degerlerin notu
    home_goals            INTEGER,
    away_goals            INTEGER,
    home_goals_ht         INTEGER,
    away_goals_ht         INTEGER,
    home_shots            INTEGER,
    away_shots            INTEGER,
    home_shots_on_target  INTEGER,
    away_shots_on_target  INTEGER,
    home_corners          INTEGER,
    away_corners          INTEGER,
    home_fouls            INTEGER,
    away_fouls            INTEGER,
    home_yellow           INTEGER,
    away_yellow           INTEGER,
    home_red              INTEGER,
    away_red              INTEGER,
    loaded_at             TIMESTAMP NOT NULL DEFAULT current_timestamp
);

-- Oranlar ayri tabloda: bir maca bircok piyasa ve zaman noktasi (acilis, kapanis)
-- baglanir. Genis bir tabloya kolon eklemek yerine uzun format (bir satir = bir oran)
-- yeni piyasa eklemeyi sema degisikligi gerektirmeden yapar.
-- Not: match_id icin FOREIGN KEY bilerek yok. DuckDB, ayni transaction icinde
-- silinen satirlari yabanci anahtar kontrolunde hala "var" sayar; bu yuzden
-- "sil + yeniden yukle" adimimiz hata verir. Butunlugu db.check_integrity() ile
-- kendimiz kontrol ediyoruz. PostgreSQL'e gecince (Faz 8) FK geri eklenecek.
CREATE TABLE IF NOT EXISTS fact_odds (
    match_id    VARCHAR NOT NULL,
    market      VARCHAR NOT NULL,   -- '1x2', 'ou25'
    selection   VARCHAR NOT NULL,   -- 'home', 'draw', 'away', 'over', 'under'
    snapshot    VARCHAR NOT NULL,   -- 'avg' (yayin anindaki ortalama), 'close' (kapanis)
    odds        DOUBLE NOT NULL,
    PRIMARY KEY (match_id, market, selection, snapshot)
);

-- Faz 2 / Adim 2 ------------------------------------------------------------

-- Understat mac istatistikleri (Super Lig haric 4 lig). football-data maciyla 1-1.
CREATE TABLE IF NOT EXISTS fact_match_understat (
    match_id            VARCHAR PRIMARY KEY,   -- fact_match.match_id
    understat_match_id  INTEGER NOT NULL,
    home_xg             DOUBLE,
    away_xg             DOUBLE,
    home_npxg           DOUBLE,
    away_npxg           DOUBLE,
    home_ppda           DOUBLE,   -- dusuk = yogun baski
    away_ppda           DOUBLE,
    home_deep           INTEGER,
    away_deep           INTEGER,
    home_xpts           DOUBLE,
    away_xpts           DOUBLE
);

-- ClubElo puan gecmisi: her satir, puanin gecerli oldugu tarih araligi.
-- Bir macin oncesindeki puani bulmak icin: valid_from <= mac_tarihi - 1 <= valid_to
CREATE TABLE IF NOT EXISTS fact_team_elo (
    team_id     VARCHAR NOT NULL,
    valid_from  DATE NOT NULL,
    valid_to    DATE NOT NULL,
    elo         DOUBLE NOT NULL,
    PRIMARY KEY (team_id, valid_from)
);

-- Kendi Elo modelimiz (features/elo.py). Mac ONCESI puanlar: tahmin ozelligi olarak
-- kullanilmaya uygun, o mactan ve sonrasindan bilgi icermez.
CREATE TABLE IF NOT EXISTS fact_match_elo (
    match_id        VARCHAR PRIMARY KEY,
    home_elo_pre    DOUBLE NOT NULL,
    away_elo_pre    DOUBLE NOT NULL,
    home_expected   DOUBLE NOT NULL   -- ev sahibinin beklenen skoru (0-1)
);

-- Faz 3 ----------------------------------------------------------------------

-- Walk-forward backtest tahminleri. Her satir, fit_date'ten ONCEKI maclarla egitilmis
-- modelin o mac icin verdigi olasiliklar. Faz 5'teki degismez tahmin defterinin
-- (ledger) gecmise donuk karsiligi.
CREATE TABLE IF NOT EXISTS backtest_prediction (
    match_id            VARCHAR PRIMARY KEY,
    league              VARCHAR NOT NULL,
    season              VARCHAR NOT NULL,
    fit_date            DATE NOT NULL,
    p_home              DOUBLE, p_draw DOUBLE, p_away DOUBLE,
    p_btts              DOUBLE,
    exp_home_goals      DOUBLE, exp_away_goals DOUBLE,
    p_over15            DOUBLE, p_over25 DOUBLE, p_over35 DOUBLE,
    p_home_over05       DOUBLE, p_home_over15 DOUBLE,
    p_away_over05       DOUBLE, p_away_over15 DOUBLE,
    ll_score            DOUBLE,   -- log P(gerceklesen kesin skor)
    exp_corners         DOUBLE, exp_home_corners DOUBLE, exp_away_corners DOUBLE,
    p_corners_over85    DOUBLE, p_corners_over95 DOUBLE,
    p_corners_over105   DOUBLE, p_corners_over115 DOUBLE,
    ll_corners          DOUBLE    -- log P(gerceklesen toplam korner)
);

-- Faz 4 ----------------------------------------------------------------------

-- Degistirilemez tahmin defteri. Kaynak: data/predictions/*.parquet (git'te, commit
-- zamani tahminin mac oncesi yapildiginin kaniti). DuckDB'de UPDATE/DELETE engellenemez;
-- Faz 8'de PostgreSQL'e gecince trigger ile engellenecek.
CREATE TABLE IF NOT EXISTS prediction_log (
    prediction_id       VARCHAR PRIMARY KEY,
    match_key           VARCHAR NOT NULL,   -- lig-sezon-ev-dep (tarihten bagimsiz)
    league              VARCHAR NOT NULL,
    season              VARCHAR NOT NULL,
    match_date          DATE NOT NULL,
    kickoff_time        VARCHAR,
    home_team           VARCHAR NOT NULL,
    away_team           VARCHAR NOT NULL,
    model_version       VARCHAR NOT NULL,
    market              VARCHAR NOT NULL,   -- 1x2, ou25, btts, correct_score, corners_ou95 ...
    selection           VARCHAR NOT NULL,   -- home/draw/away, over/under, yes/no, '2-1' ...
    prob                DOUBLE NOT NULL,
    market_prob         DOUBLE,             -- tahmin anindaki piyasa olasiligi (varsa)
    created_at          TIMESTAMP NOT NULL,
    data_snapshot_hash  VARCHAR NOT NULL
);

-- Mac bitince hesaplanir; defter degismez.
CREATE TABLE IF NOT EXISTS prediction_eval (
    prediction_id  VARCHAR PRIMARY KEY,
    outcome        DOUBLE NOT NULL,   -- 1 = secenek gerceklesti
    brier          DOUBLE NOT NULL,
    log_loss       DOUBLE NOT NULL
);
