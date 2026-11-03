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
