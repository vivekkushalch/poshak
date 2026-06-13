-- poshak image-search service — vector store schema.
--
-- Idempotent: safe to run repeatedly. The migration *from* the old
-- 512-d single-vec schema (image-search/schema.sql) is handled in
-- scripts/init_db.py — that script detects the old shape and DROPs it
-- BEFORE running this file.
--
-- Apply via:
--   docker compose exec image-search python -m scripts.init_db

CREATE TABLE IF NOT EXISTS `stock_image_vectors` (
  `id`         INT AUTO_INCREMENT PRIMARY KEY,
  `stock_id`   INT NOT NULL,
  -- Drives which in-memory index the row joins at boot. Derived from
  -- stockitem.name during reindex (CLOTH_ITEMS / FABRIC_ITEMS in
  -- service/config.py).
  `category`   ENUM('cloth','fabric') NOT NULL,
  `photo_name` VARCHAR(255) NOT NULL,
  -- Filename of the bg-removed PNG counterpart in /corpus/data. NULL
  -- for fabric and for cloth rows where bg-removal hasn't run yet.
  `rembg_name` VARCHAR(255) NULL,
  `orig_vec`   JSON NOT NULL,
  `rembg_vec`  JSON NULL,
  -- Service refuses to load rows whose model_tag doesn't match its
  -- current MODEL_TAG — the guard that stops a half-migrated index
  -- from serving mixed-model results.
  `model_tag`  VARCHAR(64) NOT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY `uk_photo_model` (`photo_name`, `model_tag`),
  KEY `idx_stock` (`stock_id`),
  KEY `idx_cat`   (`category`),
  KEY `idx_model` (`model_tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
