-- =============================================================================
-- BigQuery 抽取脚本 —— 论文全部原始数据
-- =============================================================================
-- 数据源: bigquery-public-data.crypto_ethereum.token_transfers
-- 观测窗口: [2026-03-26 00:00:00 UTC, 2026-06-25 00:00:00 UTC)
--           左闭右开，实际覆盖至 2026-06-24 23:59:59 UTC
--           ★ 两个代币严格一致 —— 这是跨代币对比成立的前提
--
-- 校验值（抽取后请核对行数）:
--   LINK 转账: 1,233,497 条 | 活跃地址 642,126
--   UNI  转账:   433,856 条 | 活跃地址  38,667
--
-- 注: value 列为 STRING 类型（容纳 uint256），下游脚本统一 /1e18 换算。
--     余额查询不在 SQL 内做除法，由 common.load_balances() 自动识别单位，
--     以保证两个代币的处理路径完全相同。
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. LINK 转账
-- -----------------------------------------------------------------------------
SELECT
  from_address     AS `from`,
  to_address       AS `to`,
  value,
  transaction_hash AS evt_tx_hash,
  block_timestamp
FROM `bigquery-public-data.crypto_ethereum.token_transfers`
WHERE token_address = '0x514910771af9ca656af840dff83e8264ecf986ca'
  AND block_timestamp >= TIMESTAMP('2026-03-26 00:00:00 UTC')
  AND block_timestamp <  TIMESTAMP('2026-06-25 00:00:00 UTC');


-- -----------------------------------------------------------------------------
-- 2. UNI 转账（与查询 1 逐字对应，仅合约地址不同）
-- -----------------------------------------------------------------------------
SELECT
  from_address     AS `from`,
  to_address       AS `to`,
  value,
  transaction_hash AS evt_tx_hash,
  block_timestamp
FROM `bigquery-public-data.crypto_ethereum.token_transfers`
WHERE token_address = '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984'
  AND block_timestamp >= TIMESTAMP('2026-03-26 00:00:00 UTC')
  AND block_timestamp <  TIMESTAMP('2026-06-25 00:00:00 UTC');


-- -----------------------------------------------------------------------------
-- 3. LINK 余额（全历史净流量累计，不限窗口）
--    单次扫描：UNNEST 展开收/发两侧，比 UNION ALL 省一半读取量
-- -----------------------------------------------------------------------------
SELECT
  addr AS address,
  SUM(delta) AS balance
FROM `bigquery-public-data.crypto_ethereum.token_transfers`,
UNNEST([
  STRUCT(to_address   AS addr,  CAST(value AS BIGNUMERIC) AS delta),
  STRUCT(from_address AS addr, -CAST(value AS BIGNUMERIC) AS delta)
])
WHERE token_address = '0x514910771af9ca656af840dff83e8264ecf986ca'
GROUP BY addr
HAVING balance > 0;


-- -----------------------------------------------------------------------------
-- 4. UNI 余额（与查询 3 逐字对应，仅合约地址不同）
-- -----------------------------------------------------------------------------
SELECT
  addr AS address,
  SUM(delta) AS balance
FROM `bigquery-public-data.crypto_ethereum.token_transfers`,
UNNEST([
  STRUCT(to_address   AS addr,  CAST(value AS BIGNUMERIC) AS delta),
  STRUCT(from_address AS addr, -CAST(value AS BIGNUMERIC) AS delta)
])
WHERE token_address = '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984'
GROUP BY addr
HAVING balance > 0;
