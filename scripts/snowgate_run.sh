#!/bin/bash
# snowgate 제출 파일 재생성: v6(gfs60) 기본 예측 → 눈 모델 예측 → 게이트 결합(배정 gfs60_nolimit).
# 원작: 팀원의 적설 전용 모델(2026-11-08 제출 묶음). 이 저장소 구조에 맞춰 경로만 옮겼다.
#   OIBC_DATA_ROOT=/path/to/OIBC_2026_data scripts/snowgate_run.sh
# 필요: 특징 캐시(snow 묶음 포함), GFS 60열, $OIBC_DATA_ROOT/submissions/composition_oof_gfs60_nolimit.csv(src.search_oof)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OIBC_DATA_ROOT:?OIBC_DATA_ROOT를 지정한다}"
export OIBC_DATA_ROOT OIBC_CACHE_NAME=${OIBC_CACHE_NAME:-cache_v6snow}
P=${PYTHON:-.venv/bin/python}
COMMON="seeds=3 common=gfs60 weight=cap target=kt train_start=2021-01-01"
S=$OIBC_DATA_ROOT/submissions
[ -f "$OIBC_DATA_ROOT/$OIBC_CACHE_NAME/test_features.parquet" ] || $P -W ignore -m src.features
$P -W ignore -m src.make_submission v6_gfs60_ref_submit reference $COMMON feats=fullc
$P -W ignore -m src.make_submission v6_gfs60_snowacc_ref reference $COMMON feats=fullc+snow
$P -W ignore -m src.snowgate_submission v6_gfs60_snowgate_nolimit v6_gfs60_ref_submit v6_gfs60_snowacc_ref "$S/composition_oof_gfs60_nolimit.csv"
shasum -a 256 "$S"/v6_gfs60_ref_submit/*.csv "$S"/v6_gfs60_snowgate_nolimit/*.csv
