#!/bin/bash
# Build the WRDS Cloud bundle (explanations/taq/pull.md, running on WRDS Cloud): the job script, the trading days, the job scripts made
# by plan_jobs.py, and the query and checks from src/dsr. Run from the repository: bash scripts/data_acquisition/wrds_cloud/build_bundle.sh
set -euo pipefail
B=data/wrds_cloud/bundle
rm -rf "$B" data/wrds_cloud/taq_cloud.tgz
mkdir -p "$B/dsr" "$B/jobs"
cp scripts/data_acquisition/wrds_cloud/taq_cloud_day.py scripts/data_acquisition/wrds_cloud/days.csv scripts/data_acquisition/wrds_cloud/pilot.sh "$B/"
cp scripts/data_acquisition/wrds_cloud/jobs/* "$B/jobs/"
cp src/dsr/__init__.py src/dsr/taq.py src/dsr/taq_sql.py "$B/dsr/"
tar czf data/wrds_cloud/taq_cloud.tgz -C "$B" .
echo "data/wrds_cloud/taq_cloud.tgz: $(tar tzf data/wrds_cloud/taq_cloud.tgz | grep -c -v '/$') files"
