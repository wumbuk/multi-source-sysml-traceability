#!/usr/bin/env bash
# Three datasets x three fresh Oracle runs; exactly one round per Slurm job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_DIR="$(pwd -P)"
ENV_FILE="${PROJECT_DIR}/deploy/baseline_server.env"
[[ -f "${ENV_FILE}" ]] || { echo "Missing server configuration: ${ENV_FILE}" >&2; exit 1; }
# shellcheck disable=SC1090
source "${ENV_FILE}"
[[ "${BASELINE_PROJECT_DIR:-}" == "${PROJECT_DIR}" ]] || { echo "BASELINE_PROJECT_DIR must be ${PROJECT_DIR}" >&2; exit 1; }
for tool in sbatch conda; do command -v "${tool}" >/dev/null || { echo "Missing command: ${tool}" >&2; exit 1; }; done
bash -n deploy/run_oracle_experiment.slurm
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${BASELINE_CONDA_ENV:?BASELINE_CONDA_ENV is required}"
RUN_NAME="${ORACLE_RUN_NAME:-oracle_type_three_rounds_$(date +%Y%m%d_%H%M%S)_$$}"
[[ "${RUN_NAME}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || { echo 'Unsafe ORACLE_RUN_NAME.' >&2; exit 1; }
SUITE="${PROJECT_DIR}/result/Qwen3.5-27B/${RUN_NAME}"
SUBMIT_ROOT="${PROJECT_DIR}/.server_submit/${RUN_NAME}"
[[ ! -e "${SUITE}" && ! -e "${SUBMIT_ROOT}" ]] || { echo 'Run already exists; choose a new ORACLE_RUN_NAME.' >&2; exit 1; }
python deploy/oracle_experiment.py prepare --project "${PROJECT_DIR}" --suite "${SUITE}"
umask 077
mkdir -p "${SUBMIT_ROOT}"
printf 'dataset\tround\tjob_id\tdependency\n' >"${SUITE}/jobs.tsv"
previous_job=""
trap 'echo "Submission interrupted/failed. Already submitted jobs are listed in ${SUITE}/jobs.tsv; inspect them before retrying." >&2' ERR
for dataset in Air_Condition Extract_Requirement_Values Architecting_Spacecraft; do
  for round in 1 2 3; do
    job_env="${SUBMIT_ROOT}/${dataset}_round_${round}.env"
    cp "${ENV_FILE}" "${job_env}"
    {
      printf '\nORACLE_SUITE_DIR=%q\n' "${SUITE}"
      printf 'ORACLE_DATASET=%q\n' "${dataset}"
      printf 'ORACLE_ROUND=%q\n' "${round}"
    } >>"${job_env}"
    options=(--parsable --export=ALL --job-name="oracle-${dataset}-r${round}")
    if [[ -n "${previous_job}" ]]; then options+=("--dependency=afterok:${previous_job}"); fi
    job_id="$(BASELINE_ENV_FILE="${job_env}" sbatch "${options[@]}" deploy/run_oracle_experiment.slurm)"
    job_id="${job_id%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch job ID: ${job_id}" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\n' "${dataset}" "${round}" "${job_id}" "${previous_job:--}" >>"${SUITE}/jobs.tsv"
    printf 'Submitted %s round_%s: job %s (afterok: %s)\n' "${dataset}" "${round}" "${job_id}" "${previous_job:--}"
    previous_job="${job_id}"
  done
done
trap - ERR
printf '\nSubmitted nine serial sbatch jobs.\nResults: %s\nJob chain: %s/jobs.tsv\n' "${SUITE}" "${SUITE}"
echo 'Final metrics.json / metrics.csv appear only after Architecting_Spacecraft round_3 succeeds.'
