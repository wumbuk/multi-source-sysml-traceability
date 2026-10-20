#!/usr/bin/env bash
# Full pipeline: three datasets, two fresh rounds per dataset, serial sbatch jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_DIR="$(pwd -P)"
ENV_FILE="${PROJECT_DIR}/deploy/baseline_server.env"
[[ -f "${ENV_FILE}" ]] || { echo "Missing configuration: ${ENV_FILE}" >&2; exit 1; }
source "${ENV_FILE}"
[[ "${BASELINE_PROJECT_DIR:-}" == "${PROJECT_DIR}" ]] || { echo "BASELINE_PROJECT_DIR must match ${PROJECT_DIR}" >&2; exit 1; }
RUN_NAME="${PIPELINE_RUN_NAME:-typed_two_rounds_$(date +%Y%m%d_%H%M%S)_$$}"
[[ "${RUN_NAME}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || { echo 'Unsafe PIPELINE_RUN_NAME.' >&2; exit 1; }
OUTPUT_ROOT="${PROJECT_DIR}/result/Qwen3.5-27B/${RUN_NAME}"
SUBMIT_ROOT="${PROJECT_DIR}/.server_submit/${RUN_NAME}"
WORK_ROOT="${PROJECT_DIR}/.server_runs/${RUN_NAME}"
[[ ! -e "${OUTPUT_ROOT}" && ! -e "${SUBMIT_ROOT}" && ! -e "${WORK_ROOT}" ]] || { echo 'Run already exists.' >&2; exit 1; }
command -v sbatch >/dev/null
bash -n deploy/run_typed_all_datasets.slurm
for dataset in Air_Condition Extract_Requirement_Values Architecting_Spacecraft; do
  [[ -f "Data/${dataset}/Judge.csv" ]] || { echo "Missing dataset: ${dataset}" >&2; exit 1; }
done
umask 077
mkdir -p "${OUTPUT_ROOT}" "${SUBMIT_ROOT}"
printf 'dataset\tround\tjob_id\tdependency\n' >"${OUTPUT_ROOT}/jobs.tsv"
previous_job=""
trap 'echo "Submission failed. Inspect ${OUTPUT_ROOT}/jobs.tsv for already submitted jobs before retrying." >&2' ERR
for dataset in Air_Condition Extract_Requirement_Values Architecting_Spacecraft; do
  for round in 1 2; do
    job_env="${SUBMIT_ROOT}/${dataset}_round_${round}.env"
    cp "${ENV_FILE}" "${job_env}"
    {
      printf '\nTYPED_RUN_NAME=%q\n' "${RUN_NAME}"
      printf 'TYPED_SINGLE_DATASET=%q\n' "${dataset}"
      printf 'TYPED_SINGLE_ROUND=%q\n' "${round}"
    } >>"${job_env}"
    options=(--parsable --export=ALL --job-name="lissa-${dataset}-r${round}")
    if [[ -n "${previous_job}" ]]; then options+=("--dependency=afterok:${previous_job}"); fi
    job_id="$(BASELINE_ENV_FILE="${job_env}" sbatch "${options[@]}" deploy/run_typed_all_datasets.slurm)"
    job_id="${job_id%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Invalid job ID: ${job_id}" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\n' "${dataset}" "${round}" "${job_id}" "${previous_job:--}" >>"${OUTPUT_ROOT}/jobs.tsv"
    printf 'Submitted %s round_%s: job %s (afterok: %s)\n' "${dataset}" "${round}" "${job_id}" "${previous_job:--}"
    previous_job="${job_id}"
  done
done
trap - ERR
printf '\nSix serial jobs submitted. Results: %s\n' "${OUTPUT_ROOT}"
