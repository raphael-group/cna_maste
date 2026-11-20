#!/bin/bash
set -o pipefail

#
# run_cnamaste on the zenodo calicost sims, for a given sample and solver configuration.
# usage:  ./batch_zenodo_run.sh
#
#         uv run run_sim_qa --root /Users/mw9568/Work/ragr/sim/ --baseline_solver calicost --one_shot --overwrite

# NB Ctrl-C stops the batch
trap 'echo "Interrupted: stopping the batch." >&2; exit 130' INT TERM

#
#  ensure <numcnas3.3_cnasize5e7_ploidy2_random4> in zenodo_sample_sheet.tsv, or update.
# 

# ROOT="/u/mw9568/scratch/calicost_sims/"
# NB ROOT and USE_EXISTING default as shown and are read from the environment where set, as MAX_SAMPLE and QUIET are.
ROOT="${ROOT:-/Users/mw9568/Work/ragr/sim/}"

SEED=12345

RANDOM_SEEDS=(0) # NB only one is currently supported.
USE_EXISTING="${USE_EXISTING:-false}"
MAX_JOBS=1
MAX_SAMPLE="${MAX_SAMPLE:-}" # NB the first MAX_SAMPLE shuffled samples; every sample where unset or empty.
QUIET="${QUIET:-true}" # NB QUIET=true: each run_cnamaste's output goes to <its output dir>/run_<id>/cnamaste_batch_<id>.log;
QUIET_TAIL="${QUIET_TAIL:-20}"
# NB CONFIG names every run (run_<config.run_id>/) and its scoring; QA=true scores the study once the jobs end,
#    run_sim_qa --config per seed into ROOT/qa/<id>/, a run that cannot be scored named in its warning.
CONFIG="${CONFIG:-zenodo_sim_config.yaml}"
QA="${QA:-true}"
# NB CalicoST's copies QA scores its baseline on, cnv<_ploidy>_seglevel.tsv: run_sim_qa --calicost_ploidy.
CALICOST_PLOIDY="${CALICOST_PLOIDY:-diploid}"

# NB <label_solver>-<initializer>-<decoder>; tests/test_run_sim_qa_cycle.py runs each, `pytest -m end2end`.
#    expansion-lattice-lattice opengm_expansion-calicost-calicost, opengm_icm-lattice-lattice, opengm_icm-calicost-calicost
SOLVERS=(
    expansion-lattice-lattice
)

# NB the solver QA scores against CalicoST and against each other solver in SOLVERS, one figure each; QA=true requires
#    it among SOLVERS.
QA_SOLVER=expansion-lattice-lattice

VALID_LABEL_SOLVERS=(icm opengm_icm opengm_expansion opengm_trws expansion trws)
VALID_INITS=(lattice calicost gmm)
VALID_DECODERS=(lattice calicost)

# NB macOS's bash 3.2 has no associative arrays: a linear scan over the valid values.
in_list() {
    local needle=$1 item
    shift
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

for SOLVER in "${SOLVERS[@]}"; do
    IFS='-' read -r LABEL_SOLVER INIT DECODER EXTRA <<< "$SOLVER"
    if [[ -z "$LABEL_SOLVER" || -z "$INIT" || -z "$DECODER" || -n "$EXTRA" ]]; then
        echo "solver ${SOLVER} is not <label_solver>-<initializer>-<decoder>." >&2
        exit 1
    fi
    in_list "$LABEL_SOLVER" "${VALID_LABEL_SOLVERS[@]}" || {
        echo "solver ${SOLVER}: label solver ${LABEL_SOLVER} is not one of ${VALID_LABEL_SOLVERS[*]}." >&2
        exit 1
    }
    in_list "$INIT" "${VALID_INITS[@]}" || {
        echo "solver ${SOLVER}: initializer ${INIT} is not one of ${VALID_INITS[*]}." >&2
        exit 1
    }
    in_list "$DECODER" "${VALID_DECODERS[@]}" || {
        echo "solver ${SOLVER}: decoder ${DECODER} is not one of ${VALID_DECODERS[*]}." >&2
        exit 1
    }
done

if [[ "$QA" == "true" ]] && ! in_list "$QA_SOLVER" "${SOLVERS[@]}"; then
    echo "QA_SOLVER ${QA_SOLVER} is not one of SOLVERS (${SOLVERS[*]}): QA has nothing to score; set QA=false." >&2
    exit 1
fi

TEMPLATE_SHEET=./zenodo_sample_sheet.tsv
TEMPLATE_SAMPLE=numcnas3.3_cnasize5e7_ploidy2_random4
TEMPLATE_ID=Z001-U1

echo "Preparing workspace..."

mkdir -p zenodo_sample_sheets

SAMPLE_IDS=()
for d in "$ROOT"/"simulated_data_related"/*/; do
   SAMPLE_IDS+=("$(basename "$d")")
done

# NB random shuffling such that a diverse set "streams" to analysis.
SAMPLE_IDS=($(printf "%s\n" "${SAMPLE_IDS[@]}" | gshuf --random-source=<(yes $SEED)))

if [[ -n "$MAX_SAMPLE" ]]; then
    SAMPLE_IDS=("${SAMPLE_IDS[@]:0:$MAX_SAMPLE}")
fi

echo "Found ${#SAMPLE_IDS[@]} sample ids @ ${ROOT}"
echo "Running with ${#RANDOM_SEEDS[@]} random seeds: ${RANDOM_SEEDS[@]}"
echo "Running ${#SOLVERS[@]} solvers: ${SOLVERS[*]}"

# NB a sheet the substitution leaves unchanged would run the template's sample under another's name: stop instead.
for TOKEN in "$TEMPLATE_SAMPLE" "$TEMPLATE_ID"; do
    if ! grep -qF "$TOKEN" "$TEMPLATE_SHEET"; then
        echo "${TEMPLATE_SHEET} does not contain ${TOKEN}: update it, or TEMPLATE_SAMPLE / TEMPLATE_ID." >&2
        exit 1
    fi
done

for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
    SHEET="zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv"
    sed "s|${TEMPLATE_SAMPLE}|${SAMPLE_ID}|g; s|${TEMPLATE_ID}|${SAMPLE_ID}|g" "$TEMPLATE_SHEET" > "$SHEET"
    if [[ "$SAMPLE_ID" != "$TEMPLATE_SAMPLE" ]] && grep -qF -e "$TEMPLATE_SAMPLE" -e "$TEMPLATE_ID" "$SHEET"; then
        echo "${SHEET} still names the template sample after substitution." >&2
        exit 1
    fi
    if ! grep -qF "/${SAMPLE_ID}/" "$SHEET"; then
        echo "${SHEET} names no directory for ${SAMPLE_ID}." >&2
        exit 1
    fi
    # NB spaceranger_dir and snp_dir: the template's prefix is kept, so a sheet written for another ROOT stops here.
    while read -r DIR; do
        if [[ ! -d "$DIR" ]]; then
            echo "${SHEET} names ${DIR}, which does not exist." >&2
            exit 1
        fi
    done < <(awk -F'\t' 'NR > 1 {print $3; print $4}' "$SHEET")
done

echo "Sample sheets generated."

run_single_job() {
    local SAMPLE_ID=$1
    local RANDOM_SEED=$2
    local LABEL_SOLVER=$3
    local INIT=$4
    local DECODER=$5
    local ROOT=$6
    local USE_EXISTING=$7

    local SOLVER="${LABEL_SOLVER}-${INIT}-${DECODER}"
    local OUT_BASE="${ROOT}/nomixing_cnamaste_related/${SAMPLE_ID}/${SOLVER}"
    local DECODER_KEY

    case "$DECODER" in
        lattice) DECODER_KEY=hmm_ab ;;
        calicost) DECODER_KEY=calicost ;;
        *) echo "unknown decoder ${DECODER}" >&2; return 2 ;;
    esac

    echo "random_seed=${RANDOM_SEED}: solving to ${OUT_BASE}"

    # NB the run writes run_<config.run_id>/ under paths.output_dir: its config, tables, log and timings.
    mkdir -p "${OUT_BASE}"
    local command=(
        uv run run_cnamaste "$CONFIG"
        -o "paths.sample_sheet=zenodo_sample_sheets/zenodo_${SAMPLE_ID}_sheet.tsv"
        -o "paths.output_dir=${OUT_BASE}"
        -o "hmrf.random_seed=${RANDOM_SEED}"
        -o "hmrf.label_solver=${LABEL_SOLVER}"
        -o "hmm.initializer=${INIT}"
        -o "ab_decode.decoder=${DECODER_KEY}"
    )
    # NB run_cnamaste keeps the run its configuration names where its final fit is under 6 h old.
    if [[ "$USE_EXISTING" == "true" ]]; then
        command+=(--use_existing)
    fi
    # NB QUIET: everything the job prints, written beside the runs, then moved into the run's own directory by the id
    #    of the configuration the job wrote, config_<id>.yaml, the one newer than the job's start; a job stopped
    #    before writing one leaves it here.
    local log="${OUT_BASE}/cnamaste_batch.log.tmp"

    local rc
    if [[ "$QUIET" == "true" ]]; then
        local started
        started=$(mktemp "${OUT_BASE}/.cnamaste_batch_started.XXXXXX")
        "${command[@]}" > "$log" 2>&1
        rc=$?
        local written=()
        while IFS= read -r config; do
            written+=("$config")
        done < <(find "$OUT_BASE" -mindepth 2 -maxdepth 2 -path "*/run_*/config_*.yaml" -newer "$started")
        rm -f "$started"
        if [[ ${#written[@]} -eq 1 ]]; then
            local run_id
            run_id=$(basename "${written[0]}" .yaml)
            run_id=${run_id#config_}
            mv "$log" "${OUT_BASE}/run_${run_id}/cnamaste_batch_${run_id}.log"
            log="${OUT_BASE}/run_${run_id}/cnamaste_batch_${run_id}.log"
        fi
    else
        "${command[@]}"
        rc=$?
    fi

    if [[ $rc -ne 0 ]]; then
        echo "FAILED ${SAMPLE_ID} ${SOLVER} random_seed=${RANDOM_SEED} (rc=${rc})" >&2
        if [[ "$QUIET" == "true" ]]; then
            echo "  log: ${log}; its last ${QUIET_TAIL} lines:" >&2
            tail -n "$QUIET_TAIL" "$log" | sed 's/^/    /' >&2
        fi
        return $rc
    fi

    if [[ "$QUIET" == "true" ]]; then
        echo "ok ${SAMPLE_ID} ${SOLVER} random_seed=${RANDOM_SEED} (log: ${log})"
    fi
    return 0
}

# NB progress line
RULE="--------------------------------------------------------------"

run_logged() {
    local SAMPLE_ID=$1
    local SOLVER="${3}-${4}-${5}"
    local DONE=0

    # NB $PROGRESS is a temporary tally file the batch uses to count finished jobs. Each job appends one line
    if [[ -s "$PROGRESS" ]]; then
        DONE=$(wc -l < "$PROGRESS" | tr -d ' ')
    fi

    printf '%s\n%s %s %d%%\n%s\n' "$RULE" "$SAMPLE_ID" "$SOLVER" "$((100 * DONE / JOB_TOTAL))" "$RULE"

    run_single_job "$@"
    local rc=$?

    echo . >> "$PROGRESS"

    return $rc
}

# NB export function and variables so GNU parallel can access them inside subshells
export -f run_single_job run_logged
export ROOT USE_EXISTING RULE QUIET QUIET_TAIL CONFIG

JOBS=()
for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
    for RANDOM_SEED in "${RANDOM_SEEDS[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do
            IFS='-' read -r LABEL_SOLVER INIT DECODER <<< "$SOLVER"
            JOBS+=("$SAMPLE_ID $RANDOM_SEED $LABEL_SOLVER $INIT $DECODER")
        done
    done
done

echo "${#JOBS[@]} jobs."

JOB_TOTAL=${#JOBS[@]}
PROGRESS=$(mktemp)

export JOB_TOTAL PROGRESS

# NB each failed job as "<sample> <random state> <solver> <init> <decoder> (rc=<n>)", reported at the end.
FAILED=()

if [[ "$MAX_JOBS" -eq 1 ]]; then
    echo "Running sequentially (streaming output)..."

    # NB Python streams its output as it is printed
    export PYTHONUNBUFFERED=1

    for JOB in "${JOBS[@]}"; do
        read -r SAMPLE_ID RANDOM_SEED LABEL_SOLVER INIT DECODER <<< "$JOB"
        run_logged "$SAMPLE_ID" "$RANDOM_SEED" "$LABEL_SOLVER" "$INIT" "$DECODER" "$ROOT" "$USE_EXISTING"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            FAILED+=("${JOB} (rc=${rc})")
        fi
    done
else
    echo "Running in parallel (output buffered until job completion)..."

    # NB parallel's joblog records each job's exit value (column 7) and signal (8); the job's arguments follow
    #    `run_logged` in its command (9).
    JOBLOG=$(mktemp)
    
    printf "%s\n" "${JOBS[@]}" | parallel -j "$MAX_JOBS" --colsep ' ' --joblog "$JOBLOG" \
        run_logged {1} {2} {3} {4} {5} "$ROOT" "$USE_EXISTING"
    PARALLEL_RC=$?
    
    LOGGED=""
    while IFS=$'\t' read -r _ _ _ _ _ _ EXITVAL SIGNAL COMMAND; do
        read -r _ SAMPLE_ID RANDOM_SEED LABEL_SOLVER INIT DECODER _ <<< "$COMMAND"
        JOB="${SAMPLE_ID} ${RANDOM_SEED} ${LABEL_SOLVER} ${INIT} ${DECODER}"
        LOGGED+="${JOB}"$'\n'
        if [[ "$EXITVAL" != "0" || "$SIGNAL" != "0" ]]; then
            FAILED+=("${JOB} (rc=${EXITVAL}, signal=${SIGNAL})")
        fi
    done < <(tail -n +2 "$JOBLOG")
    rm -f "$JOBLOG"
    
    for JOB in "${JOBS[@]}"; do
        if ! grep -qxF -- "$JOB" <<< "$LOGGED"; then
            FAILED+=("${JOB} (never ran, parallel rc=${PARALLEL_RC})")
        fi
    done
fi

DONE=0
if [[ -s "$PROGRESS" ]]; then
    DONE=$(wc -l < "$PROGRESS" | tr -d ' ')
fi

printf '%s\n%d of %d runs complete %d%%\n%s\n' "$RULE" "$DONE" "$JOB_TOTAL" "$((100 * DONE / JOB_TOTAL))" "$RULE"

rm -f "$PROGRESS"

# NB per seed, ${#SOLVERS[@]} scorings: QA_SOLVER against CalicoST, then against each other solver in SOLVERS' order,
#    one figure each (run_sim_qa --solver); the table, every solver's rows, is rewritten alike by each.
QA_RC=0
if [[ "$QA" == "true" ]]; then
    for RANDOM_SEED in "${RANDOM_SEEDS[@]}"; do
        for BASELINE in calicost "${SOLVERS[@]}"; do
            [[ "$BASELINE" == "$QA_SOLVER" ]] && continue
            uv run run_sim_qa --root "$ROOT" --config "$CONFIG" -o "hmrf.random_seed=${RANDOM_SEED}" \
                --solver "$QA_SOLVER" --baseline_solver "$BASELINE" --calicost_ploidy "$CALICOST_PLOIDY" --one_shot \
                --overwrite || QA_RC=$?
        done
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "${#FAILED[@]} of ${#JOBS[@]} cna-maste jobs failed:" >&2
    printf "  %s\n" "${FAILED[@]}" >&2
    exit 1
fi

echo "All ${#JOBS[@]} cna-maste jobs completed."
exit $QA_RC
