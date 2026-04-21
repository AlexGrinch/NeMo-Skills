# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Formatted translation pipeline.

Stages:
  1. make_concise      – keep only fields_to_consider from each JSON object
  2. wrap              – add source_lang / target_lang, serialize concise JSON as `src`
  3. generate          – run LLM to produce a formatted translation
  4. unwrap            – extract the translated JSON from the model's generation field
  5. filter_and_merge  – validate with format checker; on pass, replace fields_to_translate
                         in the original object with translated values
  6. curate            – placeholder for downstream data curation steps

Standalone scripts for each stage (except generate) live under:
  recipes/translation/scripts/formatted-translation/

Required top-level config keys:
  base_output_dir:      root directory for all stage outputs
  cluster:              cluster name (passed to pipeline utilities)
  expname:              base experiment name
  fields_to_translate:  list of top-level fields that must be translated
  fields_to_consider:   (optional) fields shown to the model; defaults to fields_to_translate
  input_file:           path to the original JSONL
  target_lang:          target language name (e.g. "German")
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate as generate_fn
from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

# Absolute path to the scripts directory (as seen inside the container/job)
_SCRIPTS_DIR = "/nemo_run/code/recipes/translation/scripts/formatted-translation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stage_expname(base_expname: str, stage_name: str, suffix: str) -> str:
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


def _stage_dir(base_output_dir: str, stage_name: str, stage_config: dict) -> str:
    return stage_config.get("output_dir", f"{base_output_dir}/{stage_name}")


# ---------------------------------------------------------------------------
# Stage: make_concise
# ---------------------------------------------------------------------------


def make_concise(cluster, expname, run_after, stage_config, **kwargs):
    """Keep only fields_to_consider from each JSON object."""
    base_output_dir = kwargs["base_output_dir"]
    fields_to_consider = kwargs["fields_to_consider"]

    input_file = stage_config.get("input_file", kwargs.get("input_file"))
    output_dir = _stage_dir(base_output_dir, "make_concise", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/concise.jsonl")

    fields_arg = " ".join(fields_to_consider)
    cmd = (
        f"python {_SCRIPTS_DIR}/make_concise.py "
        f"    --input {input_file} "
        f"    --output {output_file} "
        f"    --fields {fields_arg} "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: wrap
# ---------------------------------------------------------------------------


def wrap(cluster, expname, run_after, stage_config, **kwargs):
    """Wrap each concise JSON object as {source_lang, target_lang, src: <json_string>}."""
    base_output_dir = kwargs["base_output_dir"]

    input_file = stage_config.get("input_file", f"{base_output_dir}/make_concise/concise.jsonl")
    target_lang = stage_config.get("target_lang", kwargs.get("target_lang"))
    output_dir = _stage_dir(base_output_dir, "wrap", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/wrapped.jsonl")

    cmd = (
        f"python {_SCRIPTS_DIR}/wrap_sft_data.py "
        f"    --input {input_file} "
        f"    --output {output_file} "
        f"    --target-lang '{target_lang}' "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: generate
# ---------------------------------------------------------------------------


def generate(cluster, expname, run_after, stage_config, **kwargs):
    """Run LLM translation with a format-specific prompt config."""
    base_output_dir = kwargs["base_output_dir"]

    input_file = stage_config.get("input_file", f"{base_output_dir}/wrap/wrapped.jsonl")
    output_dir = _stage_dir(base_output_dir, "generate", stage_config)
    format_config = stage_config["format_config"]

    generate_fn(
        ctx=wrap_arguments(f"++prompt_config={format_config} {stage_config.get('inline_args', '')} "),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        expname=expname,
        run_after=run_after,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: mock_generate  (testing only — replaces generate for local runs)
# ---------------------------------------------------------------------------


def mock_generate(cluster, expname, run_after, stage_config, **kwargs):
    """Mock generation for local testing. Writes to the same dir as generate."""
    base_output_dir = kwargs["base_output_dir"]

    input_file = stage_config.get("input_file", f"{base_output_dir}/wrap/wrapped.jsonl")
    # Write to generate/ so downstream stages need no config changes.
    output_dir = stage_config.get("output_dir", f"{base_output_dir}/generate")
    output_file = stage_config.get("output_file", f"{output_dir}/output.jsonl")

    cmd = f"python {_SCRIPTS_DIR}/mock_generate.py     --input {input_file}     --output {output_file} "
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: unwrap
# ---------------------------------------------------------------------------


def unwrap(cluster, expname, run_after, stage_config, **kwargs):
    """Extract the translated JSON object from the model's generation field."""
    base_output_dir = kwargs["base_output_dir"]

    input_file = stage_config.get("input_file", f"{base_output_dir}/generate/output.jsonl")
    output_dir = _stage_dir(base_output_dir, "unwrap", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/unwrapped.jsonl")

    verbose_flag = "--verbose" if stage_config.get("verbose", False) else ""
    cmd = f"python {_SCRIPTS_DIR}/unwrap.py     --input {input_file}     --output {output_file}     {verbose_flag} "
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: filter_and_merge
# ---------------------------------------------------------------------------


def filter_and_merge(cluster, expname, run_after, stage_config, **kwargs):
    """Validate each translated output; on pass, merge translated fields into the original."""
    base_output_dir = kwargs["base_output_dir"]
    fields_to_translate = kwargs["fields_to_translate"]

    original_file = stage_config.get("original_file", kwargs.get("input_file"))
    generation_file = stage_config.get("generation_file", f"{base_output_dir}/generate/output.jsonl")
    checker_name = stage_config["checker"]
    output_dir = _stage_dir(base_output_dir, "filter_and_merge", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/translated.jsonl")

    fields_arg = " ".join(fields_to_translate)
    on_fail = stage_config.get("on_fail", "keep")
    verbose_flag = "--verbose" if stage_config.get("verbose", False) else ""

    cmd = (
        f"python {_SCRIPTS_DIR}/filter_and_merge.py "
        f"    --checker {checker_name} "
        f"    --original-file {original_file} "
        f"    --generation-file {generation_file} "
        f"    --fields-to-translate {fields_arg} "
        f"    --output {output_file} "
        f"    --on-fail {on_fail} "
        f"    {verbose_flag} "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: curate
# ---------------------------------------------------------------------------


def curate(cluster, expname, run_after, stage_config, **kwargs):
    """Placeholder for downstream data curation steps."""
    raise NotImplementedError(
        "The 'curate' stage is not yet implemented. "
        "Remove it from 'pipeline_stages' in your config to run the pipeline without it."
    )


# ---------------------------------------------------------------------------
# Stage registry & entrypoint
# ---------------------------------------------------------------------------

stages_map = {
    "make_concise": make_concise,
    "wrap": wrap,
    "generate": generate,
    "mock_generate": mock_generate,
    "unwrap": unwrap,
    "filter_and_merge": filter_and_merge,
    "curate": curate,
}


def get_available_configs(config_dir):
    config_dir = Path(config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        return []
    return [f.stem for f in config_dir.glob("*.yaml")]


if __name__ == "__main__":
    config_dir = Path(__file__).parents[1] / "config" / "pipeline"
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(description="Formatted translation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=available_configs,
        help="Name of a YAML config file under config/pipeline/",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to run. Defaults to all stages in config.",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default=None,
        help="Override base_output_dir from the config file.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Override input_file from the config file.",
    )
    args = parser.parse_args()

    config_path = config_dir / f"{args.config}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    if "pipeline_stages" not in config or not config["pipeline_stages"]:
        raise ValueError(f"{config_path} must define a non-empty 'pipeline_stages' list.")
    full_stage_sequence = config["pipeline_stages"]

    stages_to_run = args.stages.split(",") if args.stages else full_stage_sequence
    print(f"Running stages: {stages_to_run}")

    for stage in stages_to_run:
        if stage not in stages_map:
            raise ValueError(f"Unknown stage '{stage}'. Available: {list(stages_map.keys())}")
        if stage not in full_stage_sequence:
            raise ValueError(f"Stage '{stage}' is not in the config's pipeline_stages.")

    base_output_dir = args.base_output_dir or config["base_output_dir"]
    if args.input_file:
        config["input_file"] = args.input_file
    suffix = config.get("suffix", args.config)
    cluster = config["cluster"]
    expname_base = config["expname"]
    fields_to_translate = config["fields_to_translate"]
    fields_to_consider = config.get("fields_to_consider", fields_to_translate)
    input_file = config.get("input_file")
    target_lang = config.get("target_lang")

    for stage in stages_to_run:
        print(f"\n--- Running stage: {stage} ---")
        stage_config = config.get("stages", {}).get(stage, {})
        current_expname = _stage_expname(expname_base, stage, suffix)

        dep_stages = stage_config.get("dependencies")
        if dep_stages is not None:
            dependencies = [_stage_expname(expname_base, d, suffix) for d in dep_stages]
        else:
            dependencies = config.get("initial_dependency", None)

        stages_map[stage](
            cluster=cluster,
            expname=current_expname,
            run_after=dependencies,
            stage_config=stage_config,
            base_output_dir=base_output_dir,
            fields_to_translate=fields_to_translate,
            fields_to_consider=fields_to_consider,
            input_file=input_file,
            target_lang=target_lang,
        )

    print("\n--- Pipeline finished. ---")
