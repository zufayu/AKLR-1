# AKLR kernel optimizer

This pack runs one scheduled workflow that selects schemas, optimizes kernels with
KernelForge, validates the results, opens draft PRs, and follows their status.
It has one entry skill and four skills for the M1 responsibilities.

## Start here

The shared entry is [FOR_AGENTS.md](FOR_AGENTS.md). It routes setup and maintenance requests to
the setup guide, and runtime invocations to [kernel-automation](skills/kernel-automation/SKILL.md).
The runtime skill loads configuration and coordinates the four responsibilities for each round.

- [automation-prompt.md](templates/automation-prompt.md) is the prompt for the scheduler. Each trigger
  reads `FOR_AGENTS.md`; the workflow rules stay in the skills.
- [configuration.example.yaml](templates/configuration.example.yaml) is a starting configuration.
  Copy it to `.kernel-optimizer/configuration.yaml` and fill in the environment, model, budgets and
  permissions. Preserve existing configuration. This local file is ignored by Git and still holds no
  host names, usernames or personal absolute paths. The template itself is not used at runtime.

Saving these files does not create or enable a scheduled task. Follow the
[setup guide](skills/kernel-automation/references/setup.md) to configure the selected scheduler.

## Skills

| Skill | Role |
|---|---|
| [kernel-automation](skills/kernel-automation/SKILL.md) | Entry: load configuration, manage the round, and call the four skills. |
| [Monitor / Analyze](skills/kernel-automation-monitor/SKILL.md) | Read repository, GPU job, and PR facts; identify changes and blockers. |
| [Plan](skills/kernel-automation-plan/SKILL.md) | Select one action after checking candidates, budgets, and existing work. |
| [Execute](skills/kernel-automation-execute/SKILL.md) | Run an admitted action, validate a kernel, clean up resources, or deliver a draft PR. |
| [Knowledge](skills/kernel-automation-knowledge/SKILL.md) | Save state, operation records, evidence, and the report for the next round. |

Knowledge works throughout the round: load state first, save intent before an
external call, save its result, then report and release the round lock. Long jobs
return a handle so a later trigger can check and continue the same work.

These skills belong to one Automation and share one state directory. They reuse
the optimization, GPU, validation, and PR skills in `harness/skills/`.

## Files

```text
kernel-optimization/
├── FOR_AGENTS.md
├── README.md
├── templates/
│   ├── automation-prompt.md
│   └── configuration.example.yaml
└── skills/
    ├── kernel-automation/
    │   ├── SKILL.md
    │   └── references/setup.md
    ├── kernel-automation-monitor/SKILL.md
    ├── kernel-automation-plan/SKILL.md
    ├── kernel-automation-execute/SKILL.md
    └── kernel-automation-knowledge/
        ├── SKILL.md
        ├── references/state.md
        └── scripts/state.py
```

The automation reads skills by repository-relative paths in its configured checkout. These pack skills
do not need to be installed in the shared skill directory. User configuration stays
outside this pack, at `.kernel-optimizer/configuration.yaml`.

## Setup

1. Read `FOR_AGENTS.md` in this directory and follow its setup route.
2. Create or update local configuration. Verify the environment, initialize state only on first use,
   and test an `observe` round.
3. Make the pack and shared skills available in the exact checkout used by every invocation.
   A scheduler that creates fresh checkouts needs committed files on its selected revision; a scheduler
   using an existing checkout must record any local changes. Provision the ignored local configuration
   in that execution environment. Keep deployment details out of commits.
4. Use `templates/automation-prompt.md` in one scheduled task. The default interval is 15 minutes;
   supported repository events can use the same entry. Record the task identity and effective model.
5. The configuration defaults to `mode: run`; enable the task after readiness checks pass and spending is authorized.

`observe` only reads and reports. It starts no GPU jobs or inner model calls and
makes no GitHub changes. `run` can admit new work. `drain` finishes existing work
without admitting another campaign.

## Current limits

The pack needs one Linux host, a persistent state directory shared across
rounds, and access to GitHub and the GPU cluster. State must survive agent checkouts.

The state helper provides a round lock and atomic saves. Budget enforcement and
validation isolation still depend on the configured execution environment. The
trusted tool service is planned work.

Only one GPU campaign runs at a time. The automation tracks existing PRs;
maintainers update their code, review them, and merge them.
