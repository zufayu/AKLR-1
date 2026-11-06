# Set up the AKLR kernel automation

Read `harness/automations/kernel-optimization/FOR_AGENTS.md` and its configuration template from the
repository root. Keep user configuration at `.kernel-optimizer/configuration.yaml`, outside the
managed pack and excluded from Git. Preserve existing values and user edits. This setup does not start a paid campaign.
Keep deployment paths, internal hostnames, account identities and credentials out of shared skills and templates.
The local configuration holds no host names, usernames or absolute personal paths either: write `state_dir`
with `$USER` or other environment variables, which the state helper expands.

## Prepare configuration

Infer the repository and branch from the target checkout; get the model,
node and model budgets, execution environment, and publication authority from existing user instructions
or configuration. Do not guess missing paid-run choices.
Resolve the campaign model to an identifier KernelForge's provider supports. Verify its `reasoning_effort`; do not invent
a CLI flag, silently drop the requested effort, or substitute another model when it is unsupported. An approved configuration supplies the budget
and model that forge-loop-campaign would otherwise ask for on every run.

`run` is the default; paid and publishing actions stay blocked until `readiness.md` and the budget authorization exist. Use `observe` for inspection only. `run` admits new work; `drain` finishes existing work without
new admissions. Missing credentials or required runtime values block the actions that need them, not
unrelated read-only inspection. Secrets belong in the execution environment's credential mechanism.

Check strategy types `manual` / `kernel-less`, positive batch/PR/cooldown limits, one supported backend,
and positive campaign limits within cumulative limits.
Total node time must cover setup and final validation as well as loop time. Model budget covers all inner
agent calls; outer automation and GPU CI expenses have separate authorizations. Do not implement hard caps by
estimating token use from logs. Verify the configured model provider's hard limit and scheduler time limit.

## Verify the actual execution environment

Run checks in the environment each Automation invocation will use, not merely in this setup session:

- It reads the operational skill from the configured revision and locally provisioned config by repository-relative path. Confirm all four
  responsibility skills linked by the entrypoint exist, along with their operational dependencies
  (campaign, remote-gpu, validation, PR, reporting and the chosen backend knowledge skill).
- It can authenticate to the configured GitHub repository and read main, PRs, CI and reviews. With publication
  enabled it can push a new branch and create draft PRs. CI label authority must be configured separately.
- It runs on the same host and reaches the **same persistent state directory** on every invocation.
  Use one dedicated identity and a local Linux filesystem with working `flock` and atomic rename.
  Do not use temporary storage, ephemeral agent checkouts, independent copies, or an unverified network filesystem.
- The existing remote-gpu CLI can read the configured gfx950 cluster. Follow remote-gpu for any setup changes.
  `forge_task` and KernelForge help commands resolve in the pinned runtime; no node allocation is needed just
  to prepare the automation pack. If runtime capability has not been tested, leave paid mode disabled.
- Establish a separate trusted validation checkout/environment: the optimization worker cannot edit its
  validator, overwrite final evidence or access GitHub/control-state credentials. Pin image/tool versions.
- Record the successful checks and the budget authorization in `readiness.md` inside the state directory. Validate that
  its config, model channel and image still match before turning on paid mode.

Read `harness/automations/kernel-optimization/skills/kernel-automation-knowledge/references/state.md`. Initialize a new state directory once with the
state helper. An existing or missing-after-use state directory requires investigation, not reinitialization.
A new directory must not hide campaigns already running under this automation's identity.

## Prepare one scheduled workflow

Use the owner's selected scheduler for this state directory. Inspect existing tasks first and update
the matching task instead of creating another. When moving between schedulers, disable the old trigger
before enabling the new one. Preserve the existing state and budgets.

Use `harness/automations/kernel-optimization/templates/automation-prompt.md` as the thin prompt. Each
trigger reads `harness/automations/kernel-optimization/FOR_AGENTS.md`, which routes runtime invocations
to the entry skill. Keep workflow rules in the skills rather than copying them into the saved prompt.

Bind the task to the intended checkout and execution session or environment. The scheduler owns the
outer provider, model and reasoning effort; verify its actual settings before enabling it. The campaign
model remains governed by `campaign`.

Make the pack, shared skills and state helper available in the execution checkout. If each invocation
creates a fresh checkout, the required files must be committed on its selected revision. If it uses an
existing checkout, record the revision and any uncommitted edits accurately. Provision the Git-ignored
configuration at the documented path and preserve access to the same persistent state directory.
Keep actual machine names, deployment paths, account details and authorization records out of commits.
Prepare edits first; do not commit unless requested.

Configure the name, repository, execution access, prompt and interval (15 minutes by default) using the
scheduler's supported interface. Supported repository events may use the same workflow; scheduled
polling remains the fallback. Do not depend on ordinary bot comments as triggers or invent an API.
If the interface is unavailable, hand over the prepared prompt, configuration location, readiness
results and remaining settings, and clearly report that the task has not been saved or enabled.

Report the saved task identity, execution binding, enabled status and returned next-run time.
Enable it only within the owner's authorization. If an initial run is requested, trigger it once after
saving. A queued or started invocation is not a completed validation; let that invocation execute and
inspect its result. Keep the scheduler service running for subsequent wakeups.

## Verify before paid operation

Run one `observe` invocation: it queries GitHub, inspects any existing runs and lists candidates, produces a
report, and makes no GPU/model/GitHub mutations. Do not require unrelated missing model values for this check.
Run the state helper's CPU tests, then prove two concurrent invocations on the configured host cannot both
acquire the round lock. Verify a second fresh invocation reads the state saved by the first.

After configuration and environment readiness pass, enable the existing automation in `run` mode within the
owner's recorded authority. First real acceptance is one bounded campaign through verified cleanup and draft
PR, followed by a fresh invocation observing that PR. Do not claim this acceptance from CPU tests or a prompt
preview. Report whether the automation is prepared, saved, enabled, or runtime-verified; distinguish them.
