BOTS = {
    # "-suffix on a random stem" cluster: throwaway repos, own-repo only
    "hkustnlp-cmyk",
    "chengzeliu79-spec",
    "riverhe459-lgtm",
    "vp2sfwntbg-cmd",
    "lyc228612-sudo",
    "nyleeaa0503-nv",
    "s741dev",
    # agent / tool-use benchmark harnesses
    "mcptest-user",
    "tooluse222",
    "tooooolathon",
    "ps-toolathlon",
    "Toolathlon-beta",
    "jsouramid",          # same toolathlon harness; 13 repos created in 1s
}

# Automation that is openly infrastructure rather than disguised.
BENIGN = {
    # AWS Amplify regional fleet: one account per region, ~100 creates and
    # ~85 deletes each, running the full day
    "aws-aemilia-arn", "aws-aemilia-bom", "aws-aemilia-cdg", "aws-aemilia-cmh",
    "aws-aemilia-dub", "aws-aemilia-fra", "aws-aemilia-gru", "aws-aemilia-iad",
    "aws-aemilia-icn", "aws-aemilia-kix", "aws-aemilia-lhr", "aws-aemilia-mxp",
    "aws-aemilia-nrt", "aws-aemilia-pdx", "aws-aemilia-sfo", "aws-aemilia-sin",
    "aws-aemilia-syd", "aws-aemilia-yul",
    "jahia-ci",
    "openshift-helm-charts-bot",
    "elasticsearchmachine",   # posts CI build-scan failures
    "yugabyte-ci",            # posts Jira-linked DocDB failures
    # numbered CI families, one account per worker
    "scalr-autotester0", "scalr-autotester5", "scalr-autotester6",
    "scalr-autotester13", "scalr-autotester19",
    "mobbcitestjob5", "mobbcitestjob6", "mobbcitestjob7", "mobbcitestjob8",
    "mobbcitestjob9",
}

ALL_BOTS = BOTS | BENIGN


def is_bot(login: str) -> bool:
    """Whether this account is known automation.

    Verified accounts first, then the ``[bot]`` suffix, which is what that
    suffix means on GitHub.
    """
    return login in ALL_BOTS or login.endswith("[bot]")


def is_benign(login: str) -> bool:
    """Automation that is openly infrastructure rather than disguised."""
    return login in BENIGN


def labelled() -> set[str]:
    """Hand-verified accounts, excluding the ones the suffix already names."""
    return set(ALL_BOTS)

HUMANS = {
    "teidesu",            # "can't select videos, images, gifs - any attachment"
    "dvianabarbosa",      # "I was passing the entire book ( Clean Arquitecure)"
    "newtonick",          # seedsigner 0.7.0 on a pi0, wrong address prefix
    "eltmon",             # root-causing wedges against internal ticket ids
    "clement-cunin",      # French bug report, 401 after a hash rotation
    "Xboooo",             # Chinese report with a screenshot
    "dennougorilla",      # cites src/features/editor/index.js:435
    "MohabMohie",         # follow-up to a specific comment on #3674
    "theDawckta",         # detailed creative spec for a 3D model
    "ZarK",               # four coupled failures found provisioning a machine
    "bamiyanapp",         # Japanese feature request citing quizRoomHandler.js
    "cmengu",             # "Child of #49 Blocked by #50, #51, #53, #55"
    "KyleMit",            # argues for a recolor edit over regeneration
    "philliphoff",        # audit of a rendering pipeline
    "malaquiasdev",       # "Implements Option B from the discovery"
    "leadmee",            # names the untested functions individually
    "NotoriousRebel",     # migration plan for an aiohttp source
    "dorukardahan",       # missing tests for a named hook adapter
    "shresthamishra76",   # N-D array over a row-major float32 buffer
    "JoNil-Botta",        # missing stdlib math functions, deferred per README
    # second reading pass
    "TeamDman",           # "i'm trying to make challenge pack where players s"
    "arcaputo3",          # "Field report (FinAgent QA vs 0.13.0) - verified on 0.14.0"
    "ChayimFriedman2",    # rust panic with a minimal repro
    "kilasuit",           # -Environment quietly ignored with -UseNewEnvironment
    "jacobbpp",           # "still have hardcoded '20' after #19's fix"
    "StephanSchmidt",     # InstallHint misdetects go install on Windows
    "esoinila",           # quotes a playtester on mouse wheels
    "Willive",            # "None included - happy to share workflow run IDs"
    "kartikgola",         # metadata.yaml describes relation as oid
    "mrofisr",            # opencode2 does not load plugins from config
    "dmmulroy",           # throwaway client against a seeded project
    "sidd190",            # IsSecurityTeam is org-agnostic
    "willgriffin",        # recovering an interrupted 0.40.6 publish
    "Hmbown",             # subagent markup leaking into resumed sessions
    "inureyes",           # reject requests with no effective input
    "h-tacayama",         # split out from #220, C# reserved words
    "tutkli",             # selection activates from any bubbled click
    "Spissable",          # metric sql references a missing column
    "ChrisTisdale",       # export configuration to stdout
    "XsQuare01",          # Korean typography on small metadata labels
    "daiki-beppu",        # Japanese, competing prompt instructions
    "marcosgvieira",      # dmn-js shared component
    "gilad-rubin",        # enforce the locked #195 contract
    "michael-conrad",     # remove a dead fts_vector check
    "alltomatos",         # Portuguese, contacts.* capability
    "realjarvisma",       # maintainer note on ESC protocol fields
    "bipinrajbhar-rh",    # where can allowlist evaluation happen
    "eminogrande",        # derive and validate passkey algorithms
    "danicallero",        # make the application state machine visible
    "Gustavo-Harnisch",   # define goals, streaks, completion semantics
    "anandaroop",         # cloudfront invalidateSlug wrapper
    "wangleiphy",         # sim-to-real for quantum gates
    "Damola-Sodiq",       # OOM during Soroban node testing
    "hshimizu",           # follow-ups banked from #85
    "hassanhabib",        # BROKERS: Run Tool contract
    "yksalun",            # Chinese launch post for a browser tool
    "AkihikoWatanabe",    # paper-notes repo, title "あ" and an arxiv link
    "allanknecht",        # "Tipo da Licença", terse and empty-bodied
    "edburns",            # FastAPI scaffolding task
    "RoshanGH",           # Chinese PRD alignment, per-control
    "bennetwi92",         # umbrella epic for Claude-in-CI
    "pranayj78",          # PTMS-005, a bare list of enums
    "YajurvaMaharana",    # student sprint board, "Assignee: Member B"
}

def validate(known: set[str]) -> list[str]:
    """Labels that match no account in the loaded data.

    A verdict on an absent login scores nothing and reports nothing, so it
    fails silently. This happens for real reasons rather than typos alone:
    dropping PushEvent removes push-only accounts from the graph entirely,
    which is how ``s741dev`` -- verified automation -- stopped existing when
    the loader began filtering pushes.
    """
    return sorted(x for x in ALL_BOTS if x not in known)
