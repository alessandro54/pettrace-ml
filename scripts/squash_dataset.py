"""One-off: drop the pettrace-reid versions that named the pets (v0–v2) and squash the Hub history.

Deletes every tag, squashes `main` into one commit (old commits and their name folders become
unreachable) and re-creates `v3` on it. Needs a WRITE token: HF_TOKEN=<write token> python ...
"""

import os

from huggingface_hub import HfApi

REPO = "alessandro54/pettrace-reid"
api = HfApi(token=os.environ["HF_TOKEN"])
refs = api.list_repo_refs(REPO, repo_type="dataset")
# Tags are annotated: resolve v3 to its commit (target_commit is the tag object).
v3 = api.list_repo_commits(REPO, repo_type="dataset", revision="v3")[0].commit_id
head = api.list_repo_commits(REPO, repo_type="dataset")[0].commit_id
assert head == v3, f"main ({head[:8]}) moved past v3 ({v3[:8]}): publish nothing before squashing"
for t in refs.tags:
    api.delete_tag(REPO, tag=t.name, repo_type="dataset")
api.super_squash_history(REPO, repo_type="dataset", commit_message="pettrace-reid v3 (pseudonymous codes)")
new = api.list_repo_commits(REPO, repo_type="dataset")
api.create_tag(REPO, tag="v3", revision=new[0].commit_id, repo_type="dataset")
files = api.list_repo_files(REPO, repo_type="dataset", revision="v3")
print(f"commits on main: {len(new)}; tags: {[t.name for t in api.list_repo_refs(REPO, repo_type='dataset').tags]}")
print(f"v3 = {new[0].commit_id} ({sum(f.startswith('images/') for f in files)} photos)")
