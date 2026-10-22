# Retired origin-invariance demos

Reviewed and retired on 2026-10-21:

- `origin_invariant_cqed_rhf/CQED_RHF.py`
- `origin_invariant_cqed_rhf/CQED_TEST_OI.py`

Both files were fully tracked and last changed in commit `deadcc5` on
2023-03-06. They are standalone water/CQED-RHF demonstration scripts, not a
separate implementation of the mean-field solver. They import the active
`helper_cqed_rhf` module, use wildcard imports and fixed output filenames, and
predate the configuration/result interface. The second script performs an
origin-shift comparison; retiring it does not establish that the corresponding
invariance property is covered elsewhere.

A repository search found no callers or executable references to the directory
or script names. The only external textual reference was the repository overview
in `CLAUDE.md`, now updated. Neither script is included in the package module
list, source manifest, or pytest's default test discovery. They may have been
run manually; the audit establishes that they are unused by the active package
and its automated test/build paths, not that external users never ran them.
The obsolete demos were not executed as part of this audit.

Git history is the archive. Keeping another source folder would invite confusion
about which examples are maintained. Both files can be recovered from the exact
pre-removal revision `deaccd108b4819463710298ce466565af8e1d829`, for example:

```sh
git show deaccd108b4819463710298ce466565af8e1d829:origin_invariant_cqed_rhf/CQED_TEST_OI.py > /tmp/CQED_TEST_OI.py
```

This retirement changes no active solver or origin-handling code. Current
maintained examples are in [examples/canonical](../../examples/canonical/README.md).
