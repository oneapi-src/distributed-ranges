.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

============
Contributing
============

Follow the standard github workflow. Fork this repo, clone your fork,
make changes, commit to a new branch, push branch to your fork. Submit
a PR from your fork.

The CI runs some formatting checks. It is easiest to resolve errors
with your local build before submitting the PR.

Run pre-commit checks::

  pre-commit run --all

Do pre-commit testing as part of commit::

  pre-commit install

``pre-commit`` will automatically fix most issues. Do a ``git add`` to
add the changes and run ``pre-commit`` or ``git commit`` again.

To fix a problem with a missing license do::

  reuse annotate --exclude-year --license BSD-3-Clause --copyright "Intel Corporation" <filename>
