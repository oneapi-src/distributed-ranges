.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

============
 Pre-commit
============

The ``checks`` job in CI runs some static tests with the
``pre-commit`` python package. You can resolve issues faster by
running the checks locally before submitting the PR.

Run pre-commit checks::

  pre-commit run --all

Do pre-commit testing as part of git commit::

  pre-commit install

``pre-commit`` checks links in the documentation. If it is taking a
long time, you can skip all checks as part of commit with ``-n``::

  git commit -n -m 'Commit message'

``pre-commit`` will automatically fix most issues. Do a ``git add`` to
add the changes and run ``pre-commit`` or ``git commit`` again.

To skip the sphinx tests::

  SKIP=sphinx pre-commit run --all

This may be convenient if you are not changing the documentation. The
Sphinx tests require installing some prerequisites, a working internet
connection, and take longer to run.

Reuse
=====

To fix a problem with a missing license do::

  reuse annotate --exclude-year --license BSD-3-Clause --copyright "Intel Corporation" <filename>

Or copy the license from the top of a file with the same filename
extension.
