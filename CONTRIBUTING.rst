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

Editing the Spec
================

Build the spec::

  make -C doc/spec html

Open in your browser: ``doc/spec/build/html/index.html``

Doxygen html is at: ``doc/spec/build/doxygen-html/index.html``

There are pre-commit checks for spelling and broken links. To run it manually::

  make -C doc/spec linkcheck
  make -C doc/spec spelling SPHINXOPTS=-q

The `SPHINXOPTS` is necessary to disable warning as errors, so you can
see all the spelling errors instead of the first one. Add spelling
exceptions to `spelling_wordlist.txt`. Do not add variable, class,
function, etc to the exceptions. Spellcheck ignores them if they are
properly delimited in the source doc.
