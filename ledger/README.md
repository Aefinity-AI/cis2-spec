# Experiment fingerprint ledger

This page is a public list of fingerprints of the records in our private experiment register. A fingerprint is a
short code that stands in for a record without revealing it. Lines marked backfill were published after the
records they cover already existed. For a line marked backfill, the fingerprint shows that the record existed by
the date the line was published here, and nothing more. It does not show when the record was really written.
Today 161 of the 193 lines on this page are marked backfill.

The original times are kept in our private history. We can share them with a client under a confidentiality
agreement, which is a signed promise to keep what we share private.

The lines are chained. Each line carries the fingerprint of the line before it. A line that is changed, removed
or moved in the middle of the list no longer fits its neighbour, and the check program reports it.

## What this page is

Aefinity AI keeps a private register of its experiments. Before an experiment runs, we write down a pass mark,
which is the result the experiment must reach to count as a success. After the run, we write down a verdict.

This page lets anyone check later that the fingerprint of a pass mark was public before the fingerprint of its
verdict was added, and that a published line was not rewritten afterwards. It does this without telling you what
was tested or what was found. The limits are listed below, and they matter.

## What is on this page

* `LEDGER.tsv` has one line for each record in the private register. A line has five parts, separated by a tab
  (the wide gap character used in plain text tables):
  1. a running line number
  2. an experiment number (E-0001, E-0002 and so on, in the order the experiments were registered)
  3. a kind
  4. the fingerprint of the record
  5. the fingerprint of the previous line
* The kind says what the record is.
  * `bar`: the pass mark was written down.
  * `bar-amended`: the pass mark was changed before the run began. The earlier version stays on the list.
  * `start`: the run began.
  * `verdict`: a result was recorded.
  * `start-unanchored`: the run began without its pass mark being public beforehand. We mark this exception openly.
  * A kind that ends in `-backfill` means the line was published after its record already existed, as the opening
    paragraph explains.
* The fingerprint of a record is a sha256 hash of the record's text joined to a salt. A sha256 hash is a standard
  way to turn any text into a short code. The salt is a secret random number, 32 bytes long (256 random bits),
  and every record has its own. Nobody can work back from the fingerprint to the record. Nobody can confirm a
  guess either, because the salt is unknown.
* `verify_ledger.py` is a short program that checks the page. It needs Python 3 and nothing else installed, and
  it needs no network.

## What this can show

* Order of publication. If the `bar` line of an experiment was added to this page on an earlier date than its
  `verdict` line, the pass mark was public before the verdict line was added.
* No quiet rewriting. A line that was published earlier cannot be changed, removed or moved later without
  breaking the chain, for anyone who compares the head hash with a saved copy. The head hash and the end of the
  file are explained in the next section.
* No gaps in the middle. The experiment numbers run without a gap and every line is chained to the one before
  it. A removed or reordered line breaks the chain, and `verify_ledger.py` says so.
* A revealed record is the real one. With a record and its salt, which we share under a confidentiality
  agreement, you can check that the record matches its fingerprint exactly. A record altered by one character no
  longer matches.

## What this cannot show

* It cannot show when an experiment ran. A lab could run an experiment in private, look at the result, and then
  publish a pass mark. Nothing on this page dates the run. The `start` line is our own statement. Our rule from
  the day of publication on is that a run does not begin until its `bar` line is public here, and an exception
  is marked `start-unanchored`. That rule is a promise. This page cannot check it.
* It cannot show any order for lines marked backfill. A backfill line was published after its record existed, so
  its date says nothing about when the record was written. Where the `verdict` line of an experiment is marked
  backfill, this page does not show that the pass mark came before the result. That evidence is in our private
  history.
* It cannot show that every experiment that was run got a verdict line. An experiment with no verdict line may be
  unfinished or dropped, and the page does not say which.
* It cannot show that every experiment was registered here. An experiment that was never listed leaves no trace.
  The numbering shows that no experiment was removed from the middle of the list. Removal at the end is covered
  under the head hash below. The same idea could also be registered again
  under a new number, and the page would not show that the two are related.
* It cannot show that an experiment was run honestly, that the pass mark was a sensible one, or anything about the
  result. Those need the record itself and the raw files, which we share under a confidentiality agreement.
* It cannot show how an experiment turned out. The verdict value is not on this page.
* It cannot protect the end of the file by itself. The head hash is the fingerprint of the last line, and
  `verify_ledger.py` prints it. Save it when you look at this page. Without a saved head hash, these changes would
  still pass the chain check: a changed last line, an extra line at the end, lines cut off the end, or a file
  rewritten from some line onward with a new chain. Compare the head hash with the copy you saved earlier, or
  look at the older versions in the history of the page.

## Where the dates come from

We do not write dates into the ledger ourselves. The commit dates in a git history (the log of changes kept by
the public repository) are set by whoever makes the commit, so do not rely on them alone. Rely on the time that
GitHub records for each push and each merged pull request, shown on the Activity page of the public repository
that hosts this page. This page ships without an OpenTimestamps proof for now, so that record is the whole of the
dating evidence.

## How a client under a confidentiality agreement checks one record

1. We send you the record as a JSON file (a plain text file of labelled fields), its salt as 64 hex characters
   (the digits 0 to 9 and the letters a to f), and the experiment number.
2. You download `LEDGER.tsv` and `verify_ledger.py` from this page.
3. You run `python3 verify_ledger.py LEDGER.tsv --record record.json --salt <the salt>`.
4. The program checks the chain. Then it prints the line, the experiment and the kind that your record matches.
   If the salt is wrong or the record was altered by even one character, it prints FAIL.

Every record has its own random salt, so one revealed record tells you nothing about the others.

## What this page reveals

At the time of this export the ledger holds 193 lines. They cover 67 registered experiments, and
55 of them have a verdict line. 161 of the 193 lines are marked backfill.

The page also shows which kinds of line each experiment number has. So it shows which experiments had their pass
mark amended before the run, which have not started, which are running and which have a verdict. The order of
the lines, and the times at which lines are added from now on, show when we work and roughly how long an
experiment takes. Apart from these counts and line kinds, the page shows nothing about what was tested or what
was found.

## Acknowledgments

A special thank you to Charles Seaman and Linda Blanchard, whose contributions have helped Aefinity AI stay on track.

And a very special thank you to **Bonnie Rae Power**: an amazing woman, a great friend and neighbor, without whom Aefinity AI would have never had a chance to ever get started. Thank you, Bonnie, for your advice, care, encouragement, guidance, intuitive wisdom, and financial assistance.
