# How ODyn uses its database

This page explains how ODyn reads and writes its database: the lock, the
caches, the call logs and the migrations, and how programs pass messages
through small files on the share (planned). It also lists what the design
assumes about the lab that uses it. If your lab works differently, the
section [When the assumptions change](#when-the-assumptions-change) says
what would have to change.

## The setting

ODyn keeps one SQLite file per project on the lab's network share, next to
the data. Three kinds of programs use it at the same time:

- **The app** on workstations (Windows, macOS): adding experiments, quality
  control, picking and approving motion corrections.
- **Notebooks and scripts** on workstations: analysis.
- **Cluster jobs** (Linux): motion corrections, reports, and a periodic
  "tick" that submits jobs and records how they ended.

```mermaid
flowchart LR
    subgraph share["Network share"]
        db[("project.db")]
        lock["project.db.lock"]
        logs[".odyn/logs/project/<br/>one file per call"]
        data["experiment folders<br/>(TIFFs, processed/)"]
    end
    app["App<br/>(workstations)"] --> share
    nb["Notebooks, scripts<br/>(workstations)"] --> share
    jobs["Cluster jobs<br/>(Linux nodes)"] --> share
```

Every file operation on the share is a network round trip: a few
milliseconds on the lab network, several times that over a VPN. Bandwidth
is rarely the limit. So the design counts round trips first and bytes
second.

## Layers

```mermaid
flowchart TB
    code["Code: db.experiments, group.run_motion_correction(...)"]
    cache["Table caches<br/>(one DataFrame per table, per Database object)"]
    lock["file_lock(database)<br/>(lock file beside the database)"]
    con["sqlite3 connection<br/>(one per Database object)"]
    file[("SQLite file on the share")]
    code --> cache --> lock --> con --> file
```

A `Database` object belongs to one thread: SQLite connections cannot be
shared between threads. The app runs all database work on one worker
thread, with a `Database` of its own. A `Group` reads through its
`Database`, with caches of its own for its rows.

## The lock

SQLite has its own locks, but they are not trusted here. A workstation and
a cluster node reach the share through different protocols and operating
systems, and do not see each other's SQLite locks. A reader that finds a
journal and no lock "recovers" the journal into the file, while the writer
is still writing. That corrupts the database.

So ODyn adds a lock of its own (`odyn/locking.py`): a file
`<database>.lock` beside the database. Creating a file that does not exist
yet (`O_CREAT | O_EXCL`) is a single step that every client sees the same
way. The lock is held for **every** use of the connection, reads included.

```mermaid
sequenceDiagram
    participant P as Process
    participant S as Share
    P->>S: create project.db.lock (exclusive)
    alt created
        P->>S: write holder (user, host, pid, time)
        P->>S: SQLite reads or writes
        P->>S: read lock back (still ours?)
        P->>S: delete project.db.lock
    else exists
        P->>S: read holder
        alt older than 120 s
            P->>S: rename it away (only one taker wins), retry
        else
            P->>P: wait 10 to 50 ms, retry
        end
    end
```

Details:

- **Re-entrant:** only the outermost `with` takes the file. Threads of one
  process wait on an in-process lock first.
- **Stale locks:** a lock older than `STALE_AFTER_S` (120 s) is taken over.
  Its age is the lock file time against the server clock (`server_now`),
  so machines whose clocks disagree agree on it. Holds are meant to last
  well under that. A process that took longer and
  finds its lock taken over logs an error and asks for an integrity check.
- **Cost:** taking and releasing the lock is four or five round trips.
  On the lab network that was about 0.3 s per take, before any query.
- **Schema check:** each outermost take also checks `PRAGMA user_version`,
  so a process left open across a migration stops instead of writing
  through the old schema.
- `db.held()` holds the lock across several reads or writes, so the batch
  pays for it once. Everyone else waits meanwhile, so keep it short.
- Programs that do not know about the lock (DB Browser, R) must read a
  copy: `python -m odyn.admin <main_folder> [--project <name>] snapshot`
  writes one to a `snapshots` folder beside the database.

## Reading: the caches

Each table is read whole into a `DataFrame` and kept until something may
have changed it.

```mermaid
flowchart TB
    read["db.some_table"] --> frozen{"inside db.frozen()<br/>and cached?"}
    frozen -- yes --> hit["cached DataFrame<br/>(no lock, no query)"]
    frozen -- no --> take["take the lock"]
    take --> version{"PRAGMA data_version<br/>changed since last look?"}
    version -- yes --> empty["empty every cache"] --> cached
    version -- no --> cached{"cached?"}
    cached -- yes --> hit2["cached DataFrame"]
    cached -- no --> query["read the table"] --> hit2
```

What empties the caches:

- **Another connection's commit.** `PRAGMA data_version` changes when any
  other connection has committed. Any commit, to any table, empties every
  cache.
- **This connection's own writes.** Each writer (`_writing`) empties the
  caches when it ends. A recorded call drops the `method_calls` views when
  it starts and when it ends.

`db.calls` is `method_calls` without its long columns (the code and the
environment), for readers that only need who called what and when.

### Reading a view: `db.frozen()`

A page that shows one view reads many tables, often while cluster jobs
commit. Without care, each commit empties the caches mid-view and the next
table read starts over.

```python
with db.frozen():
    picks, approvals = latest_picks(db), latest_approvals(db)
```

Inside `frozen()`, the database is checked once on the way in. After that,
a cached table is returned without the lock or a query. A table first read
inside the block is read normally, at that moment. Commits by others take
effect on the next `frozen()` or normal read. Writes made inside still
empty the caches.

This makes a view consistent per table, not a true snapshot: two tables
read at different moments inside the block can disagree, if a job commits
in between. A true snapshot would need one SQLite read transaction across
the block, which holds back every writer on the share for as long as the
view takes to build.

## Writing: recorded calls

Most writes happen inside methods decorated with `@record_call`. Each one
leaves a row in `method_calls`: who, when, the code version, the inputs,
the parameters used, the flag and the output.

```mermaid
sequenceDiagram
    participant C as Caller
    participant R as record_call
    participant D as Database (share)
    participant L as Log file (share)
    C->>R: group.run_motion_correction(...)
    R->>D: INSERT INTO method_calls (ended_at NULL)
    R->>L: open <call id>.log
    loop while the method runs
        R->>L: log lines (saved every 5 s)
        R->>D: the method's own writes
    end
    R->>L: save and close
    R->>D: UPDATE flag, output, parameters used, ended_at
    R-->>C: result (or the exception)
```

- A call whose `ended_at` is NULL is still running, or its process died.
  A process killed by SLURM (a time limit, out of memory) never reaches
  the end, so its row stays open. The cluster tick records how each job
  ended.
- Calls nest. An inner call has its own row and its own log; its lines are
  in the outer call's log too.

### Call logs

Each call's log is a file in the `.odyn` folder, named like the backups
(`main` for the main database):

```
.odyn/logs/<project or main>/<call id // 1000>/<call id>.log
```

`db.call_log(call_id)` prints one, and `db.call_log_file(call_id)` gives
its path. The rules:

- The file is opened when the call starts and saved every
  `CALL_LOG_FLUSH_S` (5 s) and at the end. A killed job keeps all but its
  last few seconds.
- Lines are plain: the time, the level, the message. No terminal colors.
- Files are the same bytes from every OS: UTF-8, with `\n` line ends.
  Python would otherwise write them on Windows in the local code page
  (which cannot hold marks like ✔) and with `\r\n` line ends.
- A call's log takes the lines of its own thread, and of threads that run
  no call of their own (helpers it started). It never takes another call's
  lines, and nothing from another process: a worker forked by caiman
  inherits the handler, and its copy of the unsaved lines would be written
  twice.
- If the file cannot be opened, the log is kept in memory and tried once
  more at the end, with a warning. The call goes on either way.
- One folder per thousand calls, since a folder of many thousand files is
  slow to list on a share.

On the cluster, the job's own output file (`.out`) has these lines too,
plus what happens outside a call: imports, the job header, caiman printing
to the terminal, and the traceback of a job that crashed.

The logs were a column (`call_log`) until schema v7. They were half the
database, and since they came before other columns in each row, every read
of `method_calls` walked their pages. SQLite reads a page (4 KB) at a time,
and on a share each page is a round trip.

## Migrations

The schema version is `PRAGMA user_version`, checked against
`SCHEMA_VERSION` in `odyn/migrate.py` whenever the lock is taken.

- Code older than the database refuses to open it ("Pull the latest
  version"). Code newer than the database refuses too, and says how to
  migrate. So **every machine pulls the new code, then one person
  migrates.** Until the migration, the new code does not run; after it,
  the old code does not.
- `python -m odyn.admin <main_folder> --project <name> migrate`, run
  while the lock is held throughout:
  1. Refuses while calls started in the last two days have not ended
     (`--force` for calls known to be dead).
  2. Checks integrity, and backs up the database to `.odyn/backups/`.
  3. Runs `BEFORE_SQL` when the version needs Python too (v7: writes each
     log to its file and reads it back).
  4. Runs `latest.sql` and sets the version, in one transaction.
  5. Checks integrity and foreign keys.
  6. Runs `VACUUM` when `VACUUM_AFTER` is set, to give back the room of
     dropped data. This rewrites the whole file and needs room for a copy.
- A version can be a rebuild instead (`MIGRATES_FROM = None`): a new
  database is started, and the old one stays readable with older code.
- `tests/test_migration.py` checks that an old database walked forward has
  exactly the schema of a new one.

## Files as messages (planned)

> Agreed in November 2026, not built yet. The QC page is the first user.

The QC page shows a few KB: four short tables and the files to download.
They change only when someone makes a QC decision or a cluster job ends.
Yet a load rebuilt them from about 285 folder listings, 95 file reads and
several database reads (about a minute over a VPN), only to learn whether
anything had changed. There is no server to ask, so the answer has to be a
file that is already computed. This section generalizes that into one
model for every file the programs share.

### Records and views

- **Records** are written once and are the source of truth: the database
  (decisions, provenance, history), the cluster's test and sweep records,
  requests, queued drafts, and claims (below).
- **Views** are built from records, by one function, and can always be
  rebuilt: `qc.json`, the heartbeat, the job states the tick writes. A view
  can be stale, never wrong for long. Deleting one is always safe.

Decisions and provenance stay in the database: they need transactions and
history. Files carry messages between programs that never run at the same
time, and views for pages that only read.

### Rules for every shared file

- **Written whole, then renamed** over the old one. A reader sees the old
  file or the new one, never half of each. So **readers take no lock**.
- **A lock per file when it has more than one writer:** the database lock,
  generalized to `file_lock(path)` (`odyn/locking.py`), a `<file>.lock`
  beside it. One lock object per file in each process (two would wait for
  each other).
- **A rename is retried** for a moment when it fails: Windows refuses to
  replace a file another program has open, and readers hold one for
  milliseconds.
- **Versioned JSON:** a reader that finds another version falls back
  instead of misreading it.
- **Times come from the storage server:** a file's modification time is
  set by the server that keeps the share, so files written from every
  machine share one clock. Files are created, never copied in with their
  times kept, and a program that needs the server's "now" writes a small
  file and reads its time back (`server_now` in `odyn/shared.py`). No
  machine's own clock is compared with another's.
- **One module** reads and writes all of them (`odyn/shared.py`:
  `read_json`, `write_json`), with one set of tests.

### The inbox

Programs ask the cluster for work by dropping a file in a folder the tick
works through, oldest first, moving each to `done/`, or to `failed/` with
the reason. Three kinds, one mechanism (`Inbox` in `odyn/shared.py`; the
queue and the requests use it, claims are planned). A message keeps its
own small files beside it, moved with it (a submitted draft keeps its
`.state.json`, since adding it can wait several ticks for its TIFFs).
All are in `.odyn/cluster/<project or main>/`:

| Folder | Written by | Asks for |
|---|---|---|
| `submissions/` | the app, on submit | adding a session's experiments |
| `requests/` | the app, scripts | a recorded call on the cluster (sync, previews) |
| `claims/` | the app, on a QC decision | recording that decision, and what follows it |

### QC decisions: claims

A decision is a claim file, one per group and step, created with the same
exclusive create the lock uses:

```
.odyn/cluster/<project>/claims/g42-approve.json
    {"user": "alice", "step": "approve", "excluded": [3, 7]}
```

```mermaid
sequenceDiagram
    participant A as App (alice)
    participant B as App (bob)
    participant S as Share
    participant T as Tick
    A->>S: create claims/g42-approve.json (exclusive)
    B->>S: list claims/ (next load)
    S-->>B: g42 approved by alice, waiting for the cluster
    B->>S: create claims/g42-approve.json (exclusive)
    S-->>B: exists: alice approved it at 11:20
    T->>S: claims, oldest first
    T->>S: check it in the database, record the call as alice
    T->>S: start what follows (a whole correction after a pick)
    T->>S: claim to done/ (or failed/ with the reason)
    T->>S: rebuild qc.json
```

- **The race is settled when the file is written:** a second claim on the
  same step fails at once and says who made the first. No later process
  has to choose between two decisions.
- **Steps** are the raw check, the pick, the approval and running again.
  The claim's time, from the server, orders different steps of one group.
- **Undo before the tick** is deleting one's own claim.
- **What waits for the tick:** the database record, its checks, and the
  next steps (submitting jobs, moving files). The app checks what it can
  from `qc.json` before writing the claim, so a claim that fails later is
  rare. Its author sees why on the page, as with a failed draft.

### The QC view: `qc.json`

`.odyn/cluster/<project>/qc.json` holds everything the QC page shows, for
every user: each experiment still in review (owner, paths, status, raw
check, latest pick and approval, its runs with their state and numbers),
the files of each downloadable folder with their sizes and times, the
recordings, the missing annotations, the queue, and when it was built.
Approved experiments keep a status line only.

- **Written only on the cluster,** by one function, `qc_summary(db)`: in
  full by each tick, and again as each test or sweep job ends. Listing the
  share there is quick, and `squeue` is at hand. Writers take its lock;
  readers do not.
- **Read by the app** off the database thread: the file's time (one round
  trip) on each load and every ~30 s while the page is on screen; the file
  itself only when that time changed; the list of `claims/` and any new
  claim. Two to five round trips, a few KB.
- **Computed in the app:** "Show all users" (a filter), and whether each
  folder is downloaded and how much is left (the file lists against the
  local copy, on this disk).
- **Shown with its age** ("as of 11:20"). When the file is missing or of
  another version, the page offers to build it here, with the same
  function, which takes as long as a load did before.

| | Before | Planned |
|---|---|---|
| Folder listings per load | ~285 | 1 (`claims/`) |
| File reads per load | ~95 | 0 to 2, plus new claims |
| Database reads per load | 1 to 9 lock takes | none |
| A QC decision | a recorded call from the app | one file created |
| Others see a decision | after their next reload | on their next load, or within ~30 s |
| A finished run shows | on the next reload | when the job ends (by the tick at the latest) |

## Assumptions

The design is reasonable because of how the lab works. These are the
assumptions, and what depends on each.

| Assumption | What depends on it |
|---|---|
| A few people and a few dozen cluster jobs use one database at a time. | One lock for the whole file. Waiting is short because holds are short and few. |
| Each hold of the lock lasts well under 120 s. | Stale-lock takeover. A hold longer than that can be taken over and lead to corruption. |
| Round trips cost more than bytes. | Whole-table reads into caches. Holding the lock once for many reads. |
| The database stays small (about 10 MB). | Reading whole tables. Copying the whole file for backups and snapshots. |
| People mostly work on different experiments, and care about the end result of a call, not its progress. | Views may be one reload old. `frozen()` ignores others' commits for the length of a view. |
| Writes are mostly new records (calls, picks, approvals), rarely edits of the same row by two people. | No conflict detection: the latest record wins. |
| Everything that writes runs ODyn's code. | The lock and the schema check protect the file only from programs that take them. |
| The share keeps a file created with `O_CREAT \| O_EXCL` exclusive across clients. | The lock itself, and claims. Some old NFS setups do not. |
| QC decisions and finished runs can take up to a tick (30 min by default) to be recorded or shown. Most QC happens a day or more after the experiment. | Claims recorded by the tick; `qc.json` rebuilt by the cluster only. |
| The storage server sets file times. | Stale-lock ages; ordering claims; file ages, with no machine's own clock compared with another's. |

## Pitfalls

- **Any commit empties every cache.** A cluster job finishing a call makes
  the next read of `acquisitions` start over, though it did not change.
- **A view in `frozen()` is not a snapshot.** See above.
- **Decisions on stale data.** Two people looking at the same group can
  both act on a view that is one reload old. Picks and approvals are
  records, and the latest one wins.
- **Holding the lock during slow work** (listing folders, reading TIFFs)
  holds back every other process, and past 120 s risks a takeover. Read
  what is needed, release, then work.
- **Killed jobs** leave calls with no `ended_at`. Migrations refuse while
  recent ones exist.
- **Copies, not the file:** pointing another program at the database
  bypasses the lock. Use a snapshot.
- **Logs and backups:** a backup copies the database, not the log files.
  The logs stay in `.odyn/logs`.
- **Clocks:** times in the database (`called_at`, `ended_at`) are written
  by each machine's clock (`localtime`), so they agree only as well as the
  clocks do. The lock and the planned files use the storage server's times
  instead (see above).
- **A view no one rebuilds** (planned files): a writer that forgets to
  rebuild `qc.json` leaves the page stale until the next tick, silently.
  Each writer needs a test that its change shows in the view.
- **The tick as the hub** (planned files): if the cluster stops, claims
  wait and the view ages. The page shows its age, and the cluster line
  warns when ticks stop.

## When the assumptions change

| If... | Then... |
|---|---|
| Many more users or jobs at once | Lock waits grow. Consider a database server (PostgreSQL) instead of a file on a share. The `Database` class is the only place that talks to SQLite. |
| Holds must last longer (large writes) | Raise `STALE_AFTER_S`, or split the writes into short transactions. Never let a hold run past it. |
| Several people work on the same experiments at once | Claims already refuse a second decision on the same step. Also make the tick refuse a claim made on a view older than the latest decision. |
| Decisions must be recorded at once | Have the app record the call after writing the claim, as it did before, and leave the tick only the next steps. |
| The cluster is not always there | Build `qc.json` in the app too (`qc_summary` is the same function), on a timer, under its lock. |
| The database grows large (hundreds of MB) | Whole-table reads and whole-file copies get slow. Read only the rows a view needs (per group, or newer than the last read), and keep large text in files, as the logs are. |
| Commits become very frequent | Empty caches per table or per group instead of all at once: a small table of counters, bumped by each write, read in one query (see below). |
| The share is far (a VPN, another site) | Read from a local copy of the database, refreshed when it changed (see below). |
| Something that does not run ODyn writes to the database | It must take the same lock file, or write only to a copy. |
| The share cannot keep `O_EXCL` exclusive | The lock needs another mechanism (a lock server, or one machine that owns the file and serves the rest). |

## Possible next steps

These follow the same assumptions and are not built yet.

- **A local copy for reading.** When the database changed, copy the whole
  file to local disk under the lock, then read everything from the copy.
  That is one sequential read instead of a page at a time, and a true
  snapshot for every view. It depends on the database staying small.
- **Counters of finished results,** for database readers that remain
  (notebooks, the registry pages): a small table holds a counter for the
  whole database and one per group, bumped by writes outside calls at
  once and by calls when they end. Readers compare them instead of
  `data_version`, so progress written by running calls never empties
  their caches. Less needed once pages read views.
