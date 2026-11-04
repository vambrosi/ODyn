"""
What other computers leave for the cluster: requests to run a recorded
call, and QC claims (`odyn.claims`), both applied by the tick.
"""

from __future__ import annotations

import inspect
import socket
import uuid

from datetime import datetime
from pathlib import Path
from typing import Any

from ..database import Database
from ..shared import DONE_FOLDER, FAILED_FOLDER, Inbox, read_json, write_json
from ..utils import get_user, logger
from .config import REQUESTS_FOLDER, TIME_FORMAT, TickReport, cluster_folder
from .records import pick_candidate

# What a request may ask for. A request is executed by name, so this list is
# the whole of what another computer can make the cluster do.
REQUESTABLE = ("process_sync",)


def send_request(
    main_folder: str | Path,
    method: str,
    *,
    project: None | str = None,
    user: None | str = None,
    **kwargs: Any,
) -> Path:
    """
    Ask the cluster to run `Database.<method>(**kwargs)` at its next tick, as
    `user` (the computer login if left out). Returns the request file.

    **EXAMPLE**
    ```python
    send_request(main_folder, "process_sync", project="main-sync", exp_id=12)
    ```

    Raises `ValueError` for a method that cannot be requested (see
    `REQUESTABLE`) or arguments it does not take, so a bad request is refused
    here rather than at the tick.
    """
    _check_request(method, kwargs)
    request = {
        "method": method,
        "kwargs": kwargs,
        "user": user or get_user(),
        "requested_at": datetime.now().strftime(TIME_FORMAT),
        "requested_on": socket.gethostname(),
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    return requests(main_folder, project).put(
        f"{stamp}-{method}-{uuid.uuid4().hex[:8]}", request
    )


def requests(main_folder: str | Path, project: None | str = None) -> Inbox:
    """Where the requests to the cluster wait (`requests/`)."""
    return Inbox(cluster_folder(main_folder, project) / REQUESTS_FOLDER)


def pending_requests(main_folder: str | Path, project: None | str = None) -> list[Path]:
    """The requests waiting for a tick, oldest first."""
    return requests(main_folder, project).items()


def _check_request(method: str, kwargs: dict[str, Any]) -> None:
    if method not in REQUESTABLE:
        raise ValueError(
            f"'{method}' cannot be requested. Requests can ask for "
            f"{', '.join(REQUESTABLE)}."
        )

    try:
        inspect.signature(getattr(Database, method)).bind(None, **kwargs)
    except TypeError as error:
        raise ValueError(f"Wrong arguments for '{method}' ({error}).") from None


def _apply(
    path: Path, main_folder: Path, project: None | str, report: TickReport
) -> None:
    """Run one request, then move it to `done/` or `failed/`."""
    try:
        request = read_json(path)
        method, kwargs = request["method"], dict(request.get("kwargs") or {})
        user = str(request["user"])
        _check_request(method, kwargs)

    except (OSError, ValueError, KeyError, TypeError) as error:
        Inbox(path.parent).file(path, FAILED_FOLDER, {"reason": str(error)})
        report.rejected.append(f"{path.name}: {error}")
        logger.warning(f"Rejected the request '{path.name}': {error}")
        return

    # Recorded as the person who asked, not as the account running the tick
    db = Database(main_folder, project=project, user=user)
    outcome: dict[str, Any] = {"applied_at": datetime.now().strftime(TIME_FORMAT)}

    try:
        getattr(db, method)(**kwargs)
        outcome["error"] = None

    except Exception as error:  # the call recorded it; the request is still done
        outcome["error"] = f"{type(error).__name__}: {error}"
        report.errors.append(f"{path.name}: {outcome['error']}")

    Inbox(path.parent).file(path, DONE_FOLDER, outcome)
    report.applied.append(f"{method}({_arguments(kwargs)}) for {user}")


def _arguments(kwargs: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in kwargs.items())


def apply_claims(main_folder: Path, project: None | str, report: TickReport) -> None:
    """
    Record each QC decision claimed in the app (`odyn.claims`), oldest first,
    as the person who made it, then file it in `claims/done/`. A decision
    that no longer applies (a pick while the last whole correction still
    runs, say) stays, with why, for its author to see.
    """
    from ..claims import claims  # it imports this module

    inbox = claims(main_folder, project)
    stamp = datetime.now().strftime(TIME_FORMAT)

    def age(path):
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    for path in sorted(inbox.items(), key=age):
        try:
            content = read_json(path)
            group_id, step = int(content["group_id"]), str(content["step"])
            user = str(content["user"])
        except (OSError, ValueError, KeyError, TypeError):
            continue  # being written: the next tick

        if content.get("failed"):
            continue  # until its author decides again

        db = Database(main_folder, project=project, user=user)
        group = db.groups[group_id]

        try:
            if step == "raw":
                group.check_raw(exclude_acq_ids=content.get("excluded") or [])
            elif step == "approve":
                group.approve_mcor_files(exclude_acq_ids=content.get("excluded") or [])
            elif step == "again":
                group.run_again(runs=content["runs"])
            elif step == "pick":
                pick_candidate(
                    db,
                    group_id,
                    content["run"],
                    main_folder / content["folder"],
                    content.get("set_aside") or [],
                )
            else:
                raise ValueError(f"Unknown QC step {step!r}")

        except Exception as error:  # refused: kept, with why, for its author
            reason = str(error) or type(error).__name__
            write_json(path, {**content, "failed": reason, "failed_at": stamp})
            report.rejected.append(f"{path.name}: {reason}")
            logger.warning(
                f"The {step} of group {group_id} for {user} failed: {reason}"
            )
            continue

        done = f"{path.stem}-{datetime.now():%Y%m%d-%H%M%S}"  # one per decision
        inbox.file(path, DONE_FOLDER, {"applied_at": stamp}, name=done)
        report.claims.append(f"{step} of group {group_id} for {user}")
