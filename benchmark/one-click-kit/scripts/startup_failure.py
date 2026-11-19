"""Surface fatal owned-service startup logs while its HTTP parent is still alive."""

import re


def inspect(path):
    if not path.exists():
        return None
    lines = path.read_text(errors="replace").splitlines()
    fatal = re.compile(
        r"^\((Worker_TP\d+|EngineCore(?:_DP\d+)?) pid=(\d+)\) ERROR .*"
        r"\[[^\]]+\] (WorkerProc hit an exception\.|EngineCore failed to start\.)$"
    )
    for index, line in enumerate(lines):
        match = fatal.match(line)
        if not match:
            continue
        owner, pid = match.group(1, 2)
        prefix = f"({owner} pid={pid}) ERROR "
        error = None
        for following in lines[index + 1:index + 257]:
            if following.startswith(prefix):
                message = re.split(r"\[[^\]]+\] ", following, maxsplit=1)[-1]
                if re.match(r"^[\w.]+(?:Error|Exception): ", message):
                    error = message
                    break
        return dict(
            schema="vllm-hust-fatal-service-startup/v1",
            log=str(path), line=index + 1, owner=owner, pid=int(pid),
            fatal=line, root_error=error,
            scope="Owned model service before first formal request; raw log retained",
        )
    return None
