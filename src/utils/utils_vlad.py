import pickle
import inspect
import sys
import subprocess
import platform
import logging
import string
import random
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import (
    Callable,
    Union,
    TypeVar,
    Any,
    Dict,
    Iterable,
    Optional,
    TypedDict,
    List,
    Tuple,
)
import concurrent.futures
import re
import time


import git
import numpy as np
from tqdm import tqdm


from .utils import mkdir

log = logging.getLogger(__name__)

reasonable_formatters = {
    "extended": logging.Formatter(
        "%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ),
    "short": logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    ),
    "shorter": logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    ),
    "shortest": logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"),
}


class CaptureLogRecordsHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.captured_records = []

    def emit(self, record):
        self.captured_records.append(record)

    def close(self):
        logging.Handler.close(self)


class LogCaptorToRecords(object):
    def __init__(self, pause_others=False):
        self.pause_others = pause_others
        self._logger = logging.getLogger()
        self._captor_handler = CaptureLogRecordsHandler()
        self.captured = []

    def _pause_other_handlers(self):
        self._other_handlers = self._logger.handlers.copy()
        for handle in self._logger.handlers:
            self._logger.removeHandler(handle)

    def _unpause_other_handlers(self):
        for handle in self._other_handlers:
            self._logger.addHandler(handle)

    def __enter__(self):
        if self.pause_others:
            self._pause_other_handlers()
        self._logger.addHandler(self._captor_handler)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.pause_others:
            self._unpause_other_handlers()
        self._logger.removeHandler(self._captor_handler)
        self.captured = self._captor_handler.captured_records[:]
        # If exception was raise - handle captured right now
        if exc_type is not None:
            log.error(
                "<<(CAPTURED BEGIN)>> Capturer encountered an "
                "exception and released captured records"
            )
            self.handle_captured()
            log.error("<<(CAPTURED END)>> End of captured records")

    def handle_captured(self):
        for record in self.captured:
            self._logger.handle(record)


def reasonable_logging_setup(
    stream_loglevel: int,
    formatter="extended",
    spammy_packages=[  # Not allowed at DEBUG loglevel
        "PIL",
        "git",
        "tensorflow",
        "matplotlib",
        "selenium",
        "urllib3",
        "rasterio",
    ],
):
    """Create STDOUT stream handler, curtail spam"""
    if isinstance(formatter, str):
        formatter = reasonable_formatters[formatter]
    # Get root logger (with NOTSET level)
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    # Stream handler takes 'loglevel'
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(stream_loglevel)
    logger.addHandler(handler)
    # Prevent some spammy packages from exceeding INFO verbosity
    for packagename in spammy_packages:
        logging.getLogger(packagename).setLevel(max(logging.INFO, stream_loglevel))
    return logger


def add_filehandler(logfilename, level=logging.DEBUG, formatter="extended"):
    if isinstance(formatter, str):
        formatter = reasonable_formatters[formatter]
    out_filehandler = logging.FileHandler(str(logfilename))
    out_filehandler.setFormatter(formatter)
    out_filehandler.setLevel(level)
    logging.getLogger().addHandler(out_filehandler)
    return logfilename


def add_logging_filehandlers(workfolder):
    # Create two output files in /_log subfolder, start loggign
    assert isinstance(
        logging.getLogger().handlers[0], logging.StreamHandler
    ), "First handler should be StreamHandler"
    logfolder = mkdir(workfolder / "_log")
    id_string = get_experiment_id_string()
    logfilename_debug = add_filehandler(
        logfolder / f"{id_string}.DEBUG.log", logging.DEBUG, "extended"
    )
    logfilename_info = add_filehandler(
        logfolder / f"{id_string}.INFO.log", logging.INFO, "short"
    )
    return logfilename_debug, logfilename_info


def get_experiment_id_string(N=2):
    time_now = datetime.now()
    str_time = time_now.strftime("%Y-%m-%d_%H-%M-%S")
    str_ms = time_now.strftime("%f")
    str_rnd = str_ms[:N] + "".join(random.choices(string.ascii_uppercase, k=N))
    str_node = platform.node()
    return f"{str_time}_{str_rnd}_{str_node}"


def platform_info():
    platform_string = f"Node: {platform.node()}"
    oar_jid = (
        subprocess.run("echo $OAR_JOB_ID", shell=True, stdout=subprocess.PIPE)
        .stdout.decode()
        .strip()
    )
    platform_string += " OAR_JOB_ID: {}".format(oar_jid if len(oar_jid) else "None")
    platform_string += f" System: {platform.system()} {platform.version()}"
    return platform_string


def is_venv():
    # https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def release_logcaptor_to_folder(lctr, p_exp_fold):
    # Create logfiles
    logfilename_debug, logfilename_info = add_logging_filehandlers(p_exp_fold)
    log.info(
        inspect.cleandoc(
            f"""Initialized the logging system!
        Platform: \t\t{platform_info()}
        Workfolder path: \t{p_exp_fold}
        --- Python --
        VENV:\t\t\t{is_venv()}
        Prefix:\t\t\t{sys.prefix}
        --- Code ---
        Experiment: \t\t{sys.argv}
        -- Logging --
        DEBUG logfile: \t\t{logfilename_debug}
        INFO logfile: \t\t{logfilename_info}
        """
        )
    )
    from pip._internal.operations import freeze

    log.debug("pip freeze: {}".format(";".join(freeze.freeze())))
    # Release previously captured logging records
    log.info("\n- [ CAPTURED: Loglines before system init")
    lctr.handle_captured()
    log.info("\n- ] CAPTURED\n")


def git_repo_query_v2(code_root: Path) -> git.Repo:
    # Try to get repo object
    try:
        repo = git.Repo(str(code_root))
    except git.exc.InvalidGitRepositoryError:
        log.warning("No git repo found")
        return None
    # Try to get branch name
    try:
        branch_name = repo.active_branch.name
    except TypeError as e:
        if repo.head.is_detached:
            branch_name = "DETACHED_HEAD"
        else:
            log.warning("Could not get git branch")
            log.exception(e)
            branch_name = "UNKNOWN_BRANCH"
    # Try to get current commit
    try:
        commit_sha = repo.head.commit.hexsha
        short_sha = repo.git.rev_parse(commit_sha, short=8)
        summary = repo.head.commit.summary
    except ValueError as e:
        if len(list(repo.iter_commits("--all"))) == 0:
            log.warning("No commits in this git repo")
        else:
            log.warning("Could not get commit info")
            log.exception(e)
        commit_sha = "UNKNOWN_SHA"
        short_sha = "NOSHA"
        summary = "UNKNOWN_SUMMARY"
    log.info(
        "Git repo found [branch {}, Commit {}({})]".format(
            branch_name, commit_sha, summary
        )
    )
    # Check if repo is dirty and log the diff
    dirty = repo.is_dirty()
    if dirty:
        dirty_diff = repo.git.diff()
        log.info("Repo is dirty")
        log.info(f"Dirty repo diff:\n===\n{dirty_diff}\n===")
        short_sha = "DIRTY" + short_sha

    # Commit ID
    return repo, short_sha


T = TypeVar("T")


def compute_or_load_pkl_silently(
    filepath: Union[str, Path], function: Callable[..., T], *args, **kwargs
) -> T:
    """Implementation without outputs"""
    try:
        with Path(filepath).open("rb") as f:
            pkl = pickle.load(f)
    except (EOFError, FileNotFoundError):
        pkl = function(*args, **kwargs)
        with Path(filepath).open("wb") as f:
            pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)
    return pkl


def stash2(stash_to) -> Callable:
    def stash_func(function, *args, **kwargs):
        return compute_or_load_pkl_silently(stash_to, function, *args, **kwargs)

    return stash_func


def save_pkl(filepath, obj):
    with Path(filepath).open("wb") as f:
        pickle.dump(obj, f)


def load_pkl(filepath):
    with Path(filepath).resolve().open("rb") as f:
        obj = pickle.load(f)
    return obj


# ISAVE


"""
SSLICES (numpy-like strided slices) spec.

Spec:
- SSLICES:  + separated SSLICEs
- SSLICE: '(inds)?:(ilimit)?:(period)?'
  - inds: 'csv list' Enumeration of indices
  - ilimit: '(MIN)?,(MAX)?'. Inclusive limit on period, "don't fire beyond"
  - period: 'int' (fire at period intervals)

Examples:
    '' or '::' - empty.
    '::5' - every 5 iters. (5, 10, 15...)
    '0::5' - 0 and every 5 iters. (0, 5, 10, 15...)
    '0:,20:5' - same, but stop at 20 inclusive. (0, 5, 10, 15, 20)
    '0,1:10,:5' - 0, 1 and every 5 starting from 10 (0, 1, 10, 15, 20...)
    ':,5:1+::25' - Union of ':,5:1' and '::25'. (0, 1, 2, 3, 4, 5, 25, 50, 75...)
"""


class SSLICE(TypedDict):
    inds: List[int]
    ilimit: Optional[Tuple[Optional[int], Optional[int]]]
    period: Optional[int]


def _parse_sslice_spec(sslice_str: str) -> SSLICE:
    """Parse SSLICE spec"""
    inds, ilimit, period = [], None, None  # type: ignore
    if not len(sslice_str):
        return SSLICE(inds=inds, ilimit=ilimit, period=period)
    spec_re = r"^([\d,]*):((?:\d*,\d*)?):([\d]*)$"
    match = re.fullmatch(spec_re, sslice_str)
    if match is None:
        raise ValueError(f"Invalid spec {sslice_str}")
    _inds, _ilimit, _period = match.groups()
    if _inds:
        inds = list(map(int, _inds.split(",")))
    if _ilimit:
        ilimit = map(lambda x: int(x) if x else None, _ilimit.split(","))
    if _period:
        period = int(_period)
    return SSLICE(inds=inds, ilimit=ilimit, period=period)  # type: ignore


def _check_step_sslice(step: int, sslice_str: str) -> bool:
    """Check whether step matches SSLICE spec"""
    sslice = _parse_sslice_spec(sslice_str)
    if step in sslice["inds"]:
        return True
    if sslice["ilimit"] is not None:
        ilmin, ilmax = sslice["ilimit"]
        if ilmin is not None and step < ilmin:
            return False
        if ilmax is not None and step > ilmax:
            return False
    if sslice["period"] is not None:
        if step % sslice["period"] == 0:
            return True
    return False


def check_step(step: int, sslices_str: str) -> bool:
    """Check whether step matches SSLICES spec"""
    return any((_check_step_sslice(step, s) for s in sslices_str.split("+")))


def tqdm_str(pbar, ninc=0):
    if pbar is None:
        tqdm_str = ""
    else:
        tqdm_str = (
            "TQDM["
            + pbar.format_meter(pbar.n + ninc, pbar.total, pbar._time() - pbar.start_t)
            + "]"
        )
    return tqdm_str


class Counter_repeated_action(object):
    """
    Will check whether repeated action should be performed
    """

    def __init__(self, sslice="::", seconds=None, iters=None):
        self.sslice = sslice
        self.seconds = seconds
        self.iters = iters
        self.tic(-1)

    def tic(self, i=None):
        self._time_last = time.perf_counter()
        if i is not None:
            self._i_last = i

    def check(self, i=None):
        ACTION = False
        if i is not None:
            ACTION |= check_step(i, self.sslice)
            if self.iters is not None:
                ACTION |= (i - self._i_last) >= self.iters

        if self.seconds is not None:
            time_since_last = time.perf_counter() - self._time_last
            ACTION |= time_since_last >= self.seconds
        return ACTION


class Isaver_base0(ABC):
    def __init__(self, folder, total):
        self._re_finished = r"item_(?P<i>\d+)_of_(?P<N>\d+).finished"
        self._fmt_finished = "item_{:04d}_of_{:04d}.finished"
        self._history_size = 3

        self._folder = folder
        self._total = total
        if self._folder is None:
            log.debug("Isaver without folder, no saving will be performed")
        else:
            self._folder = mkdir(self._folder)

    def _get_filenames(self, i) -> Dict[str, Path]:
        if self._folder is None:
            raise RuntimeError("Filenames are undefined without folder")
        filenames = {
            "finished": self._folder / self._fmt_finished.format(i, self._total)
        }
        return filenames

    def _get_intermediate_files(self) -> Dict[int, Dict[str, Path]]:
        """Check re_finished, query existing filenames"""
        if (self._folder is None) or (not self._folder.exists()):
            return {}
        intermediate_files = {}
        for ffilename in self._folder.iterdir():
            matched = re.match(self._re_finished, ffilename.name)
            if matched:
                i = int(matched.groupdict()["i"])
                # Check if filenames exist
                filenames = self._get_filenames(i)
                all_exist = all([v.exists() for v in filenames.values()])
                assert ffilename == filenames["finished"], (
                    "Incompatible isaver tempfiles found."
                    "Probably remnants of previous run, kill them. "
                    "Found {} should be {}".format(
                        ffilename, filenames["finished"].name
                    )
                )
                if all_exist:
                    intermediate_files[i] = filenames
        return intermediate_files

    def _purge_intermediate_files(self) -> None:
        if self._folder is None:
            log.debug("Isaver folder is None, no purging")
            return
        """Remove old saved states"""
        intermediate_files: Dict[int, Dict[str, Path]] = self._get_intermediate_files()
        inds_to_purge = np.sort(np.fromiter(intermediate_files.keys(), np.int64))[
            : -self._history_size
        ]
        files_purged = 0
        for ind in inds_to_purge:
            filenames = intermediate_files[ind]
            for filename in filenames.values():
                filename.unlink()
                files_purged += 1
        log.debug("Purged {} states, {} files".format(len(inds_to_purge), files_purged))


class Isaver_base(Isaver_base0):
    result: Any

    def __init__(self, folder, total):
        super().__init__(folder, total)

    def _get_filenames(self, i) -> Dict[str, Path]:
        filenames = super()._get_filenames(i)
        filenames["pkl"] = filenames["finished"].with_suffix(".pkl")
        return filenames

    def _restore(self) -> int:
        intermediate_files: Dict[int, Dict[str, Path]] = self._get_intermediate_files()
        start_i, ifiles = max(intermediate_files.items(), default=(-1, None))
        if ifiles is not None:
            restore_from = ifiles["pkl"]
            self.result = load_pkl(restore_from)
            log.debug("Restore from {}".format(restore_from))
        return start_i

    def _save(self, i):
        if self._folder is None:
            log.debug("Isaver folder is None, no saving")
            return
        ifiles = self._get_filenames(i)
        savepath = ifiles["pkl"]
        mkdir(savepath.parent)
        save_pkl(savepath, self.result)
        ifiles["finished"].touch()


class Isaver_simple(Isaver_base):
    """
    Execute *func* (in parallel) over *arg_list* arguments, with checkpoints

    - Legacy. Use Isaver_fast instead
    """

    def __init__(
        self,
        folder,
        arg_list: Iterable[Iterable[Any]],
        func: Callable,
        *,
        save_period="::",  # SSLICES
        save_interval=120,  # every 2 minutes by default
        progress: Optional[str] = None,
        log_interval=None,  # Works only if progress is defined
    ):
        arg_list = list(arg_list)
        super().__init__(folder, len(arg_list))
        self.arg_list = arg_list
        self.result = []
        self.func = func
        self._save_period = save_period
        self._save_interval = save_interval
        self._progress = progress
        self._log_interval = log_interval

    def run(self):
        start_i = self._restore()
        run_range = np.arange(start_i + 1, self._total)
        self._time_last_save = time.perf_counter()
        self._time_last_log = time.perf_counter()
        pbar = run_range
        if self._progress:
            pbar = tqdm(pbar, self._progress)
        for i in pbar:
            args = self.arg_list[i]
            self.result.append(self.func(*args))
            # Save check
            SAVE = check_step(i, self._save_period)
            if self._save_interval:
                since_last_save = time.perf_counter() - self._time_last_save
                SAVE |= since_last_save > self._save_interval
            SAVE |= i + 1 == self._total
            if SAVE:
                self._save(i)
                self._purge_intermediate_files()
                self._time_last_save = time.perf_counter()
            # Log check
            if self._progress and self._log_interval:
                since_last_log = time.perf_counter() - self._time_last_log
                if since_last_log > self._log_interval:
                    log.info(tqdm_str(pbar))
                    self._time_last_log = time.perf_counter()
        return self.result


class Isaver_fast(Isaver_base):
    """
    Execute *func* (in parallel) over *arg_list* arguments, with checkpoints
    """

    def __init__(
        self,
        folder: Optional[Path],
        arg_list: Iterable[Iterable[Any]],
        func: Callable,
        *,
        async_kind="thread",
        num_workers=None,
        save_iters=np.inf,
        save_interval=120,
        progress: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        arg_list = list(arg_list)
        super().__init__(folder, len(arg_list))
        self.arg_list = arg_list
        self.func = func
        self._async_kind = async_kind
        self._save_iters = save_iters
        self._save_interval = save_interval
        self._num_workers = num_workers
        self._progress = progress
        self._timeout = timeout
        self.result = {}

    def run(self):
        self._restore()
        countra = Counter_repeated_action(seconds=self._save_interval)

        all_ii = set(range(len(self.arg_list)))
        remaining_ii = all_ii - set(self.result.keys())

        flush_dict = {}

        def flush_purge():
            if not len(flush_dict):
                return
            self.result.update(flush_dict)
            flush_dict.clear()
            self._save(len(self.result))
            self._purge_intermediate_files()

        if self._num_workers == 0:
            # Run with zero threads, easy to debug
            pbar = remaining_ii
            if self._progress:
                pbar = tqdm(pbar, self._progress)
            for i in pbar:
                result = self.func(*self.arg_list[i])
                flush_dict[i] = result
                if countra.check() or len(flush_dict) >= self._save_iters:
                    flush_purge()
                    countra.tic()
        else:
            # Run asynchronously
            if self._async_kind == "thread":
                io_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._num_workers
                )
            elif self._async_kind == "process":
                io_executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self._num_workers
                )
            else:
                raise RuntimeError(f"Unknown {self._async_kind=}")
            io_futures = []
            for i in remaining_ii:
                args = self.arg_list[i]
                submitted = io_executor.submit(self.func, *args)
                submitted._i = i
                io_futures.append(submitted)
            pbar = concurrent.futures.as_completed(io_futures)
            if self._progress:
                pbar = tqdm(pbar, self._progress, total=len(io_futures))
            for io_future in pbar:
                result = io_future.result(timeout=self._timeout)
                i = io_future._i
                flush_dict[i] = result
                if countra.check() or len(flush_dict) >= self._save_iters:
                    flush_purge()
                    countra.tic()

        flush_purge()
        assert len(self.result) == len(self.arg_list)
        result_list = [self.result[i] for i in all_ii]
        return result_list


class Isaver_threading(Isaver_fast):
    """Present for backwards compatability, soon to be deprecated"""

    def __init__(
        self,
        folder,
        arg_list,
        func,
        *,
        max_workers=None,
        save_iters=np.inf,
        save_interval=120,
        progress=None,
        timeout=None,
    ):
        super().__init__(
            folder,
            arg_list,
            func,
            async_kind="thread",
            num_workers=max_workers,
            save_iters=save_iters,
            save_interval=save_interval,
            progress=progress,
            timeout=timeout,
        )


class Isaver_dataloader(Isaver_base):
    """
    Will process a list with a 'func',
    - prepare_func(start_i) is to be run before processing

    Example:
        def prepare_func(i_last):
            dset = creator_tdata_eval(negatives_cvt[i_last+1:])
            dload = creator_dload_eval(dset)
            return dload

        def func(data_input):
            data, target, meta = helper_metamodel.data_preprocess(data_input)
            data, target = map(lambda x: x.to(device), (data, target))
            with torch.no_grad():
                output = helper_metamodel.get_eval_output(data, meta)
            score_np = output.detach().cpu().numpy()
            inds = [x['item']['ind'] for x in meta]
            result_dict = {'score': score_np, 'ind': inds}
            i_last = negatives_inds.index(inds[-1])
            return result_dict, i_last
    """

    def __init__(
        self,
        folder,
        total,
        func,
        prepare_func,
        *,
        save_period="::",
        save_interval=120,
        log_interval=None,
        progress: Optional[str] = None,
    ):
        super().__init__(folder, total)
        self.func = func
        self.prepare_func = prepare_func
        self._save_period = save_period
        self._save_interval = save_interval
        self._log_interval = log_interval
        self._progress = progress
        self.result = []

    def run(self):
        i_last = self._restore()
        if i_last + 1 >= self._total:  # Avoid running with empty dataloader
            return self.result
        countra = Counter_repeated_action(
            sslice=self._save_period, seconds=self._save_interval
        )

        result_cache = []

        def flush_purge():
            if not len(result_cache):
                return
            self.result.extend(result_cache)
            result_cache.clear()
            self._save(i_last)
            self._purge_intermediate_files()

        loader = self.prepare_func(i_last)
        pbar = enumerate(loader)
        if self._progress:
            pbar = tqdm(pbar, self._progress, total=len(loader))
        for i_batch, data_input in pbar:
            result_dict, i_last = self.func(data_input)
            result_cache.append(result_dict)
            if countra.check(i_batch):
                flush_purge()
                if self._progress:
                    log.debug(tqdm_str(pbar))
                countra.tic(i_batch)
        flush_purge()
        return self.result
