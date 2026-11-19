"""Version-bound launch geometry for the measured r8 combination."""
import hashlib
import json

RELEASE = 'full-answer-b1-20261117-r8'
CONFIG = dict(schema='vllm-hust-runtime-geometry/v1', id='r8-E36-R54',
              execution_seats=36, resident_seats=54, speculative_tokens=2,
              prefill_tokens=4096, host_bytes_per_rank=4 * 2**30)
BASE = (3, 6, 12, 16, 24, 32, 40, 48, 64, 128, 256, 512, 1024, 1536, 2048, 4096)
BINS = (3, 6, 12, 24, 40, 48, 80, 96, 108, 144, 180)


def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_profile(profile):
    config = profile.get('runtime_configuration')
    if profile['release'] != RELEASE:
        if config is not None or 'runtime_configuration_sha256' in profile:
            raise ValueError('Runtime geometry is bound to the r8 release')
        return None
    if (not isinstance(config, dict) or config != CONFIG
            or any(type(config[k]) is not type(v) for k, v in CONFIG.items())
            or profile.get('runtime_configuration_sha256') != digest(CONFIG)):
        raise ValueError('Missing, changed or unbound r8 runtime geometry')
    return config


def bind_profile(profile):
    if profile['release'] != RELEASE:
        raise ValueError('Only r8 owns this runtime configuration')
    profile['runtime_configuration'] = dict(CONFIG)
    profile['runtime_configuration_sha256'] = digest(CONFIG)
    validate_profile(profile)


def captures(execution):
    if type(execution) is not int or execution not in (24, 36):
        raise ValueError('Unsupported geometry')
    maximum = next(n for n in BINS if n >= execution * 3)
    return sorted(set(BASE) | {n for n in BINS if n <= maximum})


def apply(profile, command, env):
    config = validate_profile(profile)
    if config is None:
        return None

    def value(flag):
        if command.count(flag) != 1:
            raise ValueError('Ambiguous launch argument: ' + flag)
        return command[command.index(flag) + 1]

    if ('--enforce-eager' in command or env.get('MTP_TOKENS') != '2'
            or env.get('BETTERSCALE_CONTEXT_PARALLEL') != '1'
            or env.get('BETTERSCALE_STATE_CACHE_HOST_BYTES') != str(4 * 2**30)
            or value('--max-num-seqs') != '24' or value('--max-num-batched-tokens') != '4096'
            or json.loads(value('--speculative-config')) != {'method': 'mtp', 'num_speculative_tokens': 2}):
        raise ValueError('Original FULL/MTP2/CP1/budget/prefill baseline drift')
    extra = json.loads(value('--additional-config'))
    expected = dict(using_live_runtime=True, state_resident_seats=36, state_cache_host_bytes=4 * 2**30,
                    state_cache_incremental=True, state_cache_streaming=True,
                    state_cache_policy=True, state_cache_partial_reclaim=True)
    if any(type(extra.get(k)) is not type(v) or extra[k] != v for k, v in expected.items()):
        raise ValueError('Original State policy drift')
    graph = json.loads(value('--compilation-config'))
    if graph.get('cudagraph_mode') != 'FULL' or sorted(graph.get('cudagraph_capture_sizes', [])) != captures(24):
        raise ValueError('Original FULL graph baseline drift')
    before = list(command)
    extra['state_resident_seats'] = 54
    graph['cudagraph_capture_sizes'] = captures(36)
    for flag, new in (('--max-num-seqs', '36'), ('--additional-config', json.dumps(extra)),
                      ('--compilation-config', json.dumps(graph))):
        command[command.index(flag) + 1] = new
    return dict(config=config, config_sha256=digest(config), original_command=before,
                effective_command=list(command), source_study='C15-r7-tail-combination-model-20261117-r1')
