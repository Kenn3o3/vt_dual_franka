import numpy as np

from dp.model.common.normalizer import SingleFieldLinearNormalizer


def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_range_symmetric_normalizer_from_stat(
    stat, output_max=1, output_min=-1, range_eps=1e-7
):
    input_max = stat["max"]
    input_min = stat["min"]
    abs_max = np.max([np.abs(stat["max"][:3]), np.abs(stat["min"][:3])])
    input_max[:3] = abs_max
    input_min[:3] = -abs_max
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_image_range_normalizer():
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_image_identity_normalizer():
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat["min"])
    offset = np.zeros_like(stat["min"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def _concat_param_info(items):
    params = [x[0] for x in items]
    infos = [x[1] for x in items]
    param = {k: np.concatenate([p[k] for p in params], axis=-1) for k in params[0]}
    info = {k: np.concatenate([i[k] for i in infos], axis=-1) for k in infos[0]}
    return param, info


def _identity_info_like(stat):
    example = stat["max"]
    return (
        {"scale": np.ones_like(example), "offset": np.zeros_like(example)},
        {
            "max": np.ones_like(example),
            "min": np.full_like(example, -1),
            "mean": np.zeros_like(example),
            "std": np.ones_like(example),
        },
    )


def _range_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    return {"scale": scale, "offset": offset}, stat


def abs_action_only_normalizer_from_stat(stat):
    pos_stat = {k: np.copy(v[..., :3]) for k, v in stat.items()}
    rot_stat = {k: np.copy(v[..., 3:-1]) for k, v in stat.items()}
    gripper_stat = {k: np.copy(v[..., -1:]) for k, v in stat.items()}
    param, info = _concat_param_info(
        [_range_info(pos_stat), _identity_info_like(rot_stat), _range_info(gripper_stat)]
    )
    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def abs_action_only_symmetric_normalizer_from_stat(stat):
    pos_stat = {k: np.copy(v[..., :3]) for k, v in stat.items()}
    rot_stat = {k: np.copy(v[..., 3:-1]) for k, v in stat.items()}
    gripper_stat = {k: np.copy(v[..., -1:]) for k, v in stat.items()}
    abs_max = np.max([np.abs(pos_stat["max"][:3]), np.abs(pos_stat["min"][:3])])
    pos_stat["max"][:3] = abs_max
    pos_stat["min"][:3] = -abs_max
    param, info = _concat_param_info(
        [_range_info(pos_stat), _identity_info_like(rot_stat), _range_info(gripper_stat)]
    )
    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def abs_action_only_so2_symmetric_normalizer_from_stat(stat):
    pos_stat = {k: np.copy(v[..., :3]) for k, v in stat.items()}
    rot_stat = {k: np.copy(v[..., 3:-1]) for k, v in stat.items()}
    gripper_stat = {k: np.copy(v[..., -1:]) for k, v in stat.items()}
    abs_max = np.max([np.abs(pos_stat["max"][:2]), np.abs(pos_stat["min"][:2])])
    pos_stat["max"][:2] = abs_max
    pos_stat["min"][:2] = -abs_max
    param, info = _concat_param_info(
        [_range_info(pos_stat), _identity_info_like(rot_stat), _range_info(gripper_stat)]
    )
    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def array_to_stats(arr: np.ndarray):
    return {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }
